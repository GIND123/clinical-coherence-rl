"""Amazon Bedrock Converse client: one uniform request builder for every vendor.

Request policy (see config/models.yaml header): no temperature/top_p ever;
effort only through the strategy the probe verified; optional fields only if
the probe verified them; no fallback model. botocore's own retries are
disabled — retries are decided here, where billing consequences are known.

Error taxonomy -> attempt status (ledger):
  throttling / quota / not-ready / 5xx before inference  -> retried in place
                                                            with jittered backoff (unbilled)
  connection failure BEFORE the request was sent          -> retried in place (unbilled)
  read timeout / connection lost AFTER sending            -> 'uncertain' (may be billed)
  ModelTimeoutException, ModelErrorException              -> 'uncertain'
  ValidationException                                     -> 'unbilled_error', fatal for the model
  AccessDenied / credentials / not found                  -> 'unbilled_error', fatal for the run
  stopReason max_tokens                                   -> 'truncated' (billed)
  stopReason content_filtered / guardrail_intervened      -> 'refused' (billed)
"""
from __future__ import annotations

import base64
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from ccrl.bedrock import pricing

RETRY_IN_PLACE = {"ThrottlingException", "ServiceQuotaExceededException", "ModelNotReadyException",
                  "ServiceUnavailableException", "InternalServerException", "TooManyRequestsException",
                  "ServiceUnavailable", "InternalFailure", "RequestTimeout"}
UNCERTAIN_CODES = {"ModelTimeoutException", "ModelErrorException", "ModelStreamErrorException"}
FATAL_RUN = {"AccessDeniedException", "UnrecognizedClientException", "ExpiredTokenException",
             "InvalidSignatureException", "IncompleteSignature", "MissingAuthenticationToken"}
FATAL_MODEL = {"ValidationException", "ResourceNotFoundException"}


@dataclass
class CallSpec:
    """Everything model-specific about a request, frozen in models.lock.yaml."""
    model_key: str
    model_id: str
    price_key: str
    region: str = "us-east-1"
    effort_strategy: str = "none"
    service_tier: str = "default"
    prompt_cache: bool = False
    extra_fields: dict = field(default_factory=dict)
    max_tokens: dict = field(default_factory=lambda: {"default": 16384})
    read_timeout_s: int = 1200
    connect_timeout_s: int = 30
    concurrency: int = 4
    default_effort: str | None = None

    def resolve_effort(self, label: str) -> str | None:
        if self.effort_strategy == "none":
            return None
        return self.default_effort if label == "default" else label

    def max_tokens_for(self, effort: str | None) -> int:
        return int(self.max_tokens.get(effort or "default", self.max_tokens.get("default", 16384)))


@dataclass
class CallResult:
    status: str                       # ok | truncated | refused | uncertain | unbilled_error
    text: str = ""
    reasoning: str = ""
    usage: dict = field(default_factory=dict)
    stop_reason: str | None = None
    served_tier: str | None = None
    latency_ms: int | None = None
    response: dict | None = None
    error_code: str | None = None
    error_msg: str | None = None
    throttles: int = 0
    fatal: str | None = None          # None | 'model' | 'run'

    @property
    def billed(self) -> bool:
        return self.status in ("ok", "truncated", "refused")


def build_request(spec: CallSpec, system: str, messages: list[dict], effort_label: str = "default",
                  max_tokens: int | None = None) -> dict:
    """Converse kwargs. messages: [{'role': 'user'|'assistant', 'text': str}, ...]."""
    effort = spec.resolve_effort(effort_label)
    req: dict[str, Any] = {
        "modelId": spec.model_id,
        "system": [{"text": system}] + ([{"cachePoint": {"type": "default"}}] if spec.prompt_cache else []),
        "messages": [{"role": m["role"], "content": [{"text": m["text"]}]} for m in messages],
        "inferenceConfig": {"maxTokens": int(max_tokens or spec.max_tokens_for(effort))},
    }
    amrf: dict[str, Any] = {}
    if effort:
        if spec.effort_strategy == "output_config":
            req["outputConfig"] = {"effort": effort}
        elif spec.effort_strategy == "reasoning_field":
            amrf["reasoning_effort"] = effort
    if spec.extra_fields:
        amrf.update(spec.extra_fields)
    if amrf:
        req["additionalModelRequestFields"] = amrf
    if spec.service_tier and spec.service_tier != "default":
        req["serviceTier"] = {"type": spec.service_tier}
    return req


def request_input_text(req: dict) -> str:
    parts = [b.get("text", "") for b in req.get("system", [])]
    for m in req.get("messages", []):
        parts.extend(b.get("text", "") for b in m["content"])
    return "\n".join(parts)


def process_response(resp: dict, latency_ms: int | None, requested_tier: str) -> CallResult:
    msg = (resp.get("output") or {}).get("message") or {}
    texts, reasons = [], []
    for b in msg.get("content", []) or []:
        if "text" in b:
            texts.append(b["text"])
        rc = b.get("reasoningContent")
        if rc:
            rt = rc.get("reasoningText") or {}
            if rt.get("text"):
                reasons.append(rt["text"])
            if rc.get("redactedContent") is not None:
                reasons.append("[redacted reasoning: %d bytes]" % len(rc["redactedContent"]))
    stop = resp.get("stopReason")
    status = "ok"
    if stop == "max_tokens":
        status = "truncated"
    elif stop in ("content_filtered", "guardrail_intervened", "refusal"):
        status = "refused"
    tier = ((resp.get("serviceTier") or {}).get("type")) or requested_tier or "default"
    return CallResult(status=status, text="".join(texts), reasoning="\n".join(reasons),
                      usage=dict(resp.get("usage") or {}), stop_reason=stop, served_tier=tier,
                      latency_ms=latency_ms or (resp.get("metrics") or {}).get("latencyMs"),
                      response=sanitize(resp))


def sanitize(obj: Any) -> Any:
    """Make a botocore response JSON-serialisable (bytes -> base64)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes_b64__": base64.b64encode(bytes(obj)).decode("ascii")}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class BedrockClient:
    """Thin wrapper over boto3 bedrock-runtime Converse with billing-aware retries."""

    def __init__(self, region: str, profile: str | None = None, read_timeout_s: int = 1200,
                 connect_timeout_s: int = 30, max_pool: int = 64, max_wait_s: float = 1800.0):
        import boto3
        from botocore.config import Config

        session = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)
        self.region = region
        self.client = session.client(
            "bedrock-runtime",
            config=Config(read_timeout=read_timeout_s, connect_timeout=connect_timeout_s,
                          retries={"max_attempts": 1, "mode": "standard"}, max_pool_connections=max_pool,
                          user_agent_extra="ccrl-panel/1.0"))
        self.max_wait_s = max_wait_s

    def converse(self, req: dict, requested_tier: str = "default") -> CallResult:
        from botocore.exceptions import (ClientError, ConnectionClosedError, ConnectTimeoutError,
                                         EndpointConnectionError, ReadTimeoutError)
        try:
            from urllib3.exceptions import ProtocolError
        except Exception:  # pragma: no cover
            ProtocolError = ConnectionClosedError  # type: ignore

        throttles = 0
        waited = 0.0
        delay = 2.0
        while True:
            t0 = time.time()
            try:
                resp = self.client.converse(**req)
            except ClientError as e:
                err = e.response.get("Error", {})
                code = err.get("Code", "ClientError")
                msg = err.get("Message", str(e))[:2000]
                if code in RETRY_IN_PLACE:
                    throttles += 1
                    if waited >= self.max_wait_s:
                        return CallResult("unbilled_error", error_code=code, error_msg=f"gave up after {waited:.0f}s: {msg}",
                                          throttles=throttles)
                    s = min(60.0, delay) * (0.5 + random.random())
                    time.sleep(s)
                    waited += s
                    delay = min(60.0, delay * 2)
                    continue
                if code in UNCERTAIN_CODES:
                    return CallResult("uncertain", error_code=code, error_msg=msg, throttles=throttles)
                fatal = "run" if code in FATAL_RUN else ("model" if code in FATAL_MODEL else None)
                return CallResult("unbilled_error", error_code=code, error_msg=msg, throttles=throttles, fatal=fatal)
            except (EndpointConnectionError, ConnectTimeoutError) as e:
                # Never reached the endpoint: not billed. Wait out short network outages (laptop Wi-Fi).
                throttles += 1
                if waited >= self.max_wait_s:
                    return CallResult("unbilled_error", error_code=type(e).__name__, error_msg=str(e)[:500],
                                      throttles=throttles)
                s = min(120.0, delay) * (0.5 + random.random())
                time.sleep(s)
                waited += s
                delay = min(120.0, delay * 2)
                continue
            except (ReadTimeoutError, ConnectionClosedError, ProtocolError) as e:
                return CallResult("uncertain", error_code=type(e).__name__, error_msg=str(e)[:500], throttles=throttles)
            latency = int((time.time() - t0) * 1000)
            res = process_response(resp, latency, requested_tier)
            res.throttles = throttles
            return res


def request_json(req: dict) -> str:
    return json.dumps(req, ensure_ascii=False, sort_keys=True)


def reservation_for(req: dict, price_key: str) -> float:
    text = request_input_text(req)
    return pricing.worst_case(pricing.estimate_tokens(text), int(req["inferenceConfig"]["maxTokens"]), price_key)
