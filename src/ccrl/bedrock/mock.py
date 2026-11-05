"""Offline mock of Bedrock Converse, for `--dry-run` and the test suite.

It answers every task the harness sends (posterior elicitation, both schema
variants, MedCaseReasoning extraction/verification, phrase bank, probes) with
well-formed, plausible output and realistic usage blocks, and it injects the
failure modes the runner must survive: malformed JSON, Markdown fences,
<think> preambles, truncation and post-send timeouts. Nothing leaves the
machine and nothing is billed.

Simulated posteriors are a function of the evidence SET (sorted sentences)
plus a model-specific order sensitivity and sampling noise, so the analysis
pipeline sees non-trivial floors, ceilings and order effects.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading

import numpy as np

from ccrl.bedrock.client import CallResult, process_response, request_input_text

_CODE_LINE = re.compile(r"^([A-Z]{2})  (.+)$", re.M)
_SENT = re.compile(r"(?<=[.!?])\s+")


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:15], 16)


class MockBedrock:
    def __init__(self, seed: int = 0, failure_rate: float = 0.01, latency_s: float = 0.0):
        self.seed = seed
        self.failure_rate = failure_rate
        self.latency_s = latency_s
        self._lock = threading.Lock()
        self._n = 0

    # -------------------------------------------------------------- dispatch
    def converse(self, req: dict, requested_tier: str = "default") -> CallResult:
        import time
        with self._lock:
            self._n += 1
            n = self._n
        if self.latency_s:
            time.sleep(self.latency_s)
        system = "\n".join(b.get("text", "") for b in req.get("system", []))
        user_turns = [b["text"] for m in req["messages"] if m["role"] == "user" for b in m["content"]]
        model = req["modelId"]
        rng = np.random.default_rng(_h(self.seed, model, n, request_input_text(req)))
        # injected transport failure (after send -> uncertain)
        if rng.random() < self.failure_rate * 0.3:
            return CallResult("uncertain", error_code="ReadTimeoutError", error_msg="mock timeout")
        if "TASK: MCR_EXTRACTION" in system:
            text = self._extract(user_turns[-1])
        elif "TASK: MCR_VERIFICATION" in system:
            text = json.dumps({"unsupported": [], "paraphrase_not_equivalent": [], "reveals_diagnosis": [],
                               "stage_errors": [], "distractors_invalid": [], "final_diagnosis_ok": True,
                               "comment": "mock"})
        elif "TASK: DDX_PHRASEBANK_VERIFY" in system:
            items = json.loads(user_turns[-1].split("Items:\n", 1)[1].rsplit("\n\nReturn", 1)[0])
            text = json.dumps({"items": [{"id": it["id"], "ok": [True, True, True]} for it in items]})
        elif "TASK: DDX_PHRASEBANK" in system:
            items = json.loads(user_turns[-1].split("Items:\n", 1)[1].rsplit("\n\nReturn", 1)[0])
            text = json.dumps({"items": [{"id": it["id"], "alternatives": [
                self._alt(it["base"], k) for k in range(3)]} for it in items]})
        elif "TASK: PROBE_PING" in system:
            text = "OK"
        else:
            text = self._posterior(system, user_turns, model, rng)
        out_tokens = int(len(text) / 4) + 1
        reasoning_tokens = int(rng.integers(150, 900))
        stop = "end_turn"
        max_tokens = int(req["inferenceConfig"]["maxTokens"])
        if out_tokens + reasoning_tokens > max_tokens or rng.random() < self.failure_rate * 0.3:
            stop = "max_tokens"
            text = text[: max(1, len(text) // 3)]
        if rng.random() < self.failure_rate:
            text = text.replace("}", "", 1)                    # malformed JSON
        elif rng.random() < 0.05:
            text = "```json\n" + text + "\n```"
        elif rng.random() < 0.05:
            text = "<think>considering the findings</think>" + text
        in_tok = int(len(request_input_text(req)) / 4)
        cached = any("cachePoint" in b for b in req.get("system", []))
        cr = int(len(system) / 4) if cached and n > 1 else 0
        usage = {"inputTokens": in_tok - cr, "outputTokens": out_tokens + reasoning_tokens,
                 "cacheReadInputTokens": cr, "cacheWriteInputTokens": 0,
                 "totalTokens": in_tok + out_tokens + reasoning_tokens}
        content = [{"reasoningContent": {"reasoningText": {"text": "mock reasoning summary", "signature": "x"}}},
                   {"text": text}]
        resp = {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": stop,
                "usage": usage, "metrics": {"latencyMs": int(rng.integers(300, 4000))},
                "serviceTier": {"type": requested_tier or "default"},
                "ResponseMetadata": {"RequestId": f"mock-{n}", "HTTPStatusCode": 200}}
        return process_response(resp, None, requested_tier)

    # -------------------------------------------------------------- tasks
    @staticmethod
    def _alt(base: str, k: int) -> str:
        return {0: base + " (as reported)", 1: base.replace("has", "reports having", 1),
                2: base + " according to the patient"}[k]

    @staticmethod
    def _extract(user: str) -> str:
        body = user.split("Case presentation:\n", 1)[1]
        case, rest = body.split("\n\nFinal diagnosis reported by the authors:", 1)
        dx = rest.split("\n", 1)[0].strip()
        dx = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", dx).replace("_", " ").strip().capitalize()
        paras = [p for p in case.split("\n\n") if p.strip()]
        findings = []
        for pi, p in enumerate(paras):
            for s in _SENT.split(p.strip()):
                s = s.strip()
                if len(s) > 20 and len(findings) < 16:
                    findings.append({"text": s, "paraphrase": "Reportedly: " + s, "stage": min(pi, 3),
                                     "chief_complaint": False, "names_diagnosis": False})
        if findings:
            findings[0]["chief_complaint"] = True
            findings[0]["stage"] = 0
        return json.dumps({"patient": "an adult patient", "findings": findings, "final_diagnosis": dx,
                           "distractors": [f"Alternative condition {k + 1}" for k in range(9)]})

    def _posterior(self, system: str, user_turns: list[str], model: str, rng: np.random.Generator) -> str:
        block = system + "\n" + user_turns[0]
        codes = [m.group(1) for m in _CODE_LINE.finditer(block)]
        if not codes:
            return '{"dx": null, "p": {}}'
        key = "s" if '"s":' in system else "p"
        vignette = "\n".join(user_turns)
        sents = sorted(s.strip().lower().replace("as noted above, ", "").replace("as noted above: ", "")
                       for s in _SENT.split(vignette) if s.strip())
        evidence = " ".join(dict.fromkeys(sents))
        # evidence-set signal
        base_rng = np.random.default_rng(_h("set", evidence))
        logits = base_rng.normal(0, 2.5, len(codes))
        # model-specific order sensitivity and sampling noise
        order_sd = 0.3 + (_h("order", model) % 100) / 100.0
        logits += np.random.default_rng(_h("order", model, vignette)).normal(0, order_sd, len(codes))
        logits += rng.normal(0, 0.4, len(codes))
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        pct = np.floor(p * 100).astype(int)
        pct[np.argmax(p)] += 100 - pct.sum()
        dx = codes[int(np.argmax(pct))]
        if key == "s":
            vals = {c: int(min(100, round(x * 300))) for c, x in zip(codes, p)}
        else:
            vals = {c: int(v) for c, v in zip(codes, pct)}
        return json.dumps({"dx": dx, key: vals})
