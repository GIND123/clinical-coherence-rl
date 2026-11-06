"""Requests are validated against the real botocore service model (Stubber):
a parameter the installed botocore does not accept fails HERE, not on the
first paid call."""
import pytest
from botocore.stub import Stubber

from ccrl.bedrock import client as C

OK_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [
        {"reasoningContent": {"reasoningText": {"text": "thinking", "signature": "sig"}}},
        {"text": '{"dx": "AA", "p": {"AA": 100}}'}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 100, "outputTokens": 50, "totalTokens": 1150, "cacheReadInputTokens": 1000,
              "cacheWriteInputTokens": 0},
    "metrics": {"latencyMs": 1000},
    "serviceTier": {"type": "flex"},
}


def _client():
    c = C.BedrockClient("us-east-1", read_timeout_s=10, connect_timeout_s=5, max_wait_s=0.1)
    return c, Stubber(c.client)


@pytest.mark.parametrize("spec", [
    C.CallSpec("opus-5.5", "us.anthropic.claude-opus-5-5", "opus-5.5", effort_strategy="output_config",
               default_effort="low", prompt_cache=True,
               extra_fields={"thinking": {"type": "adaptive", "display": "summarized"}}),
    C.CallSpec("gpt-6-sol", "global.openai.gpt-6-sol", "gpt-6-sol", effort_strategy="reasoning_field",
               default_effort="high"),
    C.CallSpec("kimi-k3", "moonshotai.kimi-k3", "kimi-k3", service_tier="flex"),
])
def test_request_shapes_validate(spec):
    c, st = _client()
    req = C.build_request(spec, "system text", [{"role": "user", "text": "u1"},
                                                {"role": "assistant", "text": "a1"},
                                                {"role": "user", "text": "u2"}])
    st.add_response("converse", OK_RESPONSE, expected_params=req)
    with st:
        res = c.converse(req, spec.service_tier)
    assert res.status == "ok" and res.text.startswith('{"dx"') and res.reasoning == "thinking"
    assert res.served_tier == "flex"


def test_pricing_cache_semantics_and_flex():
    from ccrl.bedrock import pricing
    u = pricing.usage_tokens(OK_RESPONSE["usage"])
    assert u["semantics"] == "exclusive" and u["uncached_input"] == 100
    rep, guard = pricing.cost(OK_RESPONSE["usage"], "kimi-k3", "flex")
    p = pricing.price_for("kimi-k3")
    full = (100 * p.input + 1000 * p.cache_read + 50 * p.output) / 1e6
    assert abs(rep - full * 0.5) < 1e-12 and abs(guard - full * 1.25) < 1e-12
    incl = pricing.usage_tokens({"inputTokens": 1100, "outputTokens": 50, "totalTokens": 1150,
                                 "cacheReadInputTokens": 1000})
    assert incl["semantics"] == "inclusive" and incl["uncached_input"] == 100


def test_error_classification(monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    spec = C.CallSpec("m", "model-x", "glm-5")
    req = C.build_request(spec, "s", [{"role": "user", "text": "u"}])
    c, st = _client()
    st.add_client_error("converse", "ThrottlingException", "slow down", 429)
    st.add_response("converse", OK_RESPONSE, expected_params=req)
    with st:
        r = c.converse(req)
    assert r.status == "ok" and r.throttles == 1
    for code, status, fatal in (("ValidationException", "unbilled_error", "model"),
                                ("AccessDeniedException", "unbilled_error", "run"),
                                ("ModelTimeoutException", "uncertain", None)):
        c, st = _client()
        st.add_client_error("converse", code, "x", 400)
        with st:
            r = c.converse(req)
        assert r.status == status and r.fatal == fatal, code


def test_truncation_and_refusal():
    for stop, status in (("max_tokens", "truncated"), ("content_filtered", "refused")):
        resp = dict(OK_RESPONSE, stopReason=stop)
        assert C.process_response(resp, 1, "default").status == status
