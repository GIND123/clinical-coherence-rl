from ccrl import config


def test_config_valid():
    assert config.validate_all() == []


def test_panel_and_roles():
    ms = config.models()
    panel = config.panel_models()
    assert len(panel) == 6
    # the confirmatory study uses only models the account can call
    assert not set(panel) & set(config.optional_models())
    assert set(config.optional_models()) == {"opus-5.5", "sonnet-5.5", "gpt-6-sol", "gpt-6-luna", "grok-4.6"}
    for name, j in config.jobs().items():
        if j["priority"] in ("A", "B"):
            assert not set(config.job_models(name)) & set(config.optional_models())
    assert config.role_model("preprocessor") != config.role_model("verifier")
    assert ms[config.role_model("preprocessor")].vendor != ms[config.role_model("verifier")].vendor


def test_probe_default_excludes_partner_models():
    assert set(config.study_models()) == set(config.panel_models())


def test_caps_under_credit():
    b = config.budget()
    assert b["hard_cap_usd"] <= b["credit_usd"] - 50
    assert b["priority_c_gate_usd"] < b["hard_cap_usd"]


def test_no_temperature_anywhere():
    # Opus 5.5 rejects sampling parameters; the harness never sends them.
    from ccrl.bedrock.client import CallSpec, build_request
    req = build_request(CallSpec("m", "x", "opus-5.5", effort_strategy="output_config", default_effort="low"),
                        "s", [{"role": "user", "text": "u"}])
    assert "temperature" not in req["inferenceConfig"] and "topP" not in req["inferenceConfig"]
    assert req["outputConfig"] == {"effort": "low"}


def test_stale_probe_detected():
    """A lock written under an older registry entry (e.g. gpt-oss-120b probed
    before it had an effort setting) must be flagged, so paid jobs re-probe."""
    from ccrl.bedrock.models import registry_sig, stale_lock_models
    doc = {"models": {
        "gpt-oss-120b": {"probe": {"ok": True}, "registry_sig": "old"},
        "kimi-k3": {"probe": {"ok": True}, "registry_sig": registry_sig("kimi-k3")},
        "opus-5.5": {"probe": {"ok": False}},
    }}
    assert stale_lock_models(doc) == ["gpt-oss-120b"]
