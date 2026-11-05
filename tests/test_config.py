from ccrl import config


def test_config_valid():
    assert config.validate_all() == []


def test_panel_and_roles():
    ms = config.models()
    assert len(config.panel_models()) == 11
    assert config.role_model("preprocessor") != config.role_model("verifier")
    assert ms[config.role_model("preprocessor")].vendor != ms[config.role_model("verifier")].vendor


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
