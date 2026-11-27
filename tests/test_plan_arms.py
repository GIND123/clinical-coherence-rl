"""P3 (plan_final.md): new arm configs and the self-consistency distillation target."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ccrl.train.grpo import load_config

CFG = Path(__file__).resolve().parents[1] / "config" / "train"


def test_new_arm_configs():
    b = load_config(str(CFG / "brier_only.yaml"))
    assert (b["reward"]["alpha"], b["reward"]["beta"], b["reward"]["gamma"], b["reward"]["acc_mode"]) == \
        (1.0, 0.0, 0.0, "brier")
    assert b["credit"] == "team" and b["readout"]["strict_sum"] and b["shared_prompt"]
    a = load_config(str(CFG / "mech_team_lenient.yaml"))
    assert a["credit"] == "team" and not a["readout"]["strict_sum"] and a["steps"] == 60
    c = load_config(str(CFG / "mech_registered_strict.yaml"))
    assert c["credit"] == "reward" and c["readout"]["strict_sum"] and c["steps"] == 60
    for x in (a, c):                              # cells differ from the method in one switch only
        m = load_config(str(CFG / "ccr_acc_fid.yaml"))
        assert x["reward"] == m["reward"]
    s = load_config(str(CFG / "sft_selfcons.yaml"))
    assert s["method"] == "sft" and s["sft_source"] == "selfcons" and s["gradient_checkpointing"]
    for f in ("bct", "sft_kb_bayes", "sft_distill"):
        assert load_config(str(CFG / f"{f}.yaml"))["gradient_checkpointing"]


def _pool():
    p = Path(os.environ.get("CCRL_DATA", "data")) / "train_pool" / "ddxplus_train.jsonl"
    if not p.exists():
        pytest.skip("DDXPlus training pool not built (ccrl train build-pool)")
    return p


def test_selfcons_target_is_mean_of_renderings(monkeypatch):
    _pool()
    from ccrl.train import sft
    from ccrl.train.data import PairedSampler
    smp = PairedSampler("ddxplus", 0)
    calls = {}

    # answer: all mass on candidate (i % 2): mean over 8 = 50/50 on the first two codes
    def fake_answers(cfg, convs, temperature=0.7, top_p=0.95):
        calls["n"] = len(convs)
        return [f"__{i % 2}__" for i in range(len(convs))]

    from ccrl.prompts import parse as P
    real = P.parse_posterior

    def parse(text, codes):
        k = int(text.strip("_"))
        return real(json.dumps({"dx": codes[k], "p": {c: (100 if j == k else 0) for j, c in enumerate(codes)}}), codes)

    monkeypatch.setattr(sft, "base_answers", fake_answers)
    monkeypatch.setattr(P, "parse_posterior", parse)
    ex = sft.selfcons_examples({"seed": 0}, smp, 3)
    assert calls["n"] == 3 * 8 and len(ex) == 3 * 4
    t = json.loads(ex[0][-1]["content"])
    vals = sorted(t["p"].values(), reverse=True)
    assert vals[:2] == [50, 50] and sum(vals) == 100
    assert ex[0][1]["content"] != ex[2][1]["content"]     # clean vs transformed rendering
