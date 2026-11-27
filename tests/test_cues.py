import re

from ccrl import config
from ccrl.data import ddxplus as dx
from ccrl.panel.items import cue_ok


def test_cue_lint():
    c = config.cues()
    bl = c["lint_blacklist"]
    for cue in c["irrelevant"]:
        low = cue["text"].lower()
        for w in bl:
            if len(w) <= 4:
                assert not re.search(rf"\b{re.escape(w)}\b", low), (cue, w)
            else:
                assert w not in low, (cue, w)


def test_cues_never_name_a_ddxplus_diagnosis():
    cands = dx.load_kb().candidates()
    for cue in config.cues()["irrelevant"]:
        assert cue_ok(cue["text"], cands), cue


def test_cue_ids_unique_and_40():
    ids = [c["id"] for c in config.cues()["irrelevant"]]
    assert len(ids) == 40 == len(set(ids))


def test_training_cues_lint_and_are_disjoint_from_test_cues():
    """plan_final.md P2: training never shows a test cue; training cues obey the same rules."""
    bl = config.cues()["lint_blacklist"]
    train = config.cues_train()["irrelevant"]
    test = {c["text"].strip().lower() for c in config.cues()["irrelevant"]}
    cands = dx.load_kb().candidates()
    assert len(train) == 40 == len({c["id"] for c in train}) == len({c["text"] for c in train})
    for cue in train:
        low = cue["text"].lower()
        assert low.strip() not in test, cue
        assert cue_ok(cue["text"], cands), cue
        for w in bl:
            if len(w) <= 4:
                assert not re.search(rf"\b{re.escape(w)}\b", low), (cue, w)
            else:
                assert w not in low, (cue, w)


def test_training_phrasebank_never_uses_a_test_paraphrase():
    import json
    from ccrl import paths
    panel = json.loads((paths.FROZEN_BANKS / "ddx_phrasebank.json").read_text(encoding="utf-8"))
    train = json.loads((paths.FROZEN_BANKS / "ddx_phrasebank_train.json").read_text(encoding="utf-8"))
    pf, tf = set(panel["fallback_ids"]), set(train["fallback_ids"])
    assert set(panel["templates"]) == set(train["templates"])
    for k, v in train["templates"].items():
        if k not in pf:                                     # the panel bank paraphrased this phrase
            assert v != panel["templates"][k], k
    assert len(tf) <= 20


def test_training_sampler_uses_training_banks():
    import os
    from pathlib import Path

    import pytest
    if not (Path(os.environ.get("CCRL_DATA", "data")) / "train_pool" / "ddxplus_train.jsonl").exists():
        pytest.skip("DDXPlus training pool not built")
    from ccrl.train.data import PairedSampler
    test_cues = [c["text"] for c in config.cues()["irrelevant"]]
    s = PairedSampler("ddxplus", 0, {"cue": 1.0}, phrasebank_path="artifacts/frozen/banks/ddx_phrasebank_train.json",
                      cue_bank="train")
    for g in s.sample(40, 1, 1):
        assert not any(c in g.xp[0] for c in test_cues)
        assert any(c["text"] in g.xp[0] for c in config.cues_train()["irrelevant"])
