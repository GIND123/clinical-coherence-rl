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
