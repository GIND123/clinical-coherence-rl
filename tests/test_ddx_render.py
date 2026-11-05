import re

from ccrl.data import ddx_render as R
from ccrl.data import ddxplus as dx
from ccrl.panel.cases import ddx_cases_raw, ddx_prepared


def test_every_code_renders():
    assert R.unrendered_codes(dx.load_kb()) == []


def test_all_frozen_cases_render_cleanly():
    bad = []
    for rec in ddx_cases_raw():
        c = ddx_prepared(rec, None)
        for f in c.findings:
            s = f.base
            if "{value}" in s or re.search(r"\b(you|your)\b", s, re.I) or "  " in s or not s.endswith("."):
                bad.append((rec["case_id"], s))
        assert c.findings[0].fixed, "chief complaint must be the first unit"
    assert not bad, bad[:5]


def test_context_articles():
    assert R.context_sentence(8, "M").startswith("An 8-year-old boy")
    assert R.context_sentence(18, "F").startswith("An 18-year-old woman")
    assert R.context_sentence(0, "F").startswith("A female infant")
    assert R.context_sentence(45, "F").startswith("A 45-year-old woman")


def test_counts_and_heldout():
    from ccrl import paths
    from ccrl.util import read_json
    rep = read_json(paths.FROZEN_DDX / "build_report.json")["counts"]
    assert rep == {"pilot": 20, "panel": 300, "panel_ext": 300, "core": 75, "sequential": 150, "distill": 2000}
    held = set(read_json(paths.FROZEN_DDX / "heldout.json")["heldout"])
    assert len(held) == 8
    assert not any(r["pathology"] in held for r in ddx_cases_raw() if r["role"] == "distill")
    panel = [r for r in ddx_cases_raw() if r["role"] == "panel"]
    assert len({r["pathology"] for r in panel}) == 49


def test_reference_tau_and_likelihood():
    lik = dx.load_likelihood()
    assert 0 < lik.tau <= 1
    rec = next(r for r in ddx_cases_raw() if r["role"] == "panel")
    q = lik.posterior(rec["tokens"])
    assert abs(q.sum() - 1) < 1e-9
    # order invariance of the reference (Proposition 1)
    assert abs(lik.posterior(list(reversed(rec["tokens"]))) - q).max() < 1e-12
