"""Amendment 3: post-freeze real case reports (selection rules, cut screening, item merge)."""
import json

from ccrl import paths
from ccrl.data import fresh as D
from ccrl.prep import fresh as F
from ccrl.util import read_jsonl


def test_selection_is_committed_and_post_freeze():
    rows = read_jsonl(paths.FROZEN / "fresh" / "cases_selected.jsonl")
    assert len(rows) >= 150
    assert all(r["publication_date"] >= D.FIRST_DAY for r in rows)
    assert all("creativecommons.org/licenses/by/" in r["license"] or "cc by" in r["license"].lower() for r in rows)
    assert [r["prep_order"] for r in rows] == list(range(len(rows)))
    assert all(r["sentences"] and " ".join(r["sentences"]).replace(" ", "") ==
               r["case_text"].replace("\n", "").replace(" ", "") for r in rows[:20])


def test_eligibility_rules():
    base = {"publication_date": "2026-11-10", "license": "https://creativecommons.org/licenses/by/4.0/",
            "case_text": "A 40-year-old man presented with fever. " * 30, "title": "Fever of unknown origin",
            "n_second_case": 0}
    assert D.eligible(base) is None
    assert D.eligible({**base, "publication_date": "2026-11-06"})
    assert D.eligible({**base, "license": "https://creativecommons.org/licenses/by-nc/4.0/"})
    assert D.eligible({**base, "case_text": "The 7 cases spanned many presentations. " * 30})
    assert D.eligible({**base, "case_text": "Cases 1 and 2 were late. " * 40})
    assert D.eligible({**base, "title": "Two cases of fever"})
    assert D.eligible({**base, "case_text": "short"})


def test_cut_validation():
    chk = F._cut_check(10)
    ok = {"single_patient": True, "final_diagnosis": "x", "diagnostic_case": True, "last_sentence": 7}
    assert chk(ok) is None
    assert chk({**ok, "last_sentence": 11}) and chk({**ok, "last_sentence": "7"}) and chk({**ok, "last_sentence": True})
    assert chk({**ok, "diagnostic_case": "yes"})


def test_screen_rules(tmp_runs, monkeypatch):
    from ccrl.workspace import get as get_ws

    ws = get_ws(True)
    sents = [f"Sentence {i} about the patient's symptoms and the examination findings recorded." for i in range(1, 13)]
    sel = [{"case_id": f"fresh-PMC{i}", "prep_order": i, "pmcid": f"PMC{i}", "sentences": sents, "title": "t",
            "journal": "j", "publication_date": "2026-11-10", "license": "cc by", "article_link": "l"} for i in range(5)]
    cut = {
        "fresh-PMC0": {"single_patient": True, "final_diagnosis": "Kikuchi disease", "diagnostic_case": True, "last_sentence": 10},
        "fresh-PMC1": {"single_patient": False, "final_diagnosis": "Kikuchi disease", "diagnostic_case": True, "last_sentence": 10},
        "fresh-PMC2": {"single_patient": True, "final_diagnosis": "Kikuchi disease", "diagnostic_case": False, "last_sentence": 10},
        "fresh-PMC3": {"single_patient": True, "final_diagnosis": "Kikuchi disease", "diagnostic_case": True, "last_sentence": 2},
        "fresh-PMC4": {"single_patient": True, "final_diagnosis": "Examination syndrome", "diagnostic_case": True, "last_sentence": 10},
    }
    monkeypatch.setattr(F, "selected", lambda limit=None: sel)
    monkeypatch.setattr(F, "role_spec", lambda ws, role: ("kimi-k3", None))
    monkeypatch.setattr(F, "outputs", lambda ws, job, key: cut)
    rows = {r["case_id"]: r for r in F.screen(ws)}
    assert rows["fresh-PMC0"]["status"] == "ok" and rows["fresh-PMC0"]["case_prompt"] == " ".join(sents[:10])
    assert "more than one patient" in rows["fresh-PMC1"]["reject_reasons"]
    assert "diagnosis known at presentation" in rows["fresh-PMC2"]["reject_reasons"]
    assert any("too short" in x for x in rows["fresh-PMC3"]["reject_reasons"])
    assert "diagnosis named in the vignette" in rows["fresh-PMC4"]["reject_reasons"]   # 'examination' leaks


def test_fresh_items_merge_into_loaders(tmp_runs):
    from ccrl.panel import items as I
    from ccrl.util import write_json, write_jsonl
    from ccrl.workspace import get as get_ws

    ws = get_ws(True)
    ws.items_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(ws.items_dir / "items.jsonl", [{"item_id": "a::base"}])
    write_json(ws.items_dir / "sets.json", {"panel": ["a::base"]})
    assert "fresh" not in I.load_sets(ws)
    d = ws.frozen / "fresh" / "items"
    d.mkdir(parents=True, exist_ok=True)
    write_jsonl(d / "items.jsonl", [{"item_id": "fresh-PMC1::base"}])
    (d / "sets.json").write_text(json.dumps({"fresh": ["fresh-PMC1::base"]}))
    assert set(I.load_items(ws)) == {"a::base", "fresh-PMC1::base"}
    assert I.load_sets(ws) == {"panel": ["a::base"], "fresh": ["fresh-PMC1::base"]}
