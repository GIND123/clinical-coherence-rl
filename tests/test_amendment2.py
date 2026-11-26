"""Amendment 2: temperature 0 on the full panel and replication sample; contamination probe."""
import numpy as np
import pandas as pd

from ccrl import config
from ccrl.bedrock import models as M
from ccrl.panel import contamination as C
from ccrl.panel.jobs import plan_tasks
from ccrl.workspace import Workspace

WS = Workspace()          # the committed frozen inputs and model lock (no network, no ledger writes)


def test_temp0_arms_cover_full_sets_without_kimi():
    specs = M.load_lock(False)
    full = plan_tasks("temp0_panel", WS, specs)
    ext = plan_tasks("temp0_ext", WS, specs)
    assert len(full) == 600 * 10 * 5 and len(ext) == 300 * 10 * 5
    assert {t.model_key for t in full} == {"glm-5", "deepseek-v3.2", "qwen3-next-80b", "gpt-oss-120b", "qwen3-32b"}
    assert all(t.temperature == 0.0 for t in full + ext)
    # new request IDs: nothing collides with the core temp0 arm or the main panel
    old = {t.request_id for j in ("temp0", "panel_main", "panel_ext") for t in plan_tasks(j, WS, specs)}
    new = [t.request_id for t in full + ext]
    assert len(set(new)) == len(new) and not (set(new) & old)


def test_new_jobs_have_caps_and_validate():
    caps = config.budget()["job_caps_usd"]
    assert {"temp0_panel", "temp0_ext", "contamination"} <= set(caps)
    assert config.validate_all() == []


def test_contamination_tasks():
    specs = M.load_lock(False)
    tasks = C.build_tasks(WS, specs)
    assert len(tasks) == 300 * len(config.panel_models()) * 4
    assert len({t.request_id for t in tasks}) == len(tasks)
    by = {}
    for t in tasks:
        by.setdefault(t.transform, []).append(t)
    assert all(len(t.codes) == 10 and t.validator is None for t in by["id"] + by["noid"])
    assert all(t.kind == "text" and t.validator is not None for t in by["guided"] + by["general"])
    i, n = by["id"][0], by["noid"][0]
    assert "PMC" in i.turns[0] and "PMC" not in n.turns[0]
    g, gen = by["guided"][0], by["general"][0]
    assert "PubMed Central" in g.turns[0] and "PubMed Central" not in gen.turns[0]
    # same withheld split in both completion prompts
    assert g.turns[0].split("\n\n", 1)[1] == gen.turns[0].split("\n\n", 1)[1]


def test_split_and_scores():
    text = "First sentence here. Second one is a bit longer than that. Third. Fourth sentence closes it."
    a, b = C.split_text(text)
    assert a and b and (a + " " + b) == text and len(a) >= len(text) / 2 - 20
    assert C.N_WORDS_MIN <= C.n_words(b) <= C.N_WORDS_MAX
    ref = C._tokens("the biopsy showed granulomas with central necrosis and the culture grew mycobacteria")
    assert C.rouge_n_f1(ref, ref) == 1.0 and C.ngram_overlap(ref, ref, 8) == 1.0
    assert C.longest_shared(ref, ref) == len(ref)
    other = C._tokens("blood tests were normal and imaging was unremarkable")
    assert C.rouge_n_f1(other, ref) == 0.0 and C.longest_shared(other, ref) <= 2
    assert C.mentions("Final diagnosis: pulmonary tuberculosis.", "Pulmonary tuberculosis")
    assert not C.mentions("A fungal infection was suspected.", "Pulmonary tuberculosis")
    ok, inf, _ = C.text_validator("<think>hidden</think>   short")
    assert not ok


def test_summarize_detects_memorisation():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        mem = i < 60
        rows.append({"model_key": "m", "case_id": f"c{i}",
                     "top1_id": float(mem or rng.random() < 0.3), "top1_noid": float(rng.random() < 0.3),
                     "pgold_id": 0.6 if mem else 0.3, "pgold_noid": 0.3,
                     "rouge2_guided": 0.5 if mem else 0.05, "rouge2_general": 0.05,
                     "ngram8_guided": 0.4 if mem else 0.0, "ngram8_general": 0.0,
                     "longest_guided": 20.0 if mem else 3.0, "longest_general": 3.0,
                     "dx_mention_guided": 0.5, "dx_mention_general": 0.5})
    pc = pd.DataFrame(rows)
    pc["suspect"] = (((pc["top1_id"] == 1) & (pc["top1_noid"] == 0)) | (pc["longest_guided"] >= 13)).astype(float)
    s = C.summarize(pc, n_boot=200).iloc[0]
    assert s["d_top1_lo"] > 0 and s["d_rouge2_lo"] > 0 and s["verbatim13_guided"] == 0.3
    assert abs(s["d_dx_mention"]) < 1e-9


def test_refreeze_keeps_hash_when_only_line_endings_differ(tmp_runs):
    from ccrl.panel import freeze
    from ccrl.util import read_json
    from ccrl.workspace import get as get_ws

    ws = get_ws(True)
    ws.items_dir.mkdir(parents=True, exist_ok=True)
    (ws.items_dir / "items.jsonl").write_bytes(b"{}\n")
    csv = ws.banks / "review.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_bytes(b"a,b\r\n1,2\r\n")
    first = freeze.freeze(ws)["files"]["frozen/banks/review.csv"]
    csv.write_bytes(b"a,b\n1,2\n")                       # same content, LF (fresh git checkout)
    assert freeze.check_freeze(ws)[0]
    doc = freeze.freeze(ws)
    assert doc["files"]["frozen/banks/review.csv"] == first
    assert read_json(ws.freeze_file)["files"] == doc["files"]
