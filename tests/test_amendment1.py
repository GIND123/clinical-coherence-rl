"""Amendment 1 (after the main panel): run-to-run noise analysis, replicate and
temperature arms, and the operational fixes from the 2026-11-17/18 run
(interleaved mirror lines, a full disk, double-counted Flex spend)."""
import json
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ccrl import config
from ccrl.analysis import stability as St
from ccrl.bedrock.runner import task_id
from ccrl.util import append_jsonl, stable_hash


def test_append_jsonl_is_thread_safe(tmp_path):
    p = tmp_path / "m.jsonl"
    big = "x" * 40000                              # Kimi rows are ~20-60 KB

    def w(k):
        for i in range(25):
            append_jsonl(p, {"k": k, "i": i, "pad": big})

    ts = [threading.Thread(target=w, args=(k,)) for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200 and all(json.loads(x)["pad"] == big for x in lines)


def test_task_id_unchanged_without_temperature():
    """Every request already in the PI's ledger keeps its ID (no re-sends)."""
    old = stable_hash("panel_main", "i1", "kimi-k3", "us.moonshotai.kimi-k3", "default", 0,
                      stable_hash("sys", "turn", n=64), n=24)
    assert task_id("panel_main", "i1", "kimi-k3", "us.moonshotai.kimi-k3", "default", 0, "sys", ["turn"]) == old
    assert task_id("temp0", "i1", "kimi-k3", "m", "default", 0, "sys", ["turn"], 0.0) != \
        task_id("temp0", "i1", "kimi-k3", "m", "default", 0, "sys", ["turn"])


def test_disk_check_refuses_when_full(tmp_runs, monkeypatch):
    import shutil

    from ccrl.panel import jobs
    from ccrl.workspace import get as get_ws
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(total=1e12, used=1e12, free=1e9))
    with pytest.raises(SystemExit, match="GB free"):
        jobs.check_disk(get_ws(True))


def test_backup_pruning_keeps_newest(tmp_runs):
    from ccrl.bedrock.ledger import Ledger
    from ccrl.panel import jobs
    from ccrl.workspace import get as get_ws
    ws = get_ws(True)
    d = ws.runs / "backups"
    d.mkdir(parents=True)
    for k in range(5):
        f = d / f"ledger-old{k}.sqlite"
        f.write_bytes(b"x")
        os.utime(f, (time.time() - 1000 + k, time.time() - 1000 + k))
    led = Ledger(ws.ledger)
    try:
        jobs.backup_ledger(ws, led, "pilot")
    finally:
        led.close()
    left = sorted(p.name for p in d.glob("ledger-*.sqlite"))
    assert len(left) == int(config.budget().get("keep_backups", 3))
    assert any(n.startswith("ledger-pilot-") for n in left) and "ledger-old4.sqlite" in left


def _post(model, cases, noise, rng, transforms=St.MEANING_PRESERVING, k=10, job="panel_main"):
    rows = []
    for c in range(cases):
        mean = np.full(k, 0.5 / (k - 1))
        mean[0] = 0.5
        for t in transforms:
            p = mean + noise * rng.dirichlet(np.ones(k))
            p = p / p.sum()
            rows.append({"job": job, "model_key": model, "effort": "default", "dataset": "medcase",
                         "case_id": f"c{c}", "transform": t, "turn": 1, "parse_ok": True, "probs": p,
                         "gold": "AA", "replicate": 0})
    return rows


def test_reliability_separates_noise_from_signal():
    rng = np.random.default_rng(0)
    post = pd.DataFrame(_post("steady", 60, 0.0, rng) + _post("noisy", 60, 2.0, rng))
    S, pc = St.reliability(post, n_boot=50)
    s = S.set_index("model_key")
    assert s.loc["steady", "retest_flip"] == 0 and s.loc["steady", "floor"] < 1e-9
    assert s.loc["steady", "variance_share"] < 1e-9 and abs(s.loc["steady", "ens_gain_top1"]) < 1e-9
    assert s.loc["noisy", "retest_flip"] > 0.2 and s.loc["noisy", "variance_share"] > 0.01
    assert s.loc["noisy", "ens_top1"] >= s.loc["noisy", "single_top1"]


def test_auroc():
    assert St.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert St.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert St.auroc([0.5, 0.5], [0, 1]) == 0.5


def test_updating_on_mean_posteriors():
    """A model whose single answers are noisy but whose MEAN update follows the
    corpus is credited with the right direction once replicates are averaged."""
    rng = np.random.default_rng(1)
    rows, items = [], {}
    for c in range(40):
        cid = f"ddx{c}"
        llr = 1.0 if c % 2 else -1.0                 # sign of the removed finding's evidence for gold
        items[f"{cid}::ablate"] = {"meta": {"log_lr_gold": llr, "kb_bayes_full": [0.6] + [0.4 / 48] * 48,
                                            "kb_bayes_ablated": [0.6 - 0.3 * llr] + [(0.4 + 0.3 * llr) / 48] * 48}}
        for job, t, reps in (("panel_main", "base", [0]), ("panel_main", "retest", [0]),
                             ("panel_main", "ablate", [0]), ("resample", "base", [1, 2, 3]),
                             ("resample", "ablate", [1, 2, 3])):
            for r in reps:
                g = 0.6 - (0.25 * llr if t == "ablate" else 0.0) + rng.normal(0, 0.15)
                g = float(np.clip(g, 0.02, 0.97))
                p = np.full(49, (1 - g) / 48)
                p[0] = g
                rows.append({"job": job, "model_key": "m", "effort": "default", "dataset": "ddxplus",
                             "case_id": cid, "transform": t, "turn": 1, "parse_ok": True, "probs": p,
                             "gold": "AA", "replicate": r})
    U = St.updating(pd.DataFrame(rows), items, n_boot=50)
    u = U.iloc[0]
    assert u.k_base_mean == 5 and u.k_ablate_mean == 4
    assert u.dir_agree_mean > 0.9 and u.mean_signed_update > 0 and u.slope_vs_reference > 0


def test_freeze_history_keeps_registered_record(tmp_runs):
    from ccrl.panel import freeze, items
    from ccrl.workspace import get as get_ws
    ws = get_ws(True)
    items.build(ws, allow_missing_para=True, allow_short_mcr=True, verbose=False)
    first = freeze.freeze(ws)
    raw = ws.freeze_file.read_bytes()
    assert freeze.freeze(ws)["created_at"] == first["created_at"] and ws.freeze_file.read_bytes() == raw
    p = ws.items_dir / "systems.json"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    second = freeze.freeze(ws)
    hist = list((ws.freeze_file.parent / "freeze_history").glob("FREEZE-*.json"))
    assert len(hist) == 1 and hist[0].read_bytes() == raw and second["previous"]["file"].endswith(hist[0].name)


def test_reprice_flex_lowers_only_finished_flex(tmp_runs, monkeypatch):
    from ccrl import billing
    from ccrl.bedrock.ledger import Ledger
    from ccrl.workspace import get as get_ws
    ws = get_ws(True)
    led = Ledger(ws.ledger)
    led.ensure_requests([{"request_id": r, "job": "panel_main", "model_key": "kimi-k3", "item_id": r}
                         for r in ("a", "b", "c")])
    for r, tier, status in (("a", "flex", "ok"), ("b", "default", "ok"), ("c", "flex", "uncertain")):
        led.open_attempt(r, 1, 1, 0.30, "{}")
        led.close_attempt(r, 1, 1, status=status, cost_reported=0.01, cost_guard=0.025, served_tier=tier)
    before = led.committed()
    led.close()
    with pytest.raises(SystemExit, match="guard_counts_flex"):
        billing.reprice_flex(ws, apply=True)
    real = dict(config.prices())
    monkeypatch.setattr(config, "prices", lambda: {**real, "guard_counts_flex": True})
    billing.reprice_flex(ws, apply=True)
    led = Ledger(ws.ledger)
    try:
        g = {r["request_id"]: r["cost_guard"] for r in led.q("SELECT request_id, cost_guard FROM attempts")}
        gf = __import__("ccrl.bedrock.pricing", fromlist=["x"]).price_for("kimi-k3").guard_factor
        assert g["a"] == pytest.approx(0.01 * gf) and g["b"] == pytest.approx(0.025)
        assert led.committed() < before
    finally:
        led.close()


def test_flex_verdict():
    from ccrl.billing import flex_verdict
    ledger = {"2026-11-07": 30.0, "2026-11-17": 150.0, "2026-11-18": 60.0}
    # AWS matches the Flex-priced ledger on complete days; the recent day is ignored
    v, info = flex_verdict({"2026-11-07": 29.5, "2026-11-17": 151.0, "2026-11-18": 5.0}, ledger, "2026-11-19")
    assert v == "confirmed" and info["days"] == ["2026-11-07", "2026-11-17"]
    # a lagging, partial day cannot fake a match: it is outside the window
    v, _ = flex_verdict({"2026-11-07": 59.0, "2026-11-17": 300.0}, ledger, "2026-11-19")
    assert v == "standard"
    v, _ = flex_verdict({"2026-11-07": 3.0}, {"2026-11-07": 3.0}, "2026-11-20")
    assert v == "inconclusive"
