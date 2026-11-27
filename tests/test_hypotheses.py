"""Registered confirmatory tests (src/ccrl/train/hypotheses.py) on synthetic per-case data."""
import numpy as np
import pandas as pd

from ccrl.train import hypotheses as H


def _frame(scale: float, n: int = 300, seed: int = 0, ds: str = "ddxplus", top1: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 0.1, n)
    df = pd.DataFrame({"case_id": [f"c{i}" for i in range(n)], "dataset": ds})
    for f in H.FAMILIES + ("floor",):
        df[f] = base * scale * rng.uniform(0.9, 1.1, n)
    df["total"] = (3 * df["js_order"] + df["js_dup"] + df["js_para"] + df["js_cue"] + df["floor"]) / 7
    df["top1"] = (rng.random(n) < top1).astype(float)
    df["top3"] = df["top1"]
    return df


def test_h2_detects_large_reduction_only():
    rlvr = _frame(1.0)
    assert H.h2(_frame(0.5, seed=1), rlvr).query("metric != 'total'")["supported"].all()      # 50% reduction
    assert not H.h2(_frame(0.9, seed=1), rlvr).query("metric != 'total'")["supported"].any()  # 10% reduction


def test_noninferiority_and_holm():
    ref, same = _frame(1.0, top1=0.5), _frame(1.0, seed=2, top1=0.5)
    worse = _frame(1.0, seed=3, top1=0.35)
    assert H.noninferiority(same, ref, "ddxplus", "top1", 0.10, "x")["supported"]
    assert not H.noninferiority(worse, ref, "ddxplus", "top1", 0.02, "x")["supported"]
    assert H.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_h7_gap_closure():
    base = _frame(1.0)
    base["gap_top1"], base["gap_brier"] = 0.10, 0.05
    ccr = base.copy()
    ccr["gap_top1"], ccr["gap_brier"] = 0.01, 0.04
    r = H.h7(ccr, base).set_index("metric")
    assert r.loc["gap_top1", "supported"] and not r.loc["gap_brier", "supported"]


def test_per_case_on_a_real_evaluation():
    import os
    p = "E:/ccrl-data/lambda_runs/x_150.136.211.207/runs/local_eval/base4b.parquet"
    if not os.path.exists(p):
        return
    pc = H.per_case(p)
    assert {"total", "retest_flip", "gap_top1", "gap_brier"} <= set(pc.columns)
    assert len(pc) == 600 and pc["total"].between(0, 1).all()
