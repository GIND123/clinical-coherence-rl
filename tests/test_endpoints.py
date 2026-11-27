"""plan_final.md §3.3: primary endpoints. The flip rate cannot be moved by tempering or by mixing
an answer with the uniform distribution (hedging); primary() orients its one-sided tests."""
import numpy as np
import pandas as pd

from ccrl.train import hypotheses as H


def _post(rng, n_cases=30, K=6, temper=1.0, mix=0.0):
    rows = []
    for c in range(n_cases):
        mu = rng.dirichlet(np.ones(K) * 0.5)
        for t in ("base", *H.NUISANCE):
            p = rng.dirichlet(mu * 20 + 0.05)
            p = p ** (1 / temper)
            p = p / p.sum()
            p = (1 - mix) * p + mix / K
            rows.append({"case_id": f"c{c}", "dataset": "ddxplus", "transform": t, "probs": p})
    return pd.DataFrame(rows)


def test_flip_rate_invariant_to_tempering_and_hedging():
    a = H.nuisance_flips(_post(np.random.default_rng(0)))
    b = H.nuisance_flips(_post(np.random.default_rng(0), temper=3.0))
    c = H.nuisance_flips(_post(np.random.default_rng(0), mix=0.8))
    assert np.allclose(a["nuisance_flip"], b["nuisance_flip"])
    assert np.allclose(a["nuisance_flip"], c["nuisance_flip"])
    assert 0 < a["nuisance_flip"].mean() < 1


def _pc(rng, flip, total, ablate, n=200):
    return pd.DataFrame({"case_id": [f"c{i}" for i in range(n)], "dataset": "ddxplus",
                         "nuisance_flip": rng.normal(flip, 0.05, n).clip(0, 1),
                         "total": rng.normal(total, 0.02, n).clip(0.001), "ceiling": rng.normal(0.6, 0.02, n),
                         "js_ablate": rng.normal(ablate, 0.02, n).clip(0.001), "js_nuis": rng.normal(total, 0.02, n).clip(0.001)})


def test_primary_orients_tests():
    rng = np.random.default_rng(1)
    better = _pc(rng, 0.10, 0.10, 0.30)
    worse = _pc(rng, 0.30, 0.25, 0.20)
    df = H.primary(better, worse, "x").set_index("metric")
    assert df["supported"].all()
    df = H.primary(worse, better, "x").set_index("metric")
    assert not df["supported"].any()
    v = H.endpoint_values(better)
    assert set(v) == {"nuisance_flip", "norm_incoherence", "selectivity"}


def test_brier_decomposition_identity():
    rng = np.random.default_rng(2)
    conf = rng.uniform(0.1, 1, 500)
    hit = (rng.uniform(0, 1, 500) < conf).astype(float)
    pc = pd.DataFrame({"dataset": "ddxplus", "conf": conf, "top1": hit})
    d = H.brier_decomposition(pc, bins=10)
    assert d["reliability"] < 0.02
    # binned identity holds up to within-bin variance of the forecasts
    assert abs(d["brier_top"] - (d["reliability"] - d["resolution"] + d["uncertainty"])) < 0.01
