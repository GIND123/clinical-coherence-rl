"""Pre-registered statistics: clustered bootstrap, Holm, non-inferiority, mixed models.

Clustering unit is the CASE (patient): every resample draws whole cases, so
the 10 calls of one case, and an estimator's floor and numerator, move
together.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

N_BOOT = 2000


def cluster_bootstrap(case_ids: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int = N_BOOT,
                      seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """stat receives an index array (rows of the caller's data) for a resample of cases.
    Returns (point, lo, hi)."""
    case_ids = np.asarray(case_ids)
    uniq, inv = np.unique(case_ids, return_inverse=True)
    rows_by_case = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    point = stat(np.arange(len(case_ids)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_case[k] for k in pick]) if len(pick) else np.array([], int)
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    return float(point), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(a: pd.Series, b: pd.Series, n_boot: int = N_BOOT, seed: int = 0,
                          alpha: float = 0.05) -> dict:
    """a, b indexed by case id (paired). Difference of means b - a with CI and two-sided p."""
    common = a.index.intersection(b.index)
    x = (b.loc[common] - a.loc[common]).to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"n": len(x), "diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    d = float(x.mean())
    p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    return {"n": int(len(x)), "diff": d, "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)), "p": max(p, 1.0 / n_boot)}


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def non_inferiority(diff_lo_one_sided: float, margin: float) -> bool:
    """H3: non-inferior iff the one-sided 95% lower bound of (new - reference) > -margin."""
    return bool(diff_lo_one_sided > -margin)


def paired_noninferiority(ref: pd.Series, new: pd.Series, margin: float, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """One-sided 95% (alpha=0.10 two-sided interval's lower end) paired bootstrap test."""
    r = paired_bootstrap_diff(ref, new, n_boot, seed, alpha=0.10)
    r["margin"] = margin
    r["non_inferior"] = non_inferiority(r["lo"], margin)
    return r


def mixed_model(df: pd.DataFrame, formula: str, group: str):
    """Linear mixed model with a random intercept per `group` (statsmodels).
    Returns the fitted result, or None if statsmodels is unavailable or the fit fails."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    try:
        return smf.mixedlm(formula, df, groups=df[group]).fit(reml=True, method="lbfgs")
    except Exception:
        return None
