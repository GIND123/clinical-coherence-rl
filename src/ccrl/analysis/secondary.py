"""Secondary arms compared with the main panel on identical cases.

    paired_vs_panel(post)   temp0 and effort_arm vs panel_main, case by case
    sequential_gap(post)    two-stage order effect beyond the one-shot retest floor

Every comparison is paired by case: an arm is compared with the main panel's
answers to the same cases (the core 150), never with the full 300-case panel.
Intervals: case bootstrap, 2,000 resamples. Top-1 is the argmax of the stated
posterior, as in the estimands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ccrl.analysis import metrics as Mx
from ccrl.util import two_letter_codes

N_BOOT = 2000
ARMS = ("temp0", "effort_arm")


def _ci(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _top1(probs: np.ndarray, gold) -> float:
    if not gold:
        return np.nan
    return float(two_letter_codes(len(probs))[int(np.argmax(probs))] == gold)


def per_case(g: pd.DataFrame) -> pd.DataFrame:
    """One row per case: retest flip and JS, JS of perm1/para/cue to base, top-1 of base."""
    v = {(r.case_id, r.transform): r for r in g.itertuples()}
    rows = []
    for c in sorted({k[0] for k in v}):
        b, rt = v.get((c, "base")), v.get((c, "retest"))
        if b is None or rt is None:
            continue
        pb, pr = np.asarray(b.probs), np.asarray(rt.probs)
        row = {"case_id": c, "flip": float(np.argmax(pb) != np.argmax(pr)), "js_retest": Mx.js(pb, pr),
               "top1": _top1(pb, b.gold)}
        for t in ("perm1", "para", "cue"):
            x = v.get((c, t))
            row[f"js_{t}"] = Mx.js(pb, np.asarray(x.probs)) if x is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("case_id") if rows else pd.DataFrame()


def _single(post: pd.DataFrame) -> pd.DataFrame:
    p = post[(post["turn"] == 1) & post["parse_ok"]]
    if "replicate" in p:
        p = p[p["replicate"].fillna(0) == 0]
    return p


def paired_vs_panel(post: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Per model x dataset x arm: default vs arm on the same cases, and each condition's
    excess order / paraphrase JS beyond its own retest JS (systematic part of the effect)."""
    rng = np.random.default_rng(seed)
    p = _single(post)
    ref_all = p[p["job"] == "panel_main"]
    out = []
    for job in ARMS:
        a = p[p["job"] == job]
        for (mk, ds, eff), g in a.groupby(["model_key", "dataset", "effort"]):
            ref = per_case(ref_all[(ref_all["model_key"] == mk) & (ref_all["dataset"] == ds)])
            new = per_case(g)
            if ref.empty or new.empty:
                continue
            cs = ref.index.intersection(new.index)
            R, N = ref.loc[cs], new.loc[cs]
            row = {"arm": job if job != "effort_arm" else f"effort_{eff}", "model_key": mk, "dataset": ds,
                   "n_cases": len(cs)}
            for col in ("flip", "js_retest", "top1"):
                row[f"{col}_panel"] = float(R[col].mean())
                row[f"{col}_arm"] = float(N[col].mean())
                m, lo, hi = _ci((N[col] - R[col]).to_numpy(float), n_boot, rng)
                row.update({f"d_{col}": m, f"d_{col}_lo": lo, f"d_{col}_hi": hi})
            for nm, df in (("panel", R), ("arm", N)):
                for t in ("perm1", "para"):
                    m, lo, hi = _ci((df[f"js_{t}"] - df["js_retest"]).to_numpy(float), n_boot, rng)
                    key = f"excess_{'order' if t == 'perm1' else 'para'}_{nm}"
                    row.update({key: m, f"{key}_lo": lo, f"{key}_hi": hi})
            out.append(row)
    return pd.DataFrame(out)


def sequential_gap(post: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """JS between the final posteriors after evidence in order A->B vs B->A, minus the
    same case's one-shot retest JS (main panel); flip_AB: the two orders' top diagnoses differ.
    Caveat: the floor is single-turn, so part of the gap can be extra two-turn sampling noise."""
    rng = np.random.default_rng(seed)
    s = post[(post["job"] == "sequential") & (post["turn"] == 2) & post["parse_ok"]]
    one = _single(post)
    one = one[one["job"] == "panel_main"]
    rows = []
    for mk, g in s.groupby("model_key"):
        A = {r.case_id: np.asarray(r.probs) for r in g.itertuples() if r.transform == "seqA"}
        B = {r.case_id: np.asarray(r.probs) for r in g.itertuples() if r.transform == "seqB"}
        o = one[one["model_key"] == mk]
        b = {r.case_id: np.asarray(r.probs) for r in o.itertuples() if r.transform == "base"}
        rt = {r.case_id: np.asarray(r.probs) for r in o.itertuples() if r.transform == "retest"}
        cs = [c for c in A if c in B and c in b and c in rt]
        if not cs:
            continue
        d = np.array([Mx.js(A[c], B[c]) - Mx.js(b[c], rt[c]) for c in cs])
        m, lo, hi = _ci(d, n_boot, rng)
        rows.append({"model_key": mk, "n_cases": len(cs), "gap": m, "gap_lo": lo, "gap_hi": hi,
                     "flip_AB": float(np.mean([np.argmax(A[c]) != np.argmax(B[c]) for c in cs]))})
    return pd.DataFrame(rows)
