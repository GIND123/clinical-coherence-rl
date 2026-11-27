"""The registered confirmatory tests of the training study (OSF osf.io/4xmq7), written before any
training result exists (audit, 27 Nov 2026).

    ccrl train hypotheses --dir runs/local_eval [--panel runs/export/posteriors.parquet]

Inputs are local evaluations named `<arm>_s<seed>.parquet` (all seeds of an arm are pooled by
averaging each case's metric over seeds) plus `base4b.parquet` / `base8b.parquet`.

Per case (all transformations of a case clustered):
  total incoherence = mean JS(base answer, other answer) over the 7 meaning-preserving answers
                      (perm1-3, dup, para, cue, retest) = (3 js_order + js_dup + js_para + js_cue + floor) / 7
  family JS         = js_order | js_dup | js_para | js_cue
  retest flip, top-1, top-3, Brier, ensemble gap (8-answer mean minus single answer; top-1 and Brier)

Tests (case bootstrap, 2,000 resamples, seed 0; one-sided where registered; Holm within a family):
  H2  per family f: relative reduction 1 - mean(CCR_f)/mean(RLVR_f) > 0.25        (Holm over 4)
  H3  top-1 CCR - RLVR on DDXPlus: one-sided 95% lower bound > -0.02
  H4  reduction ratio (RLVR - CCR) on held-out pathologies (and on MedCaseReasoning) /
      (RLVR - CCR) in distribution: point >= 0.5 and 95% CI excludes 0
  H5  label-free vs base on MedCaseReasoning: total incoherence lower (one-sided);
      top-3 non-inferior (lower bound > -0.03)
  H6  8B CCR vs Qwen3-32B and vs the panel median, DDXPlus: total incoherence and retest flip
      lower (one-sided), Holm over the 4 comparisons. Panel median per case = mean of the two
      middle-ranked models (by mean) of the six.
  H7  gap closure (gap_base - gap_CCR) / gap_base >= 0.5, for top-1 and Brier: one-sided test
      of closure > 0.5
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ccrl.workspace import Workspace

N_BOOT = 2000
FAMILIES = ("js_order", "js_dup", "js_para", "js_cue")
METRICS = ("total", *FAMILIES, "floor", "retest_flip", "top1", "top3", "brier", "gap_top1", "gap_brier")


# ------------------------------------------------------------------ per-case metrics
def per_case(parquet: str, model: str | None = None, ws: Workspace | None = None) -> pd.DataFrame:
    from ccrl.analysis.stability import reliability_per_case
    from ccrl.train.evaluate import estimands_for, load_posteriors

    pc = estimands_for(parquet, ws, model)["per_case"].copy()
    pc["total"] = (3 * pc["js_order"] + pc["js_dup"] + pc["js_para"] + pc["js_cue"] + pc["floor"]) / 7.0
    post = load_posteriors(parquet, model)
    post = post[post["parse_ok"].astype(bool) & post["probs"].notna()]
    st = []
    for ds, g in post.groupby("dataset"):
        r = reliability_per_case(g)
        r["dataset"] = ds
        st.append(r)
    if st:
        st = pd.concat(st, ignore_index=True)
        st["gap_top1"] = st["ens_top1"] - st["single_top1"]
        st["gap_brier"] = st["single_brier"] - st["ens_brier"]        # positive = ensemble better
        pc = pc.merge(st[["case_id", "dataset", "retest_flip", "gap_top1", "gap_brier"]],
                      on=["case_id", "dataset"], how="left")
    keep = ["case_id", "dataset"] + [m for m in METRICS if m in pc]
    return pc[keep]


def pooled(paths: list[str], model: str | None = None, ws: Workspace | None = None) -> pd.DataFrame:
    """Average each case's metrics over seeds (registered: mean over 3 seeds)."""
    frames = [per_case(p, model, ws) for p in paths]
    return pd.concat(frames).groupby(["case_id", "dataset"], as_index=False).mean(numeric_only=True)


# ------------------------------------------------------------------ bootstrap helpers
def _paired(a: pd.DataFrame, b: pd.DataFrame, ds: str | None, col: str, cases: set | None = None):
    A = a if ds is None else a[a["dataset"] == ds]
    B = b if ds is None else b[b["dataset"] == ds]
    m = A[["case_id", col]].merge(B[["case_id", col]], on="case_id", suffixes=("_a", "_b")).set_index("case_id")
    if cases is not None:
        m = m[m.index.isin(cases)]
    m = m.dropna()
    return m[f"{col}_a"].to_numpy(float), m[f"{col}_b"].to_numpy(float)


def _boot(stat, n: int, rng: np.random.Generator, n_boot: int = N_BOOT) -> np.ndarray:
    return np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ------------------------------------------------------------------ hypotheses
def h2(ccr: pd.DataFrame, rlvr: pd.DataFrame, ds: str = "ddxplus", threshold: float = 0.25, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for f in ("total", *FAMILIES):
        r, c = _paired(rlvr, ccr, ds, f)
        if not len(r) or r.mean() <= 0:
            continue
        est = 1 - c.mean() / r.mean()
        bs = _boot(lambda i: 1 - c[i].mean() / max(r[i].mean(), 1e-12), len(r), rng)
        rows.append({"hypothesis": "H2", "dataset": ds, "metric": f, "n": len(r), "ref_mean": r.mean(),
                     "new_mean": c.mean(), "rel_reduction": est, "lo95": np.percentile(bs, 2.5),
                     "hi95": np.percentile(bs, 97.5), "p_one_sided": float(np.mean(bs <= threshold))})
    df = pd.DataFrame(rows)
    fam = df["metric"].isin(FAMILIES)
    df.loc[fam, "p_holm"] = holm(df.loc[fam, "p_one_sided"].tolist())
    df["supported"] = df["p_holm"] < 0.05
    return df


def noninferiority(new: pd.DataFrame, ref: pd.DataFrame, ds: str, col: str, margin: float, label: str,
                   seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    r, n = _paired(ref, new, ds, col)
    d = n - r
    bs = _boot(lambda i: d[i].mean(), len(d), rng)
    lo = float(np.percentile(bs, 5))
    return {"hypothesis": label, "dataset": ds, "metric": col, "n": len(d), "diff": float(d.mean()),
            "lo_one_sided_95": lo, "margin": -margin, "supported": lo > -margin}


def superiority_lower(new: pd.DataFrame, ref: pd.DataFrame, ds: str, col: str, label: str, seed: int = 0,
                      cases: set | None = None) -> dict:
    """One-sided test that `new` is LOWER than `ref` on col (p = share of bootstrap diffs >= 0)."""
    rng = np.random.default_rng(seed)
    r, n = _paired(ref, new, ds, col, cases)
    d = n - r
    bs = _boot(lambda i: d[i].mean(), len(d), rng)
    return {"hypothesis": label, "dataset": ds, "metric": col, "n": len(d), "diff": float(d.mean()),
            "hi_one_sided_95": float(np.percentile(bs, 95)), "p_one_sided": float(np.mean(bs >= 0))}


def h4(ccr: pd.DataFrame, rlvr: pd.DataFrame, heldout_cases: set, seed: int = 0) -> pd.DataFrame:
    """Share of the in-distribution reduction in total incoherence (RLVR - CCR) that is retained on
    held-out pathologies and on MedCaseReasoning; the two samples are bootstrapped independently."""
    rng = np.random.default_rng(seed)
    ddx_cases = set(rlvr.loc[rlvr["dataset"] == "ddxplus", "case_id"])
    r_in, c_in = _paired(rlvr, ccr, "ddxplus", "total", ddx_cases - set(heldout_cases))
    rows = []
    for label, (r_o, c_o) in (("held-out pathologies", _paired(rlvr, ccr, "ddxplus", "total", set(heldout_cases))),
                              ("MedCaseReasoning", _paired(rlvr, ccr, "medcase", "total"))):
        if not len(r_o) or not len(r_in):
            continue
        den = r_in.mean() - c_in.mean()
        est = (r_o.mean() - c_o.mean()) / den if den else np.nan
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(r_o), len(r_o))
            j = rng.integers(0, len(r_in), len(r_in))
            dj = r_in[j].mean() - c_in[j].mean()
            bs.append((r_o[i].mean() - c_o[i].mean()) / dj if dj else np.nan)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append({"hypothesis": "H4", "transfer_to": label, "n_out": len(r_o), "n_in": len(r_in),
                     "in_dist_reduction": den, "retained": est, "lo95": lo, "hi95": hi,
                     "supported": bool(den > 0 and est >= 0.5 and lo > 0)})
    return pd.DataFrame(rows)


def panel_median(panel: pd.DataFrame, models: list[str], col: str) -> pd.DataFrame:
    """Per case, the mean of the two middle-ranked panel models (ranked by their mean on col)."""
    means = {m: panel[m][col].mean() for m in models}
    mid = sorted(means, key=means.get)[len(models) // 2 - 1: len(models) // 2 + 1]
    a = panel[mid[0]].set_index(["case_id", "dataset"])[[col]]
    b = panel[mid[1]].set_index(["case_id", "dataset"])[[col]]
    return ((a + b) / 2).reset_index()


def h7(ccr: pd.DataFrame, base: pd.DataFrame, ds: str = "ddxplus", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for col in ("gap_top1", "gap_brier"):
        b, c = _paired(base, ccr, ds, col)
        if not len(b) or b.mean() <= 0:
            rows.append({"hypothesis": "H7", "metric": col, "note": "base model has no ensemble gap"})
            continue
        est = (b.mean() - c.mean()) / b.mean()
        bs = _boot(lambda i: (b[i].mean() - c[i].mean()) / max(b[i].mean(), 1e-12), len(b), rng)
        rows.append({"hypothesis": "H7", "dataset": ds, "metric": col, "n": len(b), "closure": est,
                     "lo95": np.percentile(bs, 2.5), "p_one_sided": float(np.mean(bs <= 0.5)),
                     "supported": bool(np.mean(bs <= 0.5) < 0.05)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ driver
def _arm_files(d: Path, arm: str) -> list[str]:
    return sorted(str(p) for p in d.glob(f"{arm}_s*.parquet") if re.fullmatch(rf"{re.escape(arm)}_s\d+", p.stem))


def run(eval_dir: str, panel_parquet: str | None = None, ws: Workspace | None = None) -> pd.DataFrame:
    from ccrl.util import read_jsonl
    ws = ws or Workspace(False)
    d = Path(eval_dir)
    out = []
    arms = {a: _arm_files(d, a) for a in ("ccr_acc_fid", "rlvr", "labelfree_mcr", "qwen3_8b_ccr_acc_fid")}
    P = {a: pooled(f, ws=ws) for a, f in arms.items() if f}
    base4 = per_case(str(d / "base4b.parquet"), ws=ws) if (d / "base4b.parquet").exists() else None
    if "ccr_acc_fid" in P and "rlvr" in P:
        out.append(h2(P["ccr_acc_fid"], P["rlvr"]))
        out.append(pd.DataFrame([noninferiority(P["ccr_acc_fid"], P["rlvr"], "ddxplus", "top1", 0.02, "H3")]))
        from ccrl import paths
        held = json.loads((paths.FROZEN_DDX / "heldout.json").read_text())["heldout"]
        ci = read_jsonl(ws.items_dir / "case_index.jsonl")
        held_cases = {r["case_id"] for r in ci if r.get("pathology") in set(held) and r["role"] == "panel"}
        out.append(h4(P["ccr_acc_fid"], P["rlvr"], held_cases))
    if "labelfree_mcr" in P and base4 is not None:
        out.append(pd.DataFrame([superiority_lower(P["labelfree_mcr"], base4, "medcase", "total", "H5"),
                                 noninferiority(P["labelfree_mcr"], base4, "medcase", "top3", 0.03, "H5")]))
    if "ccr_acc_fid" in P and base4 is not None:
        out.append(h7(P["ccr_acc_fid"], base4))
    if "qwen3_8b_ccr_acc_fid" in P and panel_parquet:
        from ccrl import config
        models = config.panel_models()
        panel = {m: per_case(panel_parquet, model=m, ws=ws) for m in models}
        rows = []
        for col in ("total", "retest_flip"):
            rows.append(superiority_lower(P["qwen3_8b_ccr_acc_fid"], panel["qwen3-32b"], "ddxplus", col, "H6 vs Qwen3-32B"))
            rows.append(superiority_lower(P["qwen3_8b_ccr_acc_fid"], panel_median(panel, models, col), "ddxplus", col,
                                          "H6 vs panel median"))
        h6 = pd.DataFrame(rows)
        h6["p_holm"] = holm(h6["p_one_sided"].tolist())
        h6["supported"] = h6["p_holm"] < 0.05
        out.append(h6)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
