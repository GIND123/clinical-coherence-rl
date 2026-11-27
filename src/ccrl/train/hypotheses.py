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

Primary endpoints (plan_final.md §3.3, 27 Nov 2026, before any run under it): nuisance flip rate,
normalised incoherence and selectivity, each arm against RLVR, Brier-only, BCT, self-consistency
distillation and the untrained model (`primary`); per-seed signs (`seed_signs`).

Informativeness guard (Amendment 4, 27 Nov 2026; revised in plan_final.md §3.3, see `guard`).
Total incoherence is minimised by answering every patient with the same flat posterior, and the
first decisive coherence run did exactly that (docs/training_runs.md). H2 is therefore reported as
supported only if, on the same DDXPlus items, the CCR arm keeps (a) at least 90% of RLVR's
cross-patient information M of the base answers (a registered safeguard estimand) and (b) a mean
posterior entropy no more than 0.5 nats above RLVR's. Reported alongside (secondary): the share of
answers whose percentages sum to 100 (+-2), and normalised incoherence = total incoherence / mean JS
between the case's base answer and the other cases' base answers (unchanged by uniform tempering).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ccrl.workspace import Workspace

N_BOOT = 2000
M0 = 0.459            # reward constant m0 (config/train/base.yaml): the information floor of the guard
FAMILIES = ("js_order", "js_dup", "js_para", "js_cue")
METRICS = ("total", *FAMILIES, "floor", "retest_flip", "top1", "top3", "brier", "gap_top1", "gap_brier",
           # plan_final.md §3.3 (27 Nov 2026): tempering-robust primary endpoints and their parts
           "nuisance_flip", "js_nuis", "ceiling", "js_ablate", "dir_agree_gold", "dir_agree_top5", "entropy", "conf")
NUISANCE = ("retest", "perm1", "perm2", "perm3", "dup", "para", "cue")


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
    pc["js_nuis"] = (3 * pc["js_order"] + pc["js_dup"] + pc["js_para"] + pc["js_cue"]) / 6.0
    pc = pc.merge(nuisance_flips(post), on=["case_id", "dataset"], how="left")
    keep = ["case_id", "dataset"] + [m for m in METRICS if m in pc]
    return pc[keep]


def nuisance_flips(post: pd.DataFrame) -> pd.DataFrame:
    """Per case: share of the 7 meaning-preserving renderings (retest, 3 orders, duplication,
    paraphrase, cue) whose top-1 answer differs from the base rendering's. Invariant to any
    monotone tempering of an answer, so flattening cannot lower it."""
    rows = []
    for (ds, c), g in post.groupby(["dataset", "case_id"]):
        v = {r.transform: int(np.argmax(np.asarray(r.probs, float))) for r in g.itertuples(index=False)}
        if "base" not in v:
            continue
        f = [float(v[t] != v["base"]) for t in NUISANCE if t in v]
        rows.append({"case_id": c, "dataset": ds, "nuisance_flip": float(np.mean(f)) if f else np.nan})
    return pd.DataFrame(rows, columns=["case_id", "dataset", "nuisance_flip"])


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


# ------------------------------------------------------------------ primary endpoints (plan_final.md §3.3)
def _ratio(a: np.ndarray, b: np.ndarray, i) -> float:
    d = b[i].mean()
    return float(a[i].mean() / d) if d > 0 else float("nan")


def endpoint_values(pc: pd.DataFrame, ds: str = "ddxplus") -> dict:
    """Arm-level values of the three primary endpoints (ratios of means over cases)."""
    x = pc[pc["dataset"] == ds]
    return {"nuisance_flip": float(x["nuisance_flip"].mean()),
            "norm_incoherence": float(x["total"].mean() / x["ceiling"].mean()),
            "selectivity": float(x["js_ablate"].mean() / x["js_nuis"].mean())}


def primary(new: pd.DataFrame, ref: pd.DataFrame, label: str, ds: str = "ddxplus", seed: int = 0) -> pd.DataFrame:
    """The three primary endpoints, `new` against `ref`: paired case bootstrap, one-sided, Holm over 3.
      nuisance flip rate      lower is better  (mean over cases)
      normalised incoherence  lower is better  (mean total incoherence / mean between-patient JS)
      selectivity             higher is better (mean JS to the ablated answer / mean nuisance JS)"""
    rng = np.random.default_rng(seed)
    cols = ["nuisance_flip", "total", "ceiling", "js_ablate", "js_nuis"]
    A = new[new["dataset"] == ds][["case_id", *cols]]
    B = ref[ref["dataset"] == ds][["case_id", *cols]]
    m = A.merge(B, on="case_id", suffixes=("_n", "_r")).dropna()
    g = {k: m[k].to_numpy(float) for k in m.columns if k != "case_id"}
    stats = {
        "nuisance_flip": (lambda i, s: float(g[f"nuisance_flip_{s}"][i].mean()), -1),
        "norm_incoherence": (lambda i, s: _ratio(g[f"total_{s}"], g[f"ceiling_{s}"], i), -1),
        "selectivity": (lambda i, s: _ratio(g[f"js_ablate_{s}"], g[f"js_nuis_{s}"], i), +1),
    }
    n = len(m)
    rows = []
    for name, (f, better) in stats.items():
        row = {"hypothesis": f"primary: {label}", "dataset": ds, "metric": name, "n": n,
               "better": "lower" if better < 0 else "higher"}
        if n:
            idx = np.arange(n)
            bs = np.array([f(i, "n") - f(i, "r") for i in (rng.integers(0, n, n) for _ in range(N_BOOT))])
            row.update({"new": f(idx, "n"), "ref": f(idx, "r"), "diff": f(idx, "n") - f(idx, "r"),
                        "lo95": float(np.nanpercentile(bs, 2.5)), "hi95": float(np.nanpercentile(bs, 97.5)),
                        "p_one_sided": float(np.mean(better * bs <= 0))})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_one_sided"].fillna(1.0).tolist()) if "p_one_sided" in df else 1.0
    df["supported"] = df["p_holm"] < 0.05
    return df


def seed_signs(new_paths: list[str], ref_paths: list[str], ds: str = "ddxplus",
               ws: Workspace | None = None) -> pd.DataFrame:
    """Each seed on its own (seed k against seed k of the reference): the sign of every primary
    endpoint's difference. A result is called consistent only if every seed has the same sign."""
    rows = []
    for pn, pr in zip(new_paths, ref_paths):
        a, b = endpoint_values(per_case(pn, ws=ws), ds), endpoint_values(per_case(pr, ws=ws), ds)
        rows.append({"new": Path(pn).stem, "ref": Path(pr).stem, **{f"d_{k}": a[k] - b[k] for k in a}})
    df = pd.DataFrame(rows)
    if not df.empty:
        want = {"d_nuisance_flip": -1, "d_norm_incoherence": -1, "d_selectivity": +1}
        df.attrs["consistent"] = {k: bool((np.sign(df[k]) == sg).all()) for k, sg in want.items()}
    return df


def brier_decomposition(pc: pd.DataFrame, ds: str = "ddxplus", bins: int = 10) -> dict:
    """Murphy decomposition of the top-label Brier score (event: the top answer is correct) on the
    base renderings: Brier = reliability - resolution + uncertainty (secondary endpoint)."""
    x = pc[pc["dataset"] == ds]
    conf, hit = x["conf"].to_numpy(float), x["top1"].to_numpy(float)
    ok = np.isfinite(conf) & np.isfinite(hit)
    conf, hit = conf[ok], hit[ok]
    if not len(conf):
        return {}
    k = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    base = hit.mean()
    rel = res = 0.0
    for b in range(bins):
        m = k == b
        if m.any():
            rel += m.mean() * (conf[m].mean() - hit[m].mean()) ** 2
            res += m.mean() * (hit[m].mean() - base) ** 2
    return {"brier_top": float(np.mean((conf - hit) ** 2)), "reliability": float(rel), "resolution": float(res),
            "uncertainty": float(base * (1 - base))}


# ------------------------------------------------------------------ informativeness guard (Amendment 4)
def informativeness(parquet: str, ds: str = "ddxplus", model: str | None = None) -> tuple[dict, pd.DataFrame]:
    """Dataset-level M, entropy and sum-to-100 compliance of the base answers, plus per-case
    between-patient JS (the denominator of normalised incoherence)."""
    from ccrl.analysis.metrics import entropy, js_rows
    from ccrl.reward.coherence import cross_patient_information
    d = pd.read_parquet(parquet)
    if model is not None and "model_key" in d:
        d = d[d["model_key"] == model]
    d = d[(d["dataset"] == ds) & (d["transform"] == "base") & d["parse_ok"].astype(bool) & d["probs"].notna()]
    if d.empty:
        return {}, pd.DataFrame(columns=["case_id", "between"])
    d = d.drop_duplicates("case_id")
    P = np.stack([np.asarray(p, dtype=float) for p in d["probs"]])
    sums = d["text"].map(_raw_sum)
    between = []
    for i in range(len(P)):
        other = np.delete(P, i, axis=0)
        between.append(float(js_rows(other, np.repeat(P[i][None], len(other), 0)).mean()))
    info = {"M": cross_patient_information(list(P)), "entropy": float(np.mean([entropy(p) for p in P])),
            "max_prob": float(P.max(1).mean()), "sum100": float(np.mean(np.abs(sums.dropna() - 100) <= 2))
            if sums.notna().any() else np.nan, "n": int(len(P))}
    return info, pd.DataFrame({"case_id": d["case_id"].values, "between": between})


def _raw_sum(text) -> float:
    m = re.search(r'"p"\s*:\s*\{(.*?)\}', text or "", re.S)
    if not m:
        return np.nan
    vals = re.findall(r'"[A-Za-z]{2}"\s*:\s*(-?[0-9.]+)', m.group(1))
    return float(sum(float(v) for v in vals)) if vals else np.nan


def guard(ccr_paths: list[str], rlvr_paths: list[str], ccr_pc: pd.DataFrame, rlvr_pc: pd.DataFrame,
          ds: str = "ddxplus") -> pd.DataFrame:
    ic = [informativeness(p, ds)[0] for p in ccr_paths]
    ir = [informativeness(p, ds)[0] for p in rlvr_paths]
    mean = lambda L, k: float(np.mean([x[k] for x in L if x]))  # noqa: E731
    Mc, Mr, Ec, Er = mean(ic, "M"), mean(ir, "M"), mean(ic, "entropy"), mean(ir, "entropy")
    # revised guard (plan_final.md §3.3, A4.4; 27 Nov 2026, before any run under it). The raw H2
    # reduction counts only if the two tempering-robust endpoints are also lower than RLVR's
    # (one-sided 95%, unadjusted) and the coherence arm's M stays at or above m0. The Amendment 4
    # rule (M >= 90% of RLVR's, entropy <= RLVR's + 0.5) is still reported, as `supported_a4`.
    pr = primary(ccr_pc, rlvr_pc, "CCR vs RLVR (guard)", ds).set_index("metric")
    robust = bool(pr.loc["nuisance_flip"].get("p_one_sided", 1.0) < 0.05
                  and pr.loc["norm_incoherence"].get("p_one_sided", 1.0) < 0.05)
    rows = [{"hypothesis": "H2 guard", "dataset": ds, "M_ccr": Mc, "M_rlvr": Mr, "entropy_ccr": Ec, "entropy_rlvr": Er,
             "sum100_ccr": mean(ic, "sum100"), "sum100_rlvr": mean(ir, "sum100"), "m0": M0,
             "supported_a4": bool(Mc >= 0.9 * Mr and Ec <= Er + 0.5),
             "supported": bool(robust and Mc >= M0)}]
    bc = pd.concat([informativeness(p, ds)[1] for p in ccr_paths]).groupby("case_id", as_index=False).mean()
    br = pd.concat([informativeness(p, ds)[1] for p in rlvr_paths]).groupby("case_id", as_index=False).mean()
    a = ccr_pc[ccr_pc["dataset"] == ds].merge(bc, on="case_id")
    b = rlvr_pc[rlvr_pc["dataset"] == ds].merge(br, on="case_id")
    if len(a) and len(b):
        nc = a["total"].mean() / a["between"].mean()
        nr = b["total"].mean() / b["between"].mean()
        rows.append({"hypothesis": "normalised incoherence (secondary)", "dataset": ds, "ccr": nc, "rlvr": nr,
                     "relative_reduction": 1 - nc / nr if nr else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ driver
def _arm_files(d: Path, arm: str) -> list[str]:
    return sorted(str(p) for p in d.glob(f"{arm}_s*.parquet") if re.fullmatch(rf"{re.escape(arm)}_s\d+", p.stem))


def run(eval_dir: str, panel_parquet: str | None = None, ws: Workspace | None = None) -> pd.DataFrame:
    from ccrl.util import read_jsonl
    ws = ws or Workspace(False)
    d = Path(eval_dir)
    out = []
    arms = {a: _arm_files(d, a) for a in ("ccr_acc_fid", "rlvr", "labelfree_mcr", "qwen3_8b_ccr_acc_fid",
                                          "brier_only", "bct", "sft_selfcons")}
    P = {a: pooled(f, ws=ws) for a, f in arms.items() if f}
    base4 = per_case(str(d / "base4b.parquet"), ws=ws) if (d / "base4b.parquet").exists() else None
    if "ccr_acc_fid" in P and "rlvr" in P:
        out.append(h2(P["ccr_acc_fid"], P["rlvr"]))
        out.append(guard(arms["ccr_acc_fid"], arms["rlvr"], P["ccr_acc_fid"], P["rlvr"]))
        out.append(pd.DataFrame([noninferiority(P["ccr_acc_fid"], P["rlvr"], "ddxplus", "top1", 0.02, "H3")]))
        from ccrl import paths
        held = json.loads((paths.FROZEN_DDX / "heldout.json").read_text())["heldout"]
        ci = read_jsonl(ws.items_dir / "case_index.jsonl")
        held_cases = {r["case_id"] for r in ci if r.get("pathology") in set(held) and r["role"] == "panel"}
        out.append(h4(P["ccr_acc_fid"], P["rlvr"], held_cases))
    if "ccr_acc_fid" in P:
        # primary endpoints (plan_final.md §3.3): C2 vs RLVR, C3 vs Brier-only, C4 vs BCT and
        # self-consistency distillation, and every arm against the untrained model
        for ref in ("rlvr", "brier_only", "bct", "sft_selfcons"):
            if ref in P:
                out.append(primary(P["ccr_acc_fid"], P[ref], f"CCR vs {ref}"))
        if base4 is not None:
            for arm in P:
                out.append(primary(P[arm], base4, f"{arm} vs base"))
        if "rlvr" in P and len(arms["ccr_acc_fid"]) > 1:
            ss = seed_signs(arms["ccr_acc_fid"], arms["rlvr"], ws=ws)
            out.append(pd.DataFrame([{"hypothesis": "seed signs CCR vs RLVR",
                                      **{f"consistent_{k}": v for k, v in ss.attrs.get("consistent", {}).items()}}]))
        for arm in P:
            out.append(pd.DataFrame([{"hypothesis": "Brier decomposition (secondary)", "arm": arm,
                                      **brier_decomposition(P[arm])}]))
    gb = d / "general_base4b.parquet"
    if gb.exists():
        from ccrl.train.general import noninferior
        base_g = pd.read_parquet(gb)
        rows = []
        for arm, files in arms.items():
            for f in files:
                g = d / f"general_{Path(f).stem}.parquet"
                if g.exists():
                    rows += noninferior(pd.read_parquet(g), base_g, Path(f).stem)
        if rows:
            out.append(pd.DataFrame(rows))
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
