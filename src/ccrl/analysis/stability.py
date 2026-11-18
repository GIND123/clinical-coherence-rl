"""Stability analysis: how much of a model's incoherence is run-to-run noise?

The main panel (2026-11-17/18) showed that, at provider-default decoding, the
difference between two *identical* requests (the retest) is as large as the
difference between re-orderings, paraphrases or an added irrelevant sentence.
Normalised effects (NOE) are therefore near zero for most models, while the
raw variation a clinician would see is large. This module measures the noise
directly and asks what it costs:

reliability(post)            one row per (job, model, effort, dataset)
  retest_flip                share of cases whose top diagnosis differs between two identical requests
  floor_mean / floor_median  JS(base, retest), bits
  single_top1, ens_top1      accuracy of one answer vs the mean of all meaning-preserving answers
  single_brier, ens_brier    the same for the Brier score; their difference is exactly the
  variance_share             mean squared spread of the answers around their mean (Brier is a
                             squared error), reported as a share of the single-answer Brier
  auroc_conf / auroc_retest / auroc_disagree
                             how well 1 - stated confidence, the base-retest JS, and the
                             disagreement of the other answers with the base top-1 detect
                             that the base answer is wrong (0.5 = chance)

updating(post, items)        the `resample` arm: several base and several ablated answers
                             per case, so the evidence update is measured on MEAN posteriors
  dir_agree_mean             sign of the mean log-odds change on the gold diagnosis vs the corpus
                             log likelihood ratio of the removed finding (DDXPlus)
  detectable                 share of cases whose mean change exceeds twice its standard error
  slope_vs_reference         regression slope of the model's mean gold log-odds change on the
                             tempered-Bayes reference change (1 = reference-sized updates)

All intervals are case-clustered bootstrap 95% CIs (cases are resampled).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ccrl.analysis import metrics as Mx
from ccrl.util import two_letter_codes

FLOOR = 0.005
MEANING_PRESERVING = ("base", "perm1", "perm2", "perm3", "dup", "para", "cue", "retest")
N_BOOT = 2000


def auroc(score, positive) -> float:
    """P(score of a positive > score of a negative), ties counted 1/2."""
    s = np.asarray(score, float)
    y = np.asarray(positive, bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _boot_ci(per_case: pd.DataFrame, fn, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(per_case)
    if n < 3:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(per_case.iloc[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _codes(probs: np.ndarray, codes: list[str] | None) -> list[str]:
    return codes if codes is not None and len(codes) == len(probs) else two_letter_codes(len(probs))


def _single_turn(post: pd.DataFrame) -> pd.DataFrame:
    p = post
    if "turn" in p:
        p = p[p["turn"] == 1]
    p = p[p["parse_ok"].astype(bool) & p["probs"].notna()]
    return p[~p["transform"].isin(["seqA", "seqB"])]


def _case_vectors(g: pd.DataFrame) -> dict[str, dict[str, list[np.ndarray]]]:
    """case -> transform -> list of posteriors (replicates appended in order)."""
    out: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    cols = ["case_id", "transform", "probs"] + (["replicate"] if "replicate" in g else [])
    rows = g[cols].copy()
    if "replicate" in rows:
        rows = rows.sort_values("replicate", kind="stable")
    for r in rows.itertuples(index=False):
        out[r.case_id][r.transform].append(np.asarray(r.probs, float))
    return out


def reliability_per_case(g: pd.DataFrame, codes_of: dict | None = None) -> pd.DataFrame:
    gold_of = g.groupby("case_id")["gold"].first().to_dict()
    rows = []
    for cid, v in _case_vectors(g).items():
        if not v.get("base") or not v.get("retest"):
            continue
        b, rt = v["base"][0], v["retest"][0]
        codes = _codes(b, (codes_of or {}).get(cid))
        gold = gold_of.get(cid)
        gi = codes.index(gold) if gold in codes else None
        samples = [x for t in MEANING_PRESERVING for x in v.get(t, [])]
        S = np.stack(samples)
        mean = S.mean(0)
        top_b = int(np.argmax(b))
        others = [x for t in MEANING_PRESERVING if t != "base" for x in v.get(t, [])]
        row = {"case_id": cid, "n_samples": len(samples),
               "retest_flip": float(top_b != int(np.argmax(rt))),
               "floor": Mx.js(b, rt),
               "disagree": float(np.mean([int(np.argmax(x)) != top_b for x in others])) if others else np.nan,
               "conf": float(b.max())}
        if gi is not None:
            row.update({"base_err": float(top_b != gi),
                        "single_top1": float(np.mean([int(np.argmax(x)) == gi for x in samples])),
                        "ens_top1": float(int(np.argmax(mean)) == gi),
                        "single_brier": float(np.mean([Mx.brier(x, gi) for x in samples])),
                        "ens_brier": float(Mx.brier(mean, gi))})
        rows.append(row)
    return pd.DataFrame(rows)


def _summarise_reliability(pc: pd.DataFrame, n_boot: int, seed: int) -> dict:
    out: dict = {"n_cases": int(len(pc))}
    if pc.empty:
        return out
    for col in ("retest_flip", "floor"):
        out[col] = float(pc[col].mean())
        out[f"{col}_lo"], out[f"{col}_hi"] = _boot_ci(pc, lambda d, c=col: float(d[c].mean()), n_boot, seed)
    out["floor_median"] = float(pc["floor"].median())
    out["n_samples_mean"] = float(pc["n_samples"].mean())
    if "single_top1" in pc:
        lab = pc.dropna(subset=["single_top1"])
        for col in ("single_top1", "ens_top1", "single_brier", "ens_brier"):
            out[col] = float(lab[col].mean())
        out["ens_gain_top1"] = out["ens_top1"] - out["single_top1"]
        out["ens_gain_top1_lo"], out["ens_gain_top1_hi"] = _boot_ci(
            lab, lambda d: float((d["ens_top1"] - d["single_top1"]).mean()), n_boot, seed)
        out["variance_share"] = (out["single_brier"] - out["ens_brier"]) / out["single_brier"] \
            if out["single_brier"] > 0 else np.nan
        for name, score in (("conf", 1 - lab["conf"]), ("retest", lab["floor"]), ("disagree", lab["disagree"])):
            out[f"auroc_{name}"] = auroc(score, lab["base_err"] == 1)
        out["auroc_disagree_minus_conf"] = out["auroc_disagree"] - out["auroc_conf"]
        out["auroc_disagree_minus_conf_lo"], out["auroc_disagree_minus_conf_hi"] = _boot_ci(
            lab, lambda d: auroc(d["disagree"], d["base_err"] == 1) - auroc(1 - d["conf"], d["base_err"] == 1),
            n_boot, seed)
    return out


def reliability(post: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0,
                codes_of: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary per job x model x effort x dataset, per-case table)."""
    p = _single_turn(post)
    if "replicate" in p:
        p = p[p["replicate"].fillna(0) == 0]          # replicate arms are handled by updating()
    summ, per = [], []
    for (job, mk, eff, ds), g in p.groupby(["job", "model_key", "effort", "dataset"]):
        pc = reliability_per_case(g, codes_of)
        if pc.empty:
            continue
        s = _summarise_reliability(pc, n_boot, seed)
        s.update({"job": job, "model_key": mk, "effort": eff, "dataset": ds})
        summ.append(s)
        per.append(pc.assign(job=job, model_key=mk, effort=eff, dataset=ds))
    cols = ["job", "model_key", "effort", "dataset"]
    S = pd.DataFrame(summ)
    if not S.empty:
        S = S[cols + [c for c in S.columns if c not in cols]]
    return S, (pd.concat(per, ignore_index=True) if per else pd.DataFrame())


def updating(post: pd.DataFrame, items: dict | None = None, job: str = "resample",
             base_jobs: tuple[str, ...] = ("panel_main",), n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Evidence updating on mean posteriors (the `resample` arm).

    Base samples: panel `base` and `retest` plus every resample `base`
    replicate; ablated samples: panel `ablate` plus every resample `ablate`
    replicate. Needs the items (ablation metadata) for direction and the
    reference comparison; without them only `detectable` is computed."""
    p = _single_turn(post)
    p = p[p["job"].isin((job,) + tuple(base_jobs))]
    if p.empty or not (p["job"] == job).any():
        return pd.DataFrame()
    out = []
    for (mk, ds), g in p.groupby(["model_key", "dataset"]):
        if not (g["job"] == job).any():
            continue
        gold_of = g.groupby("case_id")["gold"].first().to_dict()
        rows = []
        for cid, v in _case_vectors(g).items():
            B = v.get("base", []) + v.get("retest", [])
            A = v.get("ablate", [])
            if len(B) < 2 or len(A) < 2:
                continue
            codes = _codes(B[0], None)
            gold = gold_of.get(cid)
            if gold not in codes:
                continue
            gi = codes.index(gold)
            lb = np.array([float(Mx.logit(x[gi], FLOOR)) for x in B])
            la = np.array([float(Mx.logit(x[gi], FLOOR)) for x in A])
            d = lb.mean() - la.mean()                      # removing the finding lowers gold log-odds by d
            se = float(np.sqrt(lb.var(ddof=1) / len(lb) + la.var(ddof=1) / len(la)))
            mb, ma = np.mean(B, 0), np.mean(A, 0)
            row = {"case_id": cid, "k_base": len(B), "k_ablate": len(A), "d_model": float(d), "se": se,
                   "detectable": float(abs(d) > 2 * se) if se > 0 else float(d != 0),
                   "js_means": Mx.js(mb, ma)}
            meta = ((items or {}).get(f"{cid}::ablate") or {}).get("meta") or {}
            if ds == "ddxplus" and "log_lr_gold" in meta:
                llr = float(meta["log_lr_gold"])
                row["llr_gold"] = llr
                row["dir_agree_mean"] = np.nan if d == 0 else float(np.sign(d) == np.sign(llr))
                if "kb_bayes_full" in meta and "kb_bayes_ablated" in meta:
                    kf, ka = np.asarray(meta["kb_bayes_full"]), np.asarray(meta["kb_bayes_ablated"])
                    row["d_reference"] = float(Mx.logit(kf[gi], FLOOR) - Mx.logit(ka[gi], FLOOR))
            rows.append(row)
        pc = pd.DataFrame(rows)
        if pc.empty:
            continue
        s = {"model_key": mk, "dataset": ds, "n_cases": len(pc), "k_base_mean": float(pc["k_base"].mean()),
             "k_ablate_mean": float(pc["k_ablate"].mean()), "detectable": float(pc["detectable"].mean()),
             "js_means": float(pc["js_means"].mean())}
        s["detectable_lo"], s["detectable_hi"] = _boot_ci(pc, lambda x: float(x["detectable"].mean()), n_boot, seed)
        if "dir_agree_mean" in pc and pc["dir_agree_mean"].notna().any():
            s["dir_agree_mean"] = float(pc["dir_agree_mean"].mean())
            s["dir_agree_mean_lo"], s["dir_agree_mean_hi"] = _boot_ci(
                pc, lambda x: float(x["dir_agree_mean"].mean()), n_boot, seed)
            sgn = np.sign(pc["llr_gold"])
            s["mean_signed_update"] = float((sgn * pc["d_model"]).mean())      # > 0: right direction on average
        if "d_reference" in pc and pc["d_reference"].notna().sum() > 2:
            x, y = pc["d_reference"].to_numpy(), pc["d_model"].to_numpy()
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() > 2 and np.var(x[ok]) > 0:
                s["slope_vs_reference"] = float(np.polyfit(x[ok], y[ok], 1)[0])
                s["corr_vs_reference"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
        out.append(s)
    return pd.DataFrame(out)
