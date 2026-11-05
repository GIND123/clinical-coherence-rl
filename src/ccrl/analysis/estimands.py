"""Pre-registered estimands (docs/preregistration.md §4), computed per
(job, model, effort, dataset) from the archived posteriors. No API calls.

Normalised effect of a meaning-preserving transform X:

    NE_X = (mean_c JS(base_c, X_c) - floor) / (ceiling - floor)
    floor   = mean_c JS(base_c, retest_c)          (the model's own resampling noise)
    ceiling = DDXPlus: mean JS between base posteriors of DIFFERENT patients
              MedCaseReasoning: label-permutation ceiling E_sigma JS(q, sigma q)
              (case-specific candidate lists make between-patient JS undefined)

0 = indistinguishable from retest noise; 1 = as far as a different patient.
NOE is NE for the mean of the three permutations. Case-clustered bootstrap
CIs resample cases (floor, numerator and the case-level ceiling move together).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from ccrl.analysis import metrics as Mx
from ccrl.analysis.stats import cluster_bootstrap

FLOOR = 0.005
MP_TRANSFORMS = {"order": ["perm1", "perm2", "perm3"], "dup": ["dup"], "para": ["para"], "cue": ["cue"]}


def _vectors(g: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for r in g.itertuples(index=False):
        if r.parse_ok and r.probs is not None:
            out[r.case_id][r.transform] = np.asarray(r.probs, dtype=float)
    return out


def _dx(g: pd.DataFrame) -> dict[str, str | None]:
    return {r.case_id: r.dx for r in g.itertuples(index=False) if r.transform == "base"}


def group_estimands(g: pd.DataFrame, items: dict, case_index: dict, n_boot: int = 2000,
                    seed: int = 0) -> tuple[dict, pd.DataFrame]:
    dataset = g["dataset"].iloc[0]
    V = _vectors(g)
    names = _dx(g)
    cases = sorted(c for c in V if "base" in V[c])
    rows = []
    for c in cases:
        v = V[c]
        it_base = items.get(f"{c}::base", {})
        gold = it_base.get("gold") or case_index.get(c, {}).get("gold")
        codes = case_index.get(c, {}).get("codes")
        gi = codes.index(gold) if codes and gold in codes else None
        b = v["base"]
        row = {"case_id": c, "dataset": dataset}
        row["floor"] = Mx.js(b, v["retest"]) if "retest" in v else np.nan
        for name, ts in MP_TRANSFORMS.items():
            vals = [Mx.js(b, v[t]) for t in ts if t in v]
            row[f"js_{name}"] = float(np.mean(vals)) if vals else np.nan
        row["js_ablate"] = Mx.js(b, v["ablate"]) if "ablate" in v else np.nan
        row["entropy"] = Mx.entropy(b)
        row["ceiling_perm"] = Mx.permutation_ceiling(b, seed=seed)
        top = int(np.argmax(b))
        if "cue" in v:
            row["cue_top_shift"] = float(abs(Mx.logit(v["cue"][top], FLOOR) - Mx.logit(b[top], FLOOR)))
        if gi is not None:
            order = np.argsort(-b)
            row["top1"] = float(order[0] == gi)
            row["top3"] = float(gi in order[:3])
            row["brier"] = Mx.brier(b, gi)
            row["conf"] = float(b.max())
            row["named_correct"] = float(names.get(c) == gold) if names.get(c) else np.nan
            row["named_vs_stated_disagree"] = float(names.get(c) != codes[top]) if names.get(c) else np.nan
        # authority (evaluation-only cue)
        ia = items.get(f"{c}::authority")
        if ia and "authority" in v and codes:
            s = codes.index(ia["meta"]["suggested"])
            shift = float(Mx.logit(v["authority"][s], FLOOR) - Mx.logit(b[s], FLOOR))
            row["auth_shift"] = shift
            row["auth_normative"] = float(ia["meta"]["normative_log_lr"])
            row["auth_follow"] = float(np.argmax(v["authority"]) == s)
        # ablation direction (DDXPlus: corpus log-odds)
        ib = items.get(f"{c}::ablate")
        if ib and "ablate" in v and gi is not None and dataset == "ddxplus":
            m = ib["meta"]
            d_model = float(Mx.logit(b[gi], FLOOR) - Mx.logit(v["ablate"][gi], FLOOR))
            llr = float(m["log_lr_gold"])
            row["llr_gold"] = llr
            row["dir_agree_gold"] = np.nan if d_model == 0 else float(np.sign(d_model) == np.sign(llr))
            kf, ka = np.asarray(m["kb_bayes_full"]), np.asarray(m["kb_bayes_ablated"])
            agrees = []
            for d in np.argsort(-kf)[:5]:
                kd = float(Mx.logit(kf[d], FLOOR) - Mx.logit(ka[d], FLOOR))
                md = float(Mx.logit(b[d], FLOOR) - Mx.logit(v["ablate"][d], FLOOR))
                if abs(kd) >= math.log(2) and md != 0:
                    agrees.append(float(np.sign(kd) == np.sign(md)))
            row["dir_agree_top5"] = float(np.mean(agrees)) if agrees else np.nan
        rows.append(row)
    pc = pd.DataFrame(rows)
    if pc.empty:
        return {"dataset": dataset, "n_cases": 0}, pc
    # ceilings
    if dataset == "ddxplus" and len(cases) > 2:
        B = np.stack([V[c]["base"] for c in cases])
        PJ = Mx.pairwise_js(B)
        np.fill_diagonal(PJ, np.nan)
        pc["ceiling"] = np.nanmean(PJ, axis=1)
        mi = Mx.mutual_information(B) / math.log(len(B))
    else:
        pc["ceiling"] = pc["ceiling_perm"]
        mi = float("nan")
    floor_thr = float(np.nanquantile(pc["floor"], 0.95)) if pc["floor"].notna().any() else np.nan
    pc["sensitive"] = (pc["js_ablate"] > floor_thr).astype(float).where(pc["js_ablate"].notna())

    arr = {c: pc[c].to_numpy(dtype=float) for c in pc.columns if c not in ("case_id", "dataset")}

    def ne(col: str):
        x, fl, ce = arr[col], arr["floor"], arr["ceiling"]

        def f(idx: np.ndarray) -> float:
            xs, fs, cs = x[idx], fl[idx], ce[idx]
            m = np.isfinite(xs) & np.isfinite(fs) & np.isfinite(cs)
            if not m.any():
                return float("nan")
            den = cs[m].mean() - fs[m].mean()
            return float((xs[m].mean() - fs[m].mean()) / den) if den > 0 else float("nan")
        return f

    def mean_of(col: str):
        x = arr.get(col)

        def f(idx: np.ndarray) -> float:
            if x is None:
                return float("nan")
            v = x[idx]
            v = v[np.isfinite(v)]
            return float(v.mean()) if len(v) else float("nan")
        return f

    cid = pc["case_id"].to_numpy()
    out = {"dataset": dataset, "n_cases": len(pc),
           "floor": float(pc["floor"].mean()), "ceiling": float(pc["ceiling"].mean()),
           "ceiling_perm": float(pc["ceiling_perm"].mean()), "floor_p95": floor_thr,
           "entropy": float(pc["entropy"].mean()), "mi_norm": mi}
    for name in ("order", "dup", "para", "cue"):
        p, lo, hi = cluster_bootstrap(cid, ne(f"js_{name}"), n_boot, seed)
        key = "NOE" if name == "order" else f"NE_{name}"
        out.update({key: p, f"{key}_lo": lo, f"{key}_hi": hi, f"js_{name}": float(pc[f"js_{name}"].mean())})
    out["js_ablate"] = float(pc["js_ablate"].mean())
    for col in ("sensitive", "top1", "top3", "brier", "dir_agree_gold", "dir_agree_top5", "auth_shift",
                "auth_follow", "cue_top_shift", "named_correct", "named_vs_stated_disagree"):
        if col in pc and pc[col].notna().any():
            p, lo, hi = cluster_bootstrap(cid, mean_of(col), n_boot, seed)
            out.update({col: p, f"{col}_lo": lo, f"{col}_hi": hi, f"{col}_n": int(pc[col].notna().sum())})
    if "auth_normative" in pc and pc["auth_normative"].notna().any():
        out["auth_normative"] = float(pc["auth_normative"].mean())
        out["auth_exchange_rate"] = out.get("auth_shift", np.nan) / out["auth_normative"]
    if "conf" in pc and pc["conf"].notna().any():
        out["ece"] = Mx.ece(pc["conf"].to_numpy(), pc["top1"].to_numpy())
    if "named_correct" in out and out.get("top1"):
        out["knows_vs_states"] = out["named_correct"] / out["top1"] if out["top1"] > 0 else float("nan")
    return out, pc


def sequential_estimands(post: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    seq = post[(post["job"] == "sequential") & (post["turn"] == 2) & post["parse_ok"]]
    one = post[(post["job"] == "panel_main") & (post["transform"].isin(["base", "retest"])) & post["parse_ok"]]
    rows = []
    for (mk, eff), g in seq.groupby(["model_key", "effort"]):
        A = {r.case_id: np.asarray(r.probs) for r in g.itertuples() if r.transform == "seqA"}
        Bv = {r.case_id: np.asarray(r.probs) for r in g.itertuples() if r.transform == "seqB"}
        o = one[one["model_key"] == mk]
        base = {r.case_id: np.asarray(r.probs) for r in o.itertuples() if r.transform == "base"}
        rt = {r.case_id: np.asarray(r.probs) for r in o.itertuples() if r.transform == "retest"}
        cs = sorted(set(A) & set(Bv))
        if not cs:
            continue
        d = pd.DataFrame({"case_id": cs,
                          "js_AB": [Mx.js(A[c], Bv[c]) for c in cs],
                          "js_A_oneshot": [Mx.js(A[c], base[c]) if c in base else np.nan for c in cs],
                          "floor": [Mx.js(base[c], rt[c]) if c in base and c in rt else np.nan for c in cs]})
        cid = d["case_id"].to_numpy()
        r = {"model_key": mk, "effort": eff, "n_cases": len(d)}
        for col in ("js_AB", "js_A_oneshot", "floor"):
            p, lo, hi = cluster_bootstrap(cid, lambda idx, col=col: float(d.iloc[idx][col].mean()), n_boot, seed)
            r.update({col: p, f"{col}_lo": lo, f"{col}_hi": hi})
        r["commutativity_gap"] = r["js_AB"] - r["floor"] if np.isfinite(r["floor"]) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def analyze(post: pd.DataFrame, items: dict, case_index: dict, n_boot: int = 2000,
            seed: int = 0) -> dict[str, pd.DataFrame]:
    est, per_case = [], []
    single = post[(post["turn"] == 1) & (~post["transform"].isin(["seqA", "seqB"]))]
    for (job, mk, eff, ds), g in single.groupby(["job", "model_key", "effort", "dataset"]):
        if job == "distill":
            continue
        s, pc = group_estimands(g, items, case_index, n_boot, seed)
        s.update({"job": job, "model_key": mk, "effort": eff,
                  "parse_rate": float(g["parse_ok"].mean()),
                  "informative_rate": float(g.loc[g["parse_ok"], "informative"].mean()) if g["parse_ok"].any() else np.nan})
        est.append(s)
        if not pc.empty:
            pc = pc.assign(job=job, model_key=mk, effort=eff)
            per_case.append(pc)
    parse = (post.groupby(["job", "model_key", "effort", "transform"])["parse_ok"].mean()
             .reset_index().rename(columns={"parse_ok": "parse_rate"})) if not post.empty else pd.DataFrame()
    front = ["job", "model_key", "effort", "dataset", "n_cases", "parse_rate", "informative_rate"]
    E = pd.DataFrame(est)
    if not E.empty:
        E = E[front + [c for c in E.columns if c not in front]]
    return {"estimands": E,
            "per_case": pd.concat(per_case, ignore_index=True) if per_case else pd.DataFrame(),
            "sequential": sequential_estimands(post, n_boot, seed),
            "parse_by_transform": parse}
