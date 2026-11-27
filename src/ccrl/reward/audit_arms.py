"""Estimator audit with the readout channel, and its predictions for every GPU arm
(plan_final.md P6 / claim C7, 27 Nov 2026; committed before any run of stage A).

ccrl.reward.audit models three knobs: temperature T, presentation sensitivity k and noise s.
The second collapse mechanism (Amendment 4) needs a fourth:

  f  filler: the policy writes 100 q_c + 100 f on every code c, so its percentages sum to
     100 (1 + K f). The registered (lenient) readout renormalises, so the parsed answer is
     (q + f) / (1 + K f), flattened at no cost. The strict readout counts the answer only if
     |sum - 100| <= 2, i.e. K f <= 0.02; otherwise it is invalid (reward 0).

At the start, 30% of rollouts comply (the untrained Qwen3-4B: docs/training_runs.md), which sets
the initial mean of log f. Every arm of stage A is simulated with its reward, credit and readout.
The prediction is the direction of each statistic over 150 steps (start vs end, 15-step means):
M, entropy, x/x' divergence, filler compliance, and whether the 20-step collapse safeguard fires.

    python -m ccrl.reward.audit_arms [out.json]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from ccrl.analysis.metrics import js_rows
from ccrl.reward.audit import _softmax, load_patients, registered_cfg
from ccrl.reward.coherence import PatientGroup, RewardConfig, advantages, batch_rewards, team_credit

KNOBS = ("log_T", "log_k", "log_s", "log_f")
NU = np.array([0.3, 0.3, 0.3, 1.0])
TOL = 0.02                                   # strict readout: |sum - 100| <= 2 percentage points


def theta0(K: int, comply: float = 0.30) -> np.ndarray:
    from scipy.stats import norm
    # P(K f <= TOL) = comply with log f ~ N(mu, NU_f)
    mu = np.log(TOL / K) - norm.ppf(comply) * NU[3]
    return np.array([0.0, 0.0, -0.3, mu])


def _answers(base, n, K, theta, rng, readout):
    u = rng.standard_normal(K)
    xi = rng.standard_normal((n, 4))
    a = theta[None] + NU[None] * xi
    T, k, s, f = (np.exp(a[:, i:i + 1]) for i in range(4))
    q = _softmax((base[None] + k * u[None] + s * rng.standard_normal((n, K))) / T)
    comply = (K * f[:, 0]) <= TOL
    if readout == "lenient":
        rows = [list((qi + fi) / (1 + K * fi)) for qi, fi in zip(q, f[:, 0])]
    elif readout == "strict":
        rows = [list(qi) if ok else None for qi, ok in zip(q, comply)]
    else:
        raise ValueError(readout)
    return rows, xi, comply


def gradient(pats, K, theta, cfg: RewardConfig, credit: str, readout: str, rng, B=16, G=4, H=2, n_batches=2):
    g = np.zeros(4)
    n = 0
    st = {"M": [], "reward": [], "js_xx": [], "entropy": [], "top1": [], "comply": []}
    for _ in range(n_batches):
        idx = rng.choice(len(pats), B, replace=False)
        groups, xis = [], []
        for j in idx:
            p = pats[j]
            rend = []
            for base, m in ((p["b_full"], G), (p["b_full"], G), (p["b_tri"], H)):
                rows, xi, comply = _answers(base, m, K, theta, rng, readout)
                rend.append(rows)
                xis.append(xi)
                st["comply"].append(float(comply.mean()))
            groups.append(PatientGroup(rend[0], rend[1], rend[2], gold=p["gold"], added_loglik=p["added"], K=K))
        xi = np.concatenate(xis)
        if credit == "team":
            sig, res, M = team_credit(groups, cfg, K)
        else:
            res, M = batch_rewards(groups, cfg, K)
            sig = [r.total for r in res]
        k0 = 0
        for gp, sgl, r in zip(groups, sig, res):
            m = len(sgl)
            adv = advantages(sgl, True, 1e-4, sizes=(len(gp.q_x), len(gp.q_xp), len(gp.q_xm)))
            if adv is not None:
                g += (adv[:, None] * xi[k0:k0 + m] / NU[None]).sum(0)
            n += m
            k0 += m
            Qx = np.array([q for q in gp.q_x if q is not None])
            Qp = np.array([q for q in gp.q_xp if q is not None])
            if len(Qx) and len(Qp):
                st["js_xx"].append(float(js_rows(Qx, np.repeat(Qp.mean(0)[None], len(Qx), 0)).mean()))
                Q = np.concatenate([Qx, Qp])
                st["entropy"].append(float(-(Q * np.log(np.clip(Q, 1e-12, 1))).sum(1).mean()))
                st["top1"].append(float((Q.argmax(1) == gp.gold).mean()))
            st["reward"].append(float(r.total.mean()))
        # M as the registered safeguard sees it: lenient readout of the same answers
        st["M"].append(float(M))
    return g / max(1, n), {k: float(np.mean(v)) if v else float("nan") for k, v in st.items()}


def simulate(pats, K, cfg, credit, readout, steps=150, lr=0.04, seed=0):
    rng = np.random.default_rng(seed)
    th = theta0(K)
    m1 = np.zeros(4)
    v = np.zeros(4)
    rows = []
    for t in range(1, steps + 1):
        g, st = gradient(pats, K, th, cfg, credit, readout, rng)
        m1 = 0.9 * m1 + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        th = th + lr * (m1 / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        rows.append({"step": t, **{k: float(x) for k, x in zip(KNOBS, th)}, **st})
    return rows


def safeguard_fires(rows, baseline=10, window=20, drop=0.30) -> int | None:
    M = [r["M"] for r in rows]
    for t in range(max(baseline, window) + 1, len(M) + 1):
        b, c = np.mean(M[:baseline]), np.mean(M[t - window:t])
        if b > 0 and c < (1 - drop) * b:
            return t
    return None


ARMS = {
    # name: (reward overrides, credit, readout)
    "attempt2 (registered credit, lenient)": ({}, "grpo", "lenient"),
    "cell A (team credit, lenient)": ({}, "team", "lenient"),
    "cell B (registered credit, strict)": ({}, "grpo", "strict"),
    "CCR (team credit, strict)": ({}, "team", "strict"),
    "RLVR (strict)": ({"beta": 0.0, "gamma": 0.0, "acc_mode": "binary"}, "team", "strict"),
    "Brier-only (strict)": ({"beta": 0.0, "gamma": 0.0, "acc_mode": "brier"}, "team", "strict"),
}


def predict(steps=150, seed=0, verbose=True) -> dict:
    pats, K = load_patients()
    out = {}
    for name, (over, credit, readout) in ARMS.items():
        rows = simulate(pats, K, registered_cfg(**over), credit, readout, steps=steps, seed=seed)
        a, b = rows[:15], rows[-15:]
        f = lambda rs, k: float(np.nanmean([r[k] for r in rs]))  # noqa: E731
        s = {k: {"start": f(a, k), "end": f(b, k)} for k in ("M", "entropy", "js_xx", "comply", "top1", "reward",
                                                           "log_T", "log_k", "log_f")}
        fire = safeguard_fires(rows)
        pred = {
            "collapse (safeguard fires)": fire is not None,
            "safeguard step": fire,
            "M": "down" if s["M"]["end"] < 0.85 * s["M"]["start"] else ("up" if s["M"]["end"] > 1.15 * s["M"]["start"] else "flat"),
            "entropy": "up" if s["entropy"]["end"] > s["entropy"]["start"] + 0.25 else
                       ("down" if s["entropy"]["end"] < s["entropy"]["start"] - 0.25 else "flat"),
            "x/x' divergence": "down" if s["js_xx"]["end"] < 0.8 * s["js_xx"]["start"] else
                               ("up" if s["js_xx"]["end"] > 1.2 * s["js_xx"]["start"] else "flat"),
            "sum-to-100 compliance": "up" if s["comply"]["end"] > s["comply"]["start"] + 0.1 else
                                     ("down" if s["comply"]["end"] < s["comply"]["start"] - 0.1 else "flat"),
        }
        out[name] = {"prediction": pred, "summary": s}
        if verbose:
            print(f"== {name}")
            for k, v in s.items():
                print(f"   {k:<8} {v['start']:>8.3f} -> {v['end']:>8.3f}")
            print("   prediction:", pred)
    return out


if __name__ == "__main__":
    res = predict()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=float)
