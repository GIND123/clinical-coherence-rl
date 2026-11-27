"""Estimator audit: does the TRAINING UPDATE follow the registered objective? (Amendment 4)

The synthetic suite (ccrl.reward.synthetic) checks the reward's expected VALUES: the KB-Bayes oracle
must beat every degenerate policy. It passed before the first decisive runs, and those runs still
collapsed, because the failure was in the gradient estimator, not in the objective
(docs/training_runs.md). This module checks the update itself, with no language model:

  * a policy whose every rollout chooses three scalars by a Gaussian draw around learnable means:
      T_i  temperature             (T > 1 flattens the answer: the hedging channel)
      k_i  presentation sensitivity (how much the answer depends on the rendering)
      s_i  sampling noise
    and answers  q_i = softmax((log b + k_i * u_r + s_i * eps_i) / T_i),
    with b the patient's (imperfect) belief for the rendering's evidence, u_r a fixed random
    direction per rendering (a presentation effect), eps_i fresh noise;
  * the registered reward on DDXPlus training patients (paired groups x, x', x-, batch of 16);
  * policy-gradient training of the three means with either
      credit="grpo"  the registered per-rendering centred advantages of R_i, or
      credit="team"  the same advantages of D_i = J(y) - J(y without i) (team credit).

The score of a Gaussian action is xi / nu, a scalar per knob, so the estimate is low-variance and
the audit runs on a CPU in about a minute.

    python -m ccrl.reward.audit            prints both trajectories; exit 1 if a check fails
"""
from __future__ import annotations

import json
import sys

import numpy as np

from ccrl.analysis.metrics import js_rows
from ccrl.reward.coherence import (PatientGroup, RewardConfig, advantages, batch_rewards, cross_patient_information,
                                   patient_mean, team_credit)

KNOBS = ("log_T", "log_k", "log_s")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_patients(n: int = 1024, omega: float = 0.3, zeta: float = 0.5, p_wrong: float = 0.65,
                  seed: int = 0) -> tuple[list[dict], int]:
    """DDXPlus training patients with an imperfect, overconfident belief, like the untrained
    Qwen3-4B (top-1 0.33 with a mean top probability of 0.67, docs/training_runs.md):
      logits = omega * log(oracle with gold and one wrong diagnosis swapped, for a share p_wrong
               of patients: a systematic misdiagnosis held with the oracle's confidence) + eta_p.
    The same swap and eta apply to the patient's full and triage renderings."""
    from ccrl import paths
    from ccrl.data import ddxplus as dx
    from ccrl.util import read_jsonl
    lik = dx.load_likelihood()
    cases = [c for c in read_jsonl(paths.FROZEN_DDX / "cases.jsonl") if c["role"] == "distill"][:n]
    rng = np.random.default_rng(seed)
    out = []
    for c in cases:
        units = [u["tokens"] for u in c["units"]]
        full = lik.posterior([t for u in units for t in u])
        tri = lik.posterior(units[0])
        gold = lik.pathologies.index(c["pathology"])
        perm = np.arange(lik.D)
        if rng.random() < p_wrong:
            w = int(rng.choice([j for j in range(lik.D) if j != gold]))
            perm[[gold, w]] = perm[[w, gold]]
        eta = rng.normal(0.0, zeta, lik.D)
        out.append({"b_full": omega * np.log(np.clip(full[perm], 1e-8, 1)) + eta,
                    "b_tri": omega * np.log(np.clip(tri[perm], 1e-8, 1)) + eta,
                    "gold": gold,
                    "added": lik.tau * lik.unit_loglik([t for u in units[1:] for t in u])})
    return out, lik.D


def _sample_batch(pats, K, theta, nu, rng, B, G, H):
    idx = rng.choice(len(pats), B, replace=False)
    groups, xis = [], []
    for j in idx:
        p = pats[j]
        rend = []
        for base, n in ((p["b_full"], G), (p["b_full"], G), (p["b_tri"], H)):
            u = rng.standard_normal(K)
            xi = rng.standard_normal((n, 3))
            a = theta[None] + nu * xi
            T, k, s = np.exp(a[:, 0:1]), np.exp(a[:, 1:2]), np.exp(a[:, 2:3])
            z = (base[None] + k * u[None] + s * rng.standard_normal((n, K))) / T
            rend.append([list(q) for q in _softmax(z)])
            xis.append(xi)
        groups.append(PatientGroup(rend[0], rend[1], rend[2], gold=p["gold"], added_loglik=p["added"], K=K))
    return groups, np.concatenate(xis)


def gradient(pats, K, theta, cfg: RewardConfig, credit: str, rng, nu=0.3, B=16, G=4, H=2,
             n_batches=4) -> tuple[np.ndarray, dict]:
    """Monte-Carlo estimate of the expected update direction for the three knob means."""
    g = np.zeros(3)
    n = 0
    st = {"M": [], "reward": [], "js_xx": [], "entropy": [], "top1": []}
    for _ in range(n_batches):
        groups, xi = _sample_batch(pats, K, theta, nu, rng, B, G, H)
        if credit == "team":
            sig, res, M = team_credit(groups, cfg, K)
        elif credit == "grpo":
            res, M = batch_rewards(groups, cfg, K)
            sig = [r.total for r in res]
        else:
            raise ValueError(credit)
        k0 = 0
        for gp, s, r in zip(groups, sig, res):
            m = len(s)
            adv = advantages(s, True, 1e-4, sizes=(len(gp.q_x), len(gp.q_xp), len(gp.q_xm)))
            if adv is not None:
                g += (adv[:, None] * xi[k0:k0 + m] / nu).sum(0)
            n += m
            k0 += m
            Qx, Qp = np.array(gp.q_x), np.array(gp.q_xp)
            st["js_xx"].append(float(js_rows(Qx, np.repeat(Qp.mean(0)[None], len(Qx), 0)).mean()))
            Q = np.concatenate([Qx, Qp])
            st["entropy"].append(float(-(Q * np.log(np.clip(Q, 1e-12, 1))).sum(1).mean()))
            st["top1"].append(float((Q.argmax(1) == gp.gold).mean()))
            st["reward"].append(float(r.total.mean()))
        st["M"].append(M)
    return g / max(1, n), {k: float(np.mean(v)) for k, v in st.items()}


def simulate(pats, K, cfg: RewardConfig, credit: str, steps: int = 120, lr: float = 0.04,
             theta0=(0.0, 0.0, -0.3), seed: int = 0, n_batches: int = 2) -> list[dict]:
    """Adam ascent on the knob means; one row per step."""
    rng = np.random.default_rng(seed)
    th = np.array(theta0, dtype=float)
    m = np.zeros(3)
    v = np.zeros(3)
    rows = []
    for t in range(1, steps + 1):
        g, st = gradient(pats, K, th, cfg, credit, rng, n_batches=n_batches)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        th = th + lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        rows.append({"step": t, **{k: float(x) for k, x in zip(KNOBS, th)}, **st})
    return rows


def registered_cfg(**kw) -> RewardConfig:
    """The reward of the decisive coherence arm (config/train/base.yaml, ccr_acc_fid)."""
    base = dict(alpha=1.0, beta=1.0, gamma=0.5, tau_I=0.788, delta_S=0.234, m0=0.459, c_fid=3.0,
                sensitivity="batch", acc_mode="brier")
    base.update(kw)
    return RewardConfig(**base)


def summary(rows: list[dict], window: int = 15) -> dict:
    a, b = rows[:window], rows[-window:]
    f = lambda rs, k: float(np.mean([r[k] for r in rs]))  # noqa: E731
    return {k: {"start": f(a, k), "end": f(b, k)} for k in ("log_T", "log_k", "log_s", "M", "js_xx", "entropy", "top1",
                                                         "reward")}


def run_audit(steps: int = 150, verbose: bool = True, seed: int = 0) -> tuple[bool, dict]:
    pats, K = load_patients()
    cfg = registered_cfg()
    out = {c: simulate(pats, K, cfg, c, steps=steps, seed=seed) for c in ("grpo", "team")}
    s = {c: summary(r) for c, r in out.items()}
    g, t = s["grpo"], s["team"]
    uniform = float(np.log(K))
    checks = {
        "registered credit reproduces the collapse (M < 0.3 m0, entropy > 0.9 log K)":
            g["M"]["end"] < 0.3 * cfg.m0 and g["entropy"]["end"] > 0.9 * uniform,
        "team credit holds the registered constraint (M >= 0.95 m0)": t["M"]["end"] >= 0.95 * cfg.m0,
        "team credit does not hedge (entropy rises < 0.5 nats)": t["entropy"]["end"] < t["entropy"]["start"] + 0.5,
        "team credit removes presentation sensitivity (log k falls > 0.5)": t["log_k"]["end"] < t["log_k"]["start"] - 0.5,
        "team credit halves the x/x' divergence": t["js_xx"]["end"] < 0.5 * t["js_xx"]["start"],
        "team credit reaches a higher registered objective": t["reward"]["end"] > g["reward"]["end"],
    }
    ok = all(checks.values())
    if verbose:
        for c in ("grpo", "team"):
            print(f"== credit={c}")
            for k, v in s[c].items():
                print(f"   {k:<8} {v['start']:>8.3f} -> {v['end']:>8.3f}")
        for k, v in checks.items():
            print(f"[{'PASS' if v else 'FAIL'}] {k}")
    return ok, {"summary": s, "checks": checks, "trajectories": out}


if __name__ == "__main__":
    ok, res = run_audit()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            json.dump(res, f, default=float)
    sys.exit(0 if ok else 1)
