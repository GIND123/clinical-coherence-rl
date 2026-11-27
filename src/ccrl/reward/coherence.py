r"""The coherence-gated reward (docs/theory.md; plan §3.3), per rollout.

For patient p, a PAIRED GROUP of rollouts is sampled:
    G rollouts on the rendering x          (evidence set E)
    G rollouts on x' = T(x), T meaning-preserving (same E)
    H rollouts on the contrast rendering x- (reduced evidence E_r), H = G/2 by default.
    Default contrast: the TRIAGE rendering (context + chief complaint only).
    The plan's single-finding ablation is available as contrast="single" but is
    degenerate on DDXPlus (docs/theory.md, Finding 1): the reference posterior
    is near-deterministic and its single-finding ablation JS has median 0.

Terms (JS = Jensen-Shannon, base 2; q-bar = leave-one-out mean posterior):

  floor_i  = JS(q_i, q-bar(own rendering, without i))           the rollout's own noise
  I_i      = 1 - min(1, JS(q_i, q-bar(partner)) / tau_I)         partner: x <-> x'; x- rollouts
                                                                 use their own rendering (no partner)
  S_i      = min(1, max(0, JS(q_i, q-bar(contrast)) - floor_i) / delta_S)
                                                                 contrast: x, x' -> x-;  x- -> x
  M        = [H(mean_p q-bar_p) - mean_p H(q-bar_p)] / log B    batch-level cross-patient information
  G_S      = sqrt(min(1, mean_batch(excess contrast JS) / delta_S))  batch-level sensitivity gate
  R_coh_i  = I_i * G_S * min(1, M / m0)                   (default: sensitivity="batch")
  R_coh_i  = I_i * sqrt(S_i) * min(1, M / m0)             (plan's per-rollout form: sensitivity="rollout")
  R_fid_i  = 1 - min(1, err_i / c_fid)                    update fidelity (log-odds, plan Prop. 3):
             err_i = mean over the top-k diagnoses of q-hat of |logit(q_i) - logit(q-hat)|,
             logits clipped to [floor, 1-floor] (the 0.5%..99.5% output resolution);
             full-evidence rollouts:  q-hat ∝ max(q-bar(x-), floor) * L(E \ E_r | d)^tau
             contrast rollouts j:     compare q-bar(full) with  max(q_j, floor) * L(E \ E_r | d)^tau
             i.e. the model's reduced-evidence belief, updated by the corpus likelihood of the
             findings it lacked, must predict its full-evidence belief. Only this direction is
             numerically stable at the 0.5% output resolution (dividing by tiny likelihoods is not).
  R_acc_i  = 1 - ||q_i - e_y||^2 / 2                              (bounded proper; needs the label)
  R_i      = alpha R_acc + beta R_coh + gamma R_fid ; invalid output -> R_i = 0

Correction to the plan (recorded in docs/theory.md, "Correction 1"): the
sensitivity term subtracts the rollout's own noise floor. Without it a
STOCHASTIC policy earns sensitivity from its own sampling noise, so pure noise
would be rewarded by S; with it, S measures movement beyond retest noise.

Correction 2 ("Finding 2" in docs/theory.md): the plan's PER-ROLLOUT S rewards
movement in every patient, but under redundant evidence a Bayesian reasoner
correctly does not move for some patients (DDXPlus: the chief complaint alone
settles ~10% of cases). The synthetic suite shows the per-rollout form then
prefers an over-reacting (tempered) oracle to the Bayes oracle. Sensitivity is
therefore a BATCH-level anti-collapse gate by default, like M; per-rollout
credit comes from invariance, fidelity and accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ccrl.analysis.metrics import entropy, js_rows, norm


@dataclass
class RewardConfig:
    alpha: float = 1.0          # accuracy
    beta: float = 1.0           # coherence
    gamma: float = 0.5          # fidelity
    tau_I: float = 0.10         # set to the 95th pct of base-model retest JS in the pilot
    delta_S: float = 0.05       # median KB-Bayes ablation JS on training patients (computed offline)
    m0: float = 0.25            # 0.5 x KB-Bayes oracle M on DDXPlus (computed offline)
    c_fid: float = 3.0          # nats of mean |log-odds error| at which fidelity reaches 0
    fid_topk: int = 3
    prob_floor: float = 0.005
    sqrt_S: bool = True
    noise_floor_correction: bool = True
    sensitivity: str = "batch"   # "batch" (default) or "rollout" (plan's per-rollout form)
    acc_mode: str = "brier"      # brier: 1-||q-e_y||^2/2 | binary: 1[argmax=y] (RLVR) | rlcr: binary + brier
    use_invariance: bool = True  # ablation switches
    use_sensitivity: bool = True
    use_m_gate: bool = True


@dataclass
class PatientGroup:
    """Posteriors for one patient's paired group. Rows may be None (invalid output)."""
    q_x: list
    q_xp: list
    q_xm: list
    gold: int | None = None                  # label index (only used by R_acc)
    added_loglik: np.ndarray | None = None   # tau * sum_{t in E \ E_r} log P(t | d) (DDXPlus), else None
    K: int | None = None                     # this patient's number of candidates (real cases vary: 9 or 10)


@dataclass
class GroupRewards:
    total: np.ndarray                      # (n_rollouts,) in order x, x', x-
    coh: np.ndarray
    acc: np.ndarray
    fid: np.ndarray
    I: np.ndarray
    S: np.ndarray
    valid: np.ndarray
    terms: dict = field(default_factory=dict)


def _stack(rows: list, K: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.array([r is not None for r in rows], dtype=bool)
    Q = np.stack([norm(r) if r is not None else np.full(K, 1.0 / K) for r in rows]) if rows else np.zeros((0, K))
    return Q, valid


def _loo_means(Q: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Leave-one-out mean of the valid rows, per row (falls back to the full mean)."""
    n = valid.sum()
    tot = Q[valid].sum(0) if n else np.zeros(Q.shape[1])
    out = np.empty_like(Q)
    for i in range(len(Q)):
        if valid[i] and n > 1:
            out[i] = (tot - Q[i]) / (n - 1)
        elif n:
            out[i] = tot / n
        else:
            out[i] = np.full(Q.shape[1], 1.0 / Q.shape[1])
    return out


def _mean(Q: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return Q[valid].mean(0) if valid.any() else np.full(Q.shape[1], 1.0 / Q.shape[1])


def cross_patient_information(patient_means: list[np.ndarray]) -> float:
    """M: normalised mutual information between patient and (mean) output."""
    B = len(patient_means)
    if B < 2:
        return 0.0
    P = norm(np.stack(patient_means))
    mi = entropy(P.mean(0)) - float(np.mean([entropy(r) for r in P]))
    return float(max(0.0, mi) / math.log(B))


def patient_mean(g: PatientGroup, K: int) -> np.ndarray:
    Qx, vx = _stack(g.q_x, K)
    Qp, vp = _stack(g.q_xp, K)
    Q = np.concatenate([Qx, Qp])
    v = np.concatenate([vx, vp])
    return _mean(Q, v)


def group_rewards(g: PatientGroup, M: float, cfg: RewardConfig, K: int, gate_S: float | None = None) -> GroupRewards:
    Qx, vx = _stack(g.q_x, K)
    Qp, vp = _stack(g.q_xp, K)
    Qm, vm = _stack(g.q_xm, K)
    mean_x, mean_p, mean_m = _mean(Qx, vx), _mean(Qp, vp), _mean(Qm, vm)
    gate = (min(1.0, M / cfg.m0) if cfg.m0 > 0 else 1.0) if cfg.use_m_gate else 1.0

    excess: list[np.ndarray] = []

    def part(Q, valid, partner_mean, contrast_mean, own_loo):
        n = len(Q)
        if n == 0:
            excess.append(np.zeros(0))
            return np.zeros(0), np.zeros(0)
        floor = js_rows(Q, own_loo) if cfg.noise_floor_correction else np.zeros(n)
        # partner: one mean for all rows (x <-> x'), or one row per rollout (leave-one-out, x-)
        partner = partner_mean if partner_mean.ndim == 2 else np.repeat(partner_mean[None], n, 0)
        inv = js_rows(Q, partner)
        I = 1.0 - np.minimum(1.0, inv / cfg.tau_I)
        d = js_rows(Q, np.repeat(contrast_mean[None], n, 0)) - floor
        excess.append(np.where(valid, np.maximum(0.0, d), 0.0))
        S = np.minimum(1.0, np.maximum(0.0, d) / cfg.delta_S)
        return I, S

    loo_x, loo_p, loo_m = _loo_means(Qx, vx), _loo_means(Qp, vp), _loo_means(Qm, vm)
    Ix, Sx = part(Qx, vx, mean_p, mean_m, loo_x)
    Ip, Sp = part(Qp, vp, mean_x, mean_m, loo_p)
    # ablated rollouts: no meaning-preserving partner exists; invariance is to their own rendering,
    # measured against the LEAVE-ONE-OUT mean (a mean that includes the rollout itself would shrink
    # its divergence by ~1/H and inflate the reward; audit 27 Nov 2026)
    Im, Sm = part(Qm, vm, loo_m, 0.5 * (mean_x + mean_p), loo_m)
    I = np.concatenate([Ix, Ip, Im])
    S = np.concatenate([Sx, Sp, Sm])
    valid = np.concatenate([vx, vp, vm])
    Iu = I if cfg.use_invariance else np.ones_like(I)
    if not cfg.use_sensitivity:
        coh = Iu * gate
    elif cfg.sensitivity == "batch":
        gs = 1.0 if gate_S is None else gate_S
        coh = Iu * gs * gate
    else:
        coh = Iu * (np.sqrt(S) if cfg.sqrt_S else S) * gate

    Q = np.concatenate([Qx, Qp, Qm])
    acc = np.zeros(len(Q))
    if g.gold is not None:
        e = np.zeros(K)
        e[g.gold] = 1.0
        brier_r = 1.0 - ((Q - e) ** 2).sum(1) / 2.0
        binary = (Q.argmax(1) == g.gold).astype(float)
        acc = {"brier": brier_r, "binary": binary, "rlcr": 0.5 * (binary + brier_r)}[cfg.acc_mode]

    fid = np.zeros(len(Q))
    if g.added_loglik is not None:
        ll = np.asarray(g.added_loglik, dtype=float)
        w = np.exp(ll - ll.max())
        nfull = len(Qx) + len(Qp)
        lo, hi = cfg.prob_floor, 1.0 - cfg.prob_floor

        def predict(base: np.ndarray) -> np.ndarray:
            b = np.maximum(norm(base), cfg.prob_floor) * w
            return b / b.sum()

        def lgt(p: np.ndarray) -> np.ndarray:
            p = np.clip(p, lo, hi)
            return np.log(p / (1 - p))

        def err(actual: np.ndarray, pred: np.ndarray) -> float:
            sup = np.argsort(-pred)[: cfg.fid_topk]
            return float(np.mean(np.abs(lgt(actual[sup]) - lgt(pred[sup]))))

        full_mean = _mean(np.concatenate([Qx, Qp]), np.concatenate([vx, vp]))
        pred_full = predict(mean_m)
        for i in range(nfull):
            fid[i] = 1.0 - min(1.0, err(Q[i], pred_full) / cfg.c_fid)
        for j in range(nfull, len(Q)):
            fid[j] = 1.0 - min(1.0, err(full_mean, predict(Q[j])) / cfg.c_fid)

    total = cfg.alpha * acc + cfg.beta * coh + cfg.gamma * fid
    total = np.where(valid, total, 0.0)
    return GroupRewards(total=total, coh=np.where(valid, coh, 0.0), acc=acc, fid=fid, I=I, S=S, valid=valid,
                        terms={"M": M, "gate": gate, "gate_S": gate_S,
                               "excess_full": np.concatenate(excess[:2]) if excess else np.zeros(0)})


def batch_rewards(groups: list[PatientGroup], cfg: RewardConfig, K: int) -> tuple[list[GroupRewards], float]:
    """Two passes: per-group terms, then the batch-level sensitivity gate. Each group uses its own
    candidate count (g.K, else K); for M the patient means are zero-padded to the largest count,
    which leaves every entropy unchanged (audit 27 Nov 2026: 29 of the 2,000 real training cases
    have 9 candidates, and a single shared K crashed or mis-scored those batches)."""
    Ks = [int(g.K or K) for g in groups]
    means = [patient_mean(g, k) for g, k in zip(groups, Ks)]
    kmax = max(Ks) if Ks else K
    M = cross_patient_information([np.pad(m, (0, kmax - len(m))) for m in means])
    first = [group_rewards(g, M, cfg, k, gate_S=1.0) for g, k in zip(groups, Ks)]
    if cfg.sensitivity != "batch":
        return first, M
    # mean floor-corrected movement of full-evidence rollouts away from the contrast
    exc = np.concatenate([gr.terms["excess_full"] for gr in first]) if first else np.zeros(0)
    raw = float(np.mean(exc)) / cfg.delta_S if len(exc) and cfg.delta_S > 0 else 0.0
    gate_S = float(np.sqrt(min(1.0, raw))) if cfg.sqrt_S else min(1.0, raw)
    return [group_rewards(g, M, cfg, k, gate_S=gate_S) for g, k in zip(groups, Ks)], M


# ----------------------------------------------------------------------------- team credit
# Amendment 4 (27 Nov 2026; docs/amendment_4.md, docs/theory.md Propositions 6-7).
#
# The registered objective of one step is the batch total J(y) = sum_j R_j(y). Several of its terms
# couple rollouts: the gates M and G_S are batch statistics, invariance compares a rollout with the
# OTHER rendering's rollouts, and fidelity compares full-evidence with reduced-evidence rollouts.
# Feeding R_j itself into per-rendering centred advantages treats every coupling as a constant: the
# update then follows the gradient of sum_j R_j with the gates (and partners) held fixed, which the
# first decisive runs showed is maximised by flattening every answer (docs/training_runs.md).
#
# Team credit gives rollout i its exact marginal contribution
#     D_i = J(y) - J(y without i),
# where J(y without i) recomputes every term of the batch with rollout i removed. J(y without i) does
# not depend on rollout i, so sum_i D_i * grad log pi(y_i) is an unbiased estimator of grad E[J]: the
# registered objective, whose optimum Proposition 4 protects, is what training now follows. For a
# reward without couplings (RLVR, RLCR: beta = gamma = 0) D_i = R_i exactly. The construction is the
# difference reward of Wolpert & Tumer (2001), applied within one prompt's group by Team-GRPO
# (Gao et al., arXiv:2611.02404); here it spans the renderings of a patient and the whole batch.

def _m_gate(M: float, cfg: RewardConfig) -> float:
    return (min(1.0, M / cfg.m0) if cfg.m0 > 0 else 1.0) if cfg.use_m_gate else 1.0


def _s_gate(exc_sum: float, exc_n: int, cfg: RewardConfig) -> float:
    """The batch sensitivity gate exactly as batch_rewards computes it (1 when it is not used)."""
    if not (cfg.use_sensitivity and cfg.sensitivity == "batch"):
        return 1.0
    raw = (exc_sum / exc_n) / cfg.delta_S if exc_n and cfg.delta_S > 0 else 0.0
    return float(np.sqrt(min(1.0, raw))) if cfg.sqrt_S else min(1.0, raw)


def _group_parts(gr: GroupRewards, cfg: RewardConfig) -> tuple[float, float, float, int]:
    """A group's total split around the batch gates: sum(total) = A + C * gate_S * gate(M).
    Returns (A, C, sum of full-evidence excess, number of full-evidence rollouts)."""
    v = gr.valid
    A = float(np.sum(np.where(v, cfg.alpha * gr.acc + cfg.gamma * gr.fid, 0.0)))
    Iu = gr.I if cfg.use_invariance else np.ones_like(gr.I)
    if cfg.use_sensitivity and cfg.sensitivity != "batch":
        Iu = Iu * (np.sqrt(gr.S) if cfg.sqrt_S else gr.S)
    C = float(cfg.beta * np.sum(np.where(v, Iu, 0.0)))
    ex = gr.terms["excess_full"]
    return A, C, float(np.sum(ex)), int(len(ex))


def drop_rollout(g: PatientGroup, i: int) -> PatientGroup:
    """The group without its i-th rollout (rollouts ordered x, x', x-)."""
    nx, nxp = len(g.q_x), len(g.q_xp)
    q_x, q_xp, q_xm = list(g.q_x), list(g.q_xp), list(g.q_xm)
    if i < nx:
        q_x.pop(i)
    elif i < nx + nxp:
        q_xp.pop(i - nx)
    else:
        q_xm.pop(i - nx - nxp)
    return PatientGroup(q_x, q_xp, q_xm, gold=g.gold, added_loglik=g.added_loglik, K=g.K)


def team_credit(groups: list[PatientGroup], cfg: RewardConfig, K: int
                ) -> tuple[list[np.ndarray], list[GroupRewards], float]:
    """Per-rollout marginal contributions D_i = J(y) - J(y without i) to the batch objective.

    Exact: removing a rollout changes its own group's terms and, through M and the sensitivity
    gate, every group's coherence weight; other groups' acc, fid and I do not depend on it, so
    each removal recomputes one group plus two batch scalars (checked against a full recomputation
    in tests/test_team_credit.py). Returns (credits per group, the registered rewards, M)."""
    res, M = batch_rewards(groups, cfg, K)
    Ks = [int(g.K or K) for g in groups]
    kmax = max(Ks) if Ks else K
    means = [np.pad(patient_mean(g, k), (0, kmax - k)) for g, k in zip(groups, Ks)]
    parts = [_group_parts(gr, cfg) for gr in res]
    A_tot = sum(p[0] for p in parts)
    C_tot = sum(p[1] for p in parts)
    E_tot = sum(p[2] for p in parts)
    N_tot = sum(p[3] for p in parts)
    J = A_tot + C_tot * _s_gate(E_tot, N_tot, cfg) * _m_gate(M, cfg)
    credits: list[np.ndarray] = []
    for gi, (g, k) in enumerate(zip(groups, Ks)):
        A_g, C_g, E_g, N_g = parts[gi]
        n = len(g.q_x) + len(g.q_xp) + len(g.q_xm)
        d = np.zeros(n)
        for i in range(n):
            gd = drop_rollout(g, i)
            A_d, C_d, E_d, N_d = _group_parts(group_rewards(gd, cfg.m0, cfg, k, gate_S=1.0), cfg)
            means_d = list(means)
            means_d[gi] = np.pad(patient_mean(gd, k), (0, kmax - k))
            M_d = cross_patient_information(means_d)
            gates_d = _s_gate(E_tot - E_g + E_d, N_tot - N_g + N_d, cfg) * _m_gate(M_d, cfg)
            d[i] = J - (A_tot - A_g + A_d + (C_tot - C_g + C_d) * gates_d)
        credits.append(d)
    return credits, res, M


def team_credit_bruteforce(groups: list[PatientGroup], cfg: RewardConfig, K: int) -> list[np.ndarray]:
    """Reference implementation (tests only): recompute the whole batch once per removed rollout."""
    J = sum(float(gr.total.sum()) for gr in batch_rewards(groups, cfg, K)[0])
    out = []
    for gi, g in enumerate(groups):
        n = len(g.q_x) + len(g.q_xp) + len(g.q_xm)
        d = np.zeros(n)
        for i in range(n):
            alt = list(groups)
            alt[gi] = drop_rollout(g, i)
            d[i] = J - sum(float(gr.total.sum()) for gr in batch_rewards(alt, cfg, K)[0])
        out.append(d)
    return out


def advantages(total: np.ndarray, normalize_std: bool = True, min_std: float = 1e-6,
               sizes: tuple[int, ...] | None = None) -> np.ndarray | None:
    """Group-relative advantages for one paired group. Returns None when the group carries no
    signal (DAPO-style dynamic sampling: skip it).

    `sizes` = (G, G, H), the rollouts on x, x' and x- in order: each rendering is centred on ITS
    OWN mean (a per-prompt baseline, as in GRPO), and all are scaled by one pooled within-rendering
    SD. Centring on the whole group's mean (sizes=None, the original form) gives the reduced-
    evidence x- rollouts a systematically negative advantage (they are less accurate by
    construction): unbiased in expectation but high-variance, and with H = 2 it pushes the policy
    away from whatever it answers on reduced evidence (audit 27 Nov 2026)."""
    total = np.asarray(total, dtype=float)
    if sizes is None:
        sd = float(total.std())
        if sd < min_std:
            return None
        a = total - total.mean()
        return a / sd if normalize_std else a
    assert sum(sizes) == len(total), (sizes, len(total))
    a = np.empty_like(total)
    k = 0
    for n in sizes:
        seg = total[k:k + n]
        a[k:k + n] = seg - seg.mean() if n else seg
        k += n
    sd = float(np.sqrt(np.mean(a ** 2)))
    if sd < min_std:
        return None
    return a / sd if normalize_std else a
