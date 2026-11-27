"""Amendment 4: team credit and the estimator audit (docs/amendment_4.md, docs/theory.md Props. 6-7)."""
import numpy as np
import pytest

from ccrl.reward.coherence import (PatientGroup, RewardConfig, batch_rewards, team_credit, team_credit_bruteforce)

K = 12


def _rand_group(rng, k=K, p_invalid=0.1, sharp=0.3):
    def d():
        if rng.random() < p_invalid:
            return None
        return list(rng.dirichlet(np.ones(k) * sharp))
    return PatientGroup([d() for _ in range(4)], [d() for _ in range(4)], [d() for _ in range(2)],
                        gold=int(rng.integers(k)), added_loglik=rng.normal(0, 2, k), K=k)


@pytest.mark.parametrize("mode", ["batch", "rollout"])
@pytest.mark.parametrize("m0", [0.2, 0.9])
def test_fast_team_credit_equals_full_recomputation(mode, m0):
    rng = np.random.default_rng(3)
    groups = [_rand_group(rng) for _ in range(7)] + [_rand_group(rng, k=K - 1)]
    cfg = RewardConfig(tau_I=0.788, delta_S=0.234, m0=m0, sensitivity=mode)
    fast, _, _ = team_credit(groups, cfg, K)
    slow = team_credit_bruteforce(groups, cfg, K)
    assert max(np.abs(a - b).max() for a, b in zip(fast, slow)) < 1e-10


def test_team_credit_is_the_reward_when_nothing_couples():
    rng = np.random.default_rng(4)
    groups = [_rand_group(rng) for _ in range(6)]
    for acc_mode in ("binary", "rlcr", "brier"):
        cfg = RewardConfig(alpha=1.0, beta=0.0, gamma=0.0, acc_mode=acc_mode)
        D, res, _ = team_credit(groups, cfg, K)
        assert max(np.abs(d - r.total).max() for d, r in zip(D, res)) < 1e-12


def _onehot(i, k=K, eps=0.0):
    v = np.full(k, eps)
    v[i] = 1.0
    return list(v / v.sum())


def test_hedging_pays_its_own_reward_but_is_charged_for_the_gate():
    """The failure of the first decisive runs in miniature. Each patient answers sharply, but the
    two renderings disagree (a presentation effect). A rollout that flattens its answer moves
    closer to the other rendering, so its own reward rises; it also lowers the batch's M. Team
    credit charges that externality to it (negative) and credits the sharp rollouts (positive).
    Whether the charge outweighs the own gain depends on how much coherence the batch has already
    earned (the gate multiplies sum I); the population-level outcome is the estimator audit's job."""
    groups = []
    for p in range(8):
        a, b = p % K, (p + 1) % K
        x = [_onehot(a, eps=0.02) for _ in range(4)]
        if p == 0:
            x[0] = list(np.full(K, 1.0 / K))                       # the hedging rollout
        groups.append(PatientGroup(x, [_onehot(b, eps=0.02) for _ in range(4)],
                                   [_onehot(a, eps=0.2) for _ in range(2)], gold=a, K=K))
    cfg = RewardConfig(alpha=0.0, beta=1.0, gamma=0.0, tau_I=0.788, delta_S=0.05, m0=0.95)
    D, res, M = team_credit(groups, cfg, K)
    assert M < cfg.m0                                                # the M gate is active
    r0 = res[0]
    assert r0.total[0] > r0.total[1:4].max()                        # registered credit: hedging pays
    ext = D[0] - r0.total                                            # what team credit adds
    assert ext[0] < 0 < ext[1:4].min()


def test_estimator_audit_update_direction():
    """With M below m0, the registered estimator pushes the temperature up (hedging) and team
    credit pushes it down; both reduce presentation sensitivity."""
    from ccrl.reward import audit as A
    pats, k = A.load_patients(n=256, omega=0.25, zeta=1.0, p_wrong=0.6)
    th = np.array([0.0, 0.0, -0.3])
    g_reg, st = A.gradient(pats, k, th, A.registered_cfg(), "grpo", np.random.default_rng(0), n_batches=6)
    g_team, _ = A.gradient(pats, k, th, A.registered_cfg(), "team", np.random.default_rng(0), n_batches=6)
    assert st["M"] < A.registered_cfg().m0
    assert g_reg[0] > 0 > g_team[0]
    assert g_reg[1] < 0 and g_team[1] < 0


def test_strict_readout_rule():
    from ccrl.prompts.parse import parse_posterior
    from ccrl.train.grpo import readout_valid
    codes = ["AA", "AB", "AC"]
    ok = parse_posterior('{"dx": "AA", "p": {"AA": 70, "AB": 20, "AC": 10}}', codes)
    filler = parse_posterior('{"dx": "AA", "p": {"AA": 50, "AB": 5, "AC": 5}}', codes)
    slip = parse_posterior('{"dx": "AA", "p": {"AA": 70, "AB": 21, "AC": 10}}', codes)
    assert readout_valid(ok, True, 2.0) and readout_valid(slip, True, 2.0)
    assert not readout_valid(filler, True, 2.0) and readout_valid(filler, False, 2.0)
    assert not readout_valid(parse_posterior("no json", codes), False, 2.0)
