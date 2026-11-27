"""Fixes from the 27 Nov 2026 audit (docs/audit_2026-11-27.md)."""
import numpy as np

from ccrl.reward.coherence import PatientGroup, RewardConfig, advantages, batch_rewards, cross_patient_information


def _group(K, rng, gold=0):
    d = lambda: list(rng.dirichlet(np.ones(K)))
    return PatientGroup([d() for _ in range(4)], [d() for _ in range(4)], [d() for _ in range(2)], gold=gold, K=K)


def test_batch_with_mixed_candidate_counts():
    rng = np.random.default_rng(0)
    groups = [_group(10, rng) for _ in range(5)] + [_group(9, rng)]
    res, M = batch_rewards(groups, RewardConfig(), K=10)          # previously: crash
    assert len(res) == 6 and all(len(r.total) == 10 for r in res) and 0 <= M <= 1


def test_zero_padding_leaves_m_unchanged():
    rng = np.random.default_rng(1)
    means = [rng.dirichlet(np.ones(9)) for _ in range(6)]
    padded = [np.pad(m, (0, 1)) for m in means]
    assert abs(cross_patient_information(means) - cross_patient_information(padded)) < 1e-12


def test_per_rendering_baseline_removes_the_contrast_offset():
    total = np.array([1.0, 1.2, 0.8, 1.0, 0.9, 1.1, 1.0, 1.0, 0.1, 0.3])   # x-, last two, much lower
    old = advantages(total)
    new = advantages(total, sizes=(4, 4, 2))
    assert old[-2:].max() < 0                                       # whole-group centring: always negative
    for seg in (slice(0, 4), slice(4, 8), slice(8, 10)):
        assert abs(new[seg].mean()) < 1e-12                         # each rendering centred on itself
    assert advantages(np.ones(10), sizes=(4, 4, 2)) is None


def test_contrast_invariance_uses_leave_one_out():
    from ccrl.reward.coherence import group_rewards
    K = 5
    a, b = [0.9, 0.025, 0.025, 0.025, 0.025], [0.025, 0.9, 0.025, 0.025, 0.025]
    g = PatientGroup([a] * 4, [a] * 4, [a, b], gold=0, K=K)
    r = group_rewards(g, M=1.0, cfg=RewardConfig(tau_I=1.0), K=K, gate_S=1.0)
    from ccrl.analysis.metrics import js
    assert abs(r.I[-1] - (1 - js(b, a))) < 1e-9                    # partner is the OTHER contrast rollout
