import numpy as np

from ccrl.reward.coherence import PatientGroup, RewardConfig, batch_rewards, cross_patient_information
from ccrl.reward.synthetic import run_suite


def test_synthetic_suite_passes():
    ok, out = run_suite(n_cases=512, verbose=False)
    assert ok, out["checks"]


def test_constant_policy_has_zero_information_and_coherence():
    q = np.full(5, 0.2)
    groups = [PatientGroup([q] * 4, [q] * 4, [q] * 2, gold=0) for _ in range(8)]
    res, M = batch_rewards(groups, RewardConfig(), 5)
    assert M == 0.0 and all((g.coh == 0).all() for g in res)


def test_information_bound_k_clusters():
    a, b = np.eye(4)[0] * 0.96 + 0.01, np.eye(4)[1] * 0.96 + 0.01
    M = cross_patient_information([a, b] * 8)                 # 2 clusters, B = 16
    assert M <= np.log(2) / np.log(16) + 1e-12


def test_invalid_rollouts_get_zero():
    q = np.array([0.7, 0.2, 0.1])
    g = PatientGroup([q, None, q, q], [q] * 4, [np.array([0.3, 0.3, 0.4])] * 2, gold=0)
    res, _ = batch_rewards([g, g], RewardConfig(), 3)
    assert res[0].total[1] == 0.0 and not res[0].valid[1]
