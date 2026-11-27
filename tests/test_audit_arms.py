"""plan_final.md P6: the readout channel of the estimator audit."""
import numpy as np

from ccrl.reward import audit_arms as A


def test_initial_compliance_is_30_percent():
    K = 49
    th = A.theta0(K)
    rng = np.random.default_rng(0)
    f = np.exp(th[3] + A.NU[3] * rng.standard_normal(200000))
    assert abs(np.mean(K * f <= A.TOL) - 0.30) < 0.01


def test_lenient_renormalises_filler_and_strict_rejects_it():
    K = 10
    rng = np.random.default_rng(1)
    base = np.log(np.linspace(1, 10, K))
    th = np.array([0.0, -5.0, -5.0, np.log(0.05)])          # heavy filler, almost no noise
    rows, _xi, comply = A._answers(base, 50, K, th, rng, "lenient")
    Q = np.array(rows)
    plain = np.exp(base) / np.exp(base).sum()
    assert np.allclose(Q.sum(1), 1) and Q.min(1).mean() > 1.5 * plain.min()     # filler flattens
    rows_s, _xi, comply_s = A._answers(base, 50, K, th, np.random.default_rng(1), "strict")
    assert all(r is None for r, ok in zip(rows_s, comply_s) if not ok) and not comply_s.all()


def test_safeguard_detects_drop():
    flat = [{"M": 0.5}] * 40
    drop = [{"M": 0.5}] * 10 + [{"M": 0.1}] * 30
    assert A.safeguard_fires(flat) is None
    assert A.safeguard_fires(drop) is not None
