"""Divergences and scores over posteriors. Shared by the panel analysis and
the Track B reward, so both tracks are scored by identical code.

JS divergence is base 2 (in [0, 1]); entropies are in nats unless noted.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def norm(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
    s = p.sum(axis=-1, keepdims=True)
    k = p.shape[-1]
    return np.where(s > 0, p / np.where(s > 0, s, 1.0), 1.0 / k)


def js(p, q) -> float:
    p, q = norm(p), norm(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > EPS
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], EPS))))
    return float(min(1.0, max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


def js_rows(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Row-wise JS (base 2) between two equally shaped matrices."""
    P, Q = norm(P), norm(Q)
    M = 0.5 * (P + Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > EPS, P * np.log2(P / np.maximum(M, EPS)), 0.0).sum(-1)
        b = np.where(Q > EPS, Q * np.log2(Q / np.maximum(M, EPS)), 0.0).sum(-1)
    return np.clip(0.5 * (a + b), 0.0, 1.0)


def pairwise_js(P: np.ndarray) -> np.ndarray:
    n = len(P)
    out = np.zeros((n, n))
    for i in range(n):
        out[i] = js_rows(np.repeat(P[i:i + 1], n, 0), P)
    return out


def entropy(p, base: float = np.e) -> float:
    p = norm(p)
    m = p > EPS
    return float(-np.sum(p[m] * np.log(p[m])) / np.log(base))


def logit(p, floor: float = 0.005) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), floor, 1 - floor)
    return np.log(p / (1 - p))


def brier(p, gold_index: int) -> float:
    p = norm(p)
    e = np.zeros_like(p)
    e[gold_index] = 1.0
    return float(np.sum((p - e) ** 2))


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def permutation_ceiling(p, n_perm: int = 20, seed: int = 0) -> float:
    """E_sigma JS(q, sigma q): the divergence between a belief and the same-
    shaped belief over relabelled hypotheses. A ceiling for case-specific
    candidate lists, where between-patient JS is undefined."""
    p = norm(p)
    rng = np.random.default_rng(seed)
    perms = np.stack([p[rng.permutation(len(p))] for _ in range(n_perm)])
    return float(js_rows(np.repeat(p[None], n_perm, 0), perms).mean())


def mutual_information(P: np.ndarray) -> float:
    """I(patient; output) for a uniform mixture over rows: H(mean) - mean H (nats)."""
    P = norm(P)
    return float(entropy(P.mean(0)) - np.mean([entropy(r) for r in P]))
