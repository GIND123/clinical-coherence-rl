"""Synthetic policies for the reward unit tests (plan §3.5; Propositions 2 and 4).

Each policy maps (patient tokens grouped in units, rendering order, ablated
unit) -> posterior over the 49 DDXPlus pathologies. The KB-Bayes oracle uses
the corpus likelihoods; the others are the degenerate or distorted policies a
coherence reward must not prefer. No language model is involved: this tests
the reward, so it must run in seconds on a CPU before any GPU job.

    python -m ccrl.reward.synthetic      prints the table; exit 1 if a criterion fails
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from ccrl.data import ddxplus as dx
from ccrl.reward.coherence import PatientGroup, RewardConfig, batch_rewards, cross_patient_information


def _h(*a) -> int:
    return int(hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()[:12], 16)


class Policy:
    name = "policy"
    stochastic = False

    def __call__(self, units: list[list[str]], order_tag: str, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class Oracle(Policy):
    name = "oracle"

    def __init__(self, lik: dx.Likelihood):
        self.lik = lik

    def __call__(self, units, order_tag, rng):
        return self.lik.posterior([t for u in units for t in u])


class ConstantUniform(Policy):
    name = "constant_uniform"

    def __init__(self, K: int):
        self.K = K

    def __call__(self, units, order_tag, rng):
        return np.full(self.K, 1.0 / self.K)


class ConstantPrior(Policy):
    name = "constant_prior"

    def __init__(self, lik: dx.Likelihood):
        self.p = np.exp(lik.log_prior)

    def __call__(self, units, order_tag, rng):
        return self.p


class TwoCluster(Policy):
    """Invariant and deterministic, but uses one bit of the patient (fever present?)."""
    name = "two_cluster"

    def __init__(self, lik: dx.Likelihood, token: str = "E_91"):
        self.a = lik.posterior([token])
        self.b = np.exp(lik.log_prior)
        self.token = token

    def __call__(self, units, order_tag, rng):
        return self.a if any(self.token in u for u in units) else self.b


class OneFinding(Policy):
    """Bayes on the chief-complaint unit only (ignores every other finding)."""
    name = "one_finding"

    def __init__(self, lik: dx.Likelihood):
        self.lik = lik

    def __call__(self, units, order_tag, rng):
        return self.lik.posterior(units[0]) if units else np.exp(self.lik.log_prior)


class PermutedOracle(Policy):
    """Oracle composed with a fixed relabelling sigma (Proposition 2)."""
    name = "oracle_permuted"

    def __init__(self, lik: dx.Likelihood, seed: int = 0):
        self.o = Oracle(lik)
        self.sigma = np.random.default_rng(seed).permutation(lik.D)

    def __call__(self, units, order_tag, rng):
        return self.o(units, order_tag, rng)[self.sigma]


class TemperedOracle(Policy):
    name = "oracle_tempered"

    def __init__(self, lik: dx.Likelihood, t: float = 0.5):
        self.o = Oracle(lik)
        self.t = t

    def __call__(self, units, order_tag, rng):
        p = self.o(units, order_tag, rng) ** self.t
        return p / p.sum()


class OrderNoiseOracle(Policy):
    """Oracle plus a deterministic perturbation that depends on the presentation order."""
    name = "oracle_order_noise"

    def __init__(self, lik: dx.Likelihood, sd: float = 1.5):
        self.o = Oracle(lik)
        self.sd = sd

    def __call__(self, units, order_tag, rng):
        lp = np.log(np.clip(self.o(units, order_tag, rng), 1e-12, 1))
        lp = lp + np.random.default_rng(_h("order", order_tag)).normal(0, self.sd, len(lp))
        p = np.exp(lp - lp.max())
        return p / p.sum()


class Noise(Policy):
    """Pure sampling noise: a fresh random posterior on every call."""
    name = "noise"
    stochastic = True

    def __init__(self, K: int):
        self.K = K

    def __call__(self, units, order_tag, rng):
        return rng.dirichlet(np.full(self.K, 0.3))


def evaluate(policy: Policy, cases: list[dict], lik: dx.Likelihood, cfg: RewardConfig, G: int = 4,
             batch: int = 16, seed: int = 0) -> dict:
    """Mean reward terms of a policy over the cases, in batches of `batch` patients."""
    rng = np.random.default_rng(seed)
    K = lik.D
    tot = {"coh": [], "acc": [], "fid": [], "total": [], "M": []}
    for b0 in range(0, len(cases) - batch + 1, batch):
        groups = []
        for c in cases[b0:b0 + batch]:
            units = [u["tokens"] for u in c["units"]]
            cc, rest = units[0], units[1:]
            r = np.random.default_rng(_h(seed, c["case_id"]))
            o1 = [cc] + [rest[i] for i in r.permutation(len(rest))]
            o2 = [cc] + [rest[i] for i in r.permutation(len(rest))]
            om = [cc]                                     # triage contrast: chief complaint only
            tag = lambda o: "|".join(",".join(u) for u in o)  # noqa: E731
            q_x = [policy(o1, tag(o1), rng) for _ in range(G)]
            q_p = [policy(o2, tag(o2), rng) for _ in range(G)]
            q_m = [policy(om, tag(om), rng) for _ in range(max(1, G // 2))]
            gi = lik.pathologies.index(c["pathology"])
            added = lik.tau * lik.unit_loglik([t for u in rest for t in u])
            groups.append(PatientGroup(q_x, q_p, q_m, gold=gi, added_loglik=added))
        res, M = batch_rewards(groups, cfg, K)
        tot["M"].append(M)
        for gr in res:
            tot["coh"].extend(gr.coh.tolist())
            tot["acc"].extend(gr.acc.tolist())
            tot["fid"].extend(gr.fid.tolist())
            tot["total"].extend(gr.total.tolist())
    return {k: float(np.mean(v)) for k, v in tot.items()}


def calibrate(cases: list[dict], lik: dx.Likelihood, batch: int = 16) -> dict:
    """delta_S = median reference JS(full, triage); m0 = 0.5 x reference M (plan §3.3)."""
    from ccrl.analysis.metrics import js
    o = Oracle(lik)
    ab, Ms = [], []
    for b0 in range(0, len(cases) - batch + 1, batch):
        means = []
        for c in cases[b0:b0 + batch]:
            units = [u["tokens"] for u in c["units"]]
            full = o(units, "", None)
            ab.append(js(full, o([units[0]], "", None)))
            means.append(full)
        Ms.append(cross_patient_information(means))
    # Same convention as m0: half of the reference's batch-mean value.
    return {"tau": lik.tau, "delta_S": 0.5 * float(np.mean(ab)), "oracle_mean_contrast_js": float(np.mean(ab)),
            "oracle_median_contrast_js": float(np.median(ab)), "oracle_M": float(np.mean(Ms)),
            "m0": 0.5 * float(np.mean(Ms))}


def run_suite(n_cases: int = 1024, verbose: bool = True, sensitivity: str = "batch") -> tuple[bool, dict]:
    from ccrl import paths
    from ccrl.util import read_jsonl
    lik = dx.load_likelihood()
    cases = [c for c in read_jsonl(paths.FROZEN_DDX / "cases.jsonl") if c["role"] == "distill"][:n_cases]
    cal = calibrate(cases, lik)
    cfg = RewardConfig(delta_S=cal["delta_S"], m0=cal["m0"], tau_I=0.10, sensitivity=sensitivity)
    K = lik.D
    pols = [ConstantUniform(K), ConstantPrior(lik), TwoCluster(lik), OneFinding(lik), PermutedOracle(lik),
            TemperedOracle(lik), Oracle(lik), OrderNoiseOracle(lik), Noise(K)]
    res = {p.name: evaluate(p, cases, lik, cfg) for p in pols}
    o = res["oracle"]
    checks = {
        "R_coh(constant_uniform) == 0": res["constant_uniform"]["coh"] == 0.0,
        "R_coh(constant_prior) == 0": res["constant_prior"]["coh"] == 0.0,
        "R_coh(two_cluster) < R_coh(oracle)": res["two_cluster"]["coh"] < o["coh"],
        "R_coh(one_finding) < R_coh(oracle)": res["one_finding"]["coh"] < o["coh"],
        "R_coh(oracle_permuted) == R_coh(oracle)  [Prop. 2]": abs(res["oracle_permuted"]["coh"] - o["coh"]) < 1e-9,
        "R_coh(noise) < 0.1 * R_coh(oracle)  [Correction 1]": res["noise"]["coh"] < 0.1 * o["coh"],
        "R_fid(oracle) > 0.95 and > every other policy": o["fid"] > 0.95 and all(
            o["fid"] > v["fid"] for k, v in res.items() if k != "oracle"),
        "R_total(oracle) > R_total(oracle_permuted)": o["total"] > res["oracle_permuted"]["total"],
        "R_total(oracle) > R_total(oracle_tempered)": o["total"] > res["oracle_tempered"]["total"],
        "R_total(oracle) > R_total(oracle_order_noise)": o["total"] > res["oracle_order_noise"]["total"],
    }
    ok = all(checks.values())
    if verbose:
        print(f"calibration: {json.dumps(cal)}")
        print(f"{'policy':<22}{'R_coh':>9}{'R_fid':>9}{'R_acc':>9}{'R_total':>9}{'M':>8}")
        for k, v in res.items():
            print(f"{k:<22}{v['coh']:>9.4f}{v['fid']:>9.4f}{v['acc']:>9.4f}{v['total']:>9.4f}{v['M']:>8.3f}")
        for k, v in checks.items():
            print(f"[{'PASS' if v else 'FAIL'}] {k}")
    return ok, {"calibration": cal, "results": res, "checks": checks}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "batch"
    ok, _ = run_suite(sensitivity=mode)
    sys.exit(0 if ok else 1)
