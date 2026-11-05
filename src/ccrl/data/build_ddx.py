"""Build the frozen DDXPlus artifacts from the raw release (run once).

    ccrl data build-ddxplus        (needs the raw corpus under $CCRL_DATA/ddxplus)

Writes, under artifacts/frozen/ddxplus/:
    kb.json            compact knowledge base + two-letter candidate codes
    likelihood.json    P(token | pathology) counts from the TRAIN split
    heldout.json       the 8 pathologies held out of Track B training
    cases.jsonl        every case the study will ever show a model:
                       pilot (20), panel (300; flags core/sequential),
                       panel_ext (300, priority C2), distill (2,000, train split)
    build_report.json  selection counts, KB-Bayes fit to the shipped differentials

Everything is a deterministic function of the raw files (md5-checked) and the
master seed in config/study.yaml. The panel machine needs only these outputs.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ccrl import config, paths
from ccrl.data import ddxplus as dx
from ccrl.util import rng, write_json, write_jsonl


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_raw(raw_dir: Path) -> None:
    files = config.study()["ddxplus"]["files"]
    for name, (_fid, md5) in files.items():
        p = raw_dir / name
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run `ccrl data download ddxplus`")
        got = md5_file(p)
        if got != md5:
            raise ValueError(f"md5 mismatch for {p}: {got} != {md5}")


def draw_heldout(seed: int, systems: dict, per_system: dict, pathologies: list[str]) -> list[str]:
    all_named = sorted(n for v in systems.values() for n in v)
    if sorted(all_named) != sorted(pathologies):
        missing = set(pathologies) - set(all_named)
        extra = set(all_named) - set(pathologies)
        raise ValueError(f"body_systems must partition the 49 pathologies; missing={missing} extra={extra}")
    out = []
    for system in sorted(systems):
        k = int(per_system.get(system, 0))
        names = sorted(systems[system])
        if k:
            idx = rng(seed, "heldout", system).choice(len(names), size=k, replace=False)
            out.extend(names[i] for i in sorted(idx))
    return sorted(out)


def _eligible(kb: dx.KB, r: dict, cfg: dict) -> tuple[bool, list[dx.Unit]]:
    if len(r["evidences"]) > cfg["max_tokens_per_case"]:
        return False, []
    units = dx.build_units(kb, r["evidences"], r["initial_evidence"])
    n = len(units)
    if not (cfg["min_units"] <= n <= cfg["max_units"]):
        return False, units
    if not any(u.is_cc for u in units):
        return False, units
    # No per-block minimum: with >= 7 units and only the chief complaint fixed,
    # the two exchangeable blocks hold >= 6 units, so one has >= 3 and every
    # case admits distinct permutations. (A "2 symptom units" rule would drop
    # Larygospasm entirely: its non-CC findings are all antecedents.)
    return True, units


def case_record(kb: dx.KB, r: dict, units: list[dx.Unit], role: str) -> dict:
    return {
        "case_id": f"ddx-{r['split']}-{r['row_id']:07d}",
        "dataset": "ddxplus",
        "split": r["split"],
        "row_id": r["row_id"],
        "role": role,
        "flags": {},
        "age": r["age"],
        "sex": r["sex"],
        "pathology": r["pathology"],
        "gold": kb.name_to_code[r["pathology"]],
        "severity": kb.conditions[r["pathology"]]["severity"],
        "initial_evidence": r["initial_evidence"],
        "tokens": list(r["evidences"]),
        "units": [u.to_json() for u in units],
        "differential": [[n, round(p, 6)] for n, p in r["differential"]],
    }


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return max(0.0, min(1.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def build(raw_dir: Path | None = None, out_dir: Path | None = None, verbose: bool = True) -> dict:
    raw_dir = Path(raw_dir or paths.DATA_DDX)
    out_dir = Path(out_dir or paths.FROZEN_DDX)
    st = config.study()
    cfg = st["ddxplus"]
    seed = int(st["seed"])
    verify_raw(raw_dir)

    def log(*a):
        if verbose:
            print("[build-ddxplus]", *a, flush=True)

    kb = dx.kb_from_raw(raw_dir / "release_conditions.json", raw_dir / "release_evidences.json")
    write_json(out_dir / "kb.json", dx.kb_to_json(kb))
    log(f"KB: {kb.K} pathologies, {len(kb.evidences)} evidences")

    heldout = draw_heldout(seed, cfg["body_systems"], cfg["heldout_per_system"], kb.pathologies)
    write_json(out_dir / "heldout.json", {"heldout": heldout,
                                          "rule": "seeded draw per body system (study.yaml)"})
    log("held-out pathologies:", heldout)

    # ---- pass 1 over TRAIN: likelihood counts + eligible rows for distillation
    log("pass 1/2 over train split (likelihoods) ...")
    pidx = {p: i for i, p in enumerate(kb.pathologies)}
    N = np.zeros(kb.K)
    counts: dict[str, np.ndarray] = {}
    distill_pool: dict[str, list[int]] = defaultdict(list)
    held = set(heldout)
    for r in dx.iter_split("train", raw_dir):
        d = pidx[r["pathology"]]
        N[d] += 1
        for t in set(r["evidences"]):
            a = counts.get(t)
            if a is None:
                a = counts[t] = np.zeros(kb.K)
            a[d] += 1
        if r["pathology"] not in held:
            ok, _ = _eligible(kb, r, cfg)
            if ok:
                distill_pool[r["pathology"]].append(r["row_id"])
    toks = sorted(counts, key=lambda t: (int(dx.split_token(t)[0].split("_")[1]), t))
    lik = dx.Likelihood(pathologies=kb.pathologies, N=N, tokens=toks,
                        counts=np.stack([counts[t] for t in toks]), alpha=float(cfg["likelihood_smoothing"]))
    write_json(out_dir / "likelihood.json", lik.to_json(), indent=None)
    log(f"likelihood: {len(toks)} tokens over {int(N.sum())} training patients")

    # ---- distillation selection (train split, non-held-out pathologies)
    train_paths = sorted(p for p in kb.pathologies if p not in held)
    per = math.ceil(cfg["distill_cases"] / len(train_paths))
    chosen_train: list[int] = []
    for p in train_paths:
        pool = sorted(distill_pool[p])
        k = min(per, len(pool))
        idx = rng(seed, "distill", p).choice(len(pool), size=k, replace=False)
        chosen_train.extend(pool[i] for i in sorted(idx))
    order = rng(seed, "distill", "trim").permutation(len(chosen_train))
    chosen_train = sorted(chosen_train[i] for i in order[: cfg["distill_cases"]])
    chosen_train_set = set(chosen_train)

    # ---- pass over TEST: eligible pool per pathology
    log("scanning test split ...")
    test_rows: dict[int, dict] = {}
    pool_by_p: dict[str, list[int]] = defaultdict(list)
    for r in dx.iter_split("test", raw_dir):
        ok, units = _eligible(kb, r, cfg)
        if ok:
            pool_by_p[r["pathology"]].append(r["row_id"])
            test_rows[r["row_id"]] = r
    used: set[int] = set()
    panel: list[int] = []
    ext: list[int] = []
    shortfall = {}
    for p in kb.pathologies:
        pool = sorted(pool_by_p[p])
        perm = rng(seed, "panel", p).permutation(len(pool))
        ordered = [pool[i] for i in perm]
        a = ordered[: cfg["panel_per_pathology"]]
        b = ordered[cfg["panel_per_pathology"]: cfg["panel_per_pathology"] + cfg["ext_per_pathology"]]
        if len(a) + len(b) < cfg["panel_per_pathology"] + cfg["ext_per_pathology"]:
            shortfall[p] = len(pool)
        panel += a
        ext += b
        used.update(a)
        used.update(b)
    rest = sorted(i for p in kb.pathologies for i in pool_by_p[p] if i not in used)
    perm = rng(seed, "panel", "extra").permutation(len(rest))
    rest = [rest[i] for i in perm]
    panel += rest[: cfg["panel_extra_random"]]
    ext += rest[cfg["panel_extra_random"]: cfg["panel_extra_random"] + cfg["ext_extra_random"]]
    pilot_start = cfg["panel_extra_random"] + cfg["ext_extra_random"]
    pilot = rest[pilot_start: pilot_start + cfg["pilot_cases"]]

    # core (75) and sequential (150) subsets of the panel
    panel_by_p: dict[str, list[int]] = defaultdict(list)
    for i in panel:
        panel_by_p[test_rows[i]["pathology"]].append(i)
    core, seq = [], []
    for p in kb.pathologies:
        lst = panel_by_p[p]
        if lst:
            core.append(lst[0])
        seq.extend(lst[:3])
    left = [i for i in panel if i not in set(core)]
    perm = rng(seed, "core", "extra").permutation(len(left))
    core += [left[j] for j in perm[: cfg["core_cases"] - len(core)]]
    left = [i for i in panel if i not in set(seq)]
    perm = rng(seed, "seq", "extra").permutation(len(left))
    seq += [left[j] for j in perm[: cfg["sequential_cases"] - len(seq)]]
    core_s, seq_s = set(core), set(seq)

    records: list[dict] = []
    for role, ids in (("pilot", pilot), ("panel", panel), ("panel_ext", ext)):
        for i in ids:
            r = test_rows[i]
            units = dx.build_units(kb, r["evidences"], r["initial_evidence"])
            rec = case_record(kb, r, units, role)
            if role == "panel":
                rec["flags"] = {"core": i in core_s, "sequential": i in seq_s}
            records.append(rec)

    log("pass 2/2 over train split (distillation cases) ...")
    for r in dx.iter_split("train", raw_dir):
        if r["row_id"] in chosen_train_set:
            units = dx.build_units(kb, r["evidences"], r["initial_evidence"])
            records.append(case_record(kb, r, units, "distill"))
    records.sort(key=lambda c: (["pilot", "panel", "panel_ext", "distill"].index(c["role"]), c["case_id"]))
    write_jsonl(out_dir / "cases.jsonl", records)
    fit_reference(out_dir, verbose=verbose)

    # ---- KB-Bayes fit report (on panel cases) — reported, never assumed
    fit = {"n": 0, "kb_bayes_top1_eq_pathology": 0, "kb_bayes_top1_eq_ddx_top1": 0, "mean_jsd_to_ddx": 0.0}
    for c in records:
        if c["role"] != "panel":
            continue
        q = lik.posterior(c["tokens"])
        ref = np.zeros(kb.K)
        for n, p in c["differential"]:
            ref[kb.pathologies.index(n)] = p
        ref = ref / ref.sum()
        fit["n"] += 1
        fit["kb_bayes_top1_eq_pathology"] += int(kb.pathologies[int(np.argmax(q))] == c["pathology"])
        fit["kb_bayes_top1_eq_ddx_top1"] += int(np.argmax(q) == np.argmax(ref))
        fit["mean_jsd_to_ddx"] += jsd(q, ref)
    if fit["n"]:
        fit["kb_bayes_top1_eq_pathology"] /= fit["n"]
        fit["kb_bayes_top1_eq_ddx_top1"] /= fit["n"]
        fit["mean_jsd_to_ddx"] /= fit["n"]
    report = {
        "seed": seed,
        "counts": {"pilot": len(pilot), "panel": len(panel), "panel_ext": len(ext),
                   "core": len(core), "sequential": len(seq), "distill": len(chosen_train)},
        "eligible_test_pool": {p: len(v) for p, v in sorted(pool_by_p.items())},
        "pathology_shortfall": shortfall,
        "kb_bayes_fit_on_panel": fit,
        "raw_md5": {k: v[1] for k, v in cfg["files"].items()},
    }
    write_json(out_dir / "build_report.json", report)
    log("counts:", report["counts"])
    log("KB-Bayes fit:", fit)
    return report


def fit_reference(out_dir: Path | None = None, verbose: bool = True) -> dict:
    """Fit the reference-posterior temperature on the TRAIN-split distillation
    cases (their released differentials). Writes reference.json."""
    from ccrl.util import read_json, read_jsonl
    out_dir = Path(out_dir or paths.FROZEN_DDX)
    lik = dx.Likelihood.from_json(read_json(out_dir / "likelihood.json"))
    cases = [c for c in read_jsonl(out_dir / "cases.jsonl") if c["role"] == "distill"]
    res = dx.fit_tau(cases, lik)
    write_json(out_dir / "reference.json", res)
    dx.load_likelihood.cache_clear()
    if verbose:
        print(f"[build-ddxplus] reference tau={res['tau']} (NLL {res['nll_at_tau']:.4f}, top-1 "
              f"{res['top1_at_tau']:.4f}, entropy {res['entropy_at_tau']:.3f} nats, n={res['n_cases']})")
    return res
