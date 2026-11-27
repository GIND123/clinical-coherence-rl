"""Go/no-go check after step 50 of the first coherence run (plan_final.md §3.5, 27 Nov 2026).

    ccrl train gonogo --run runs/train/ccr_acc_fid_s0 --pilot <step-50 pilot eval> --base <untrained pilot eval>

All four must hold, or the queue stops and the rest of the money is kept:
  1. sum-to-100 compliance of the rollouts >= 0.8 (mean over steps 41-50 of the training log);
  2. the collapse safeguard has not fired;
  3. pilot posterior entropy (base renderings) <= the untrained model's + 0.5 nats;
  4. pilot nuisance flip rate <= the untrained model's + 0.05 (30 pilot cases: the allowance is
     about one standard error).
Exit code 0 = go, 3 = no-go. Writes <run>/gonogo.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _entropy(parquet: str) -> float:
    d = pd.read_parquet(parquet)
    d = d[(d["transform"] == "base") & d["parse_ok"].astype(bool) & d["probs"].notna()]
    P = [np.clip(np.asarray(p, float), 1e-12, 1) for p in d["probs"]]
    return float(np.mean([-(p * np.log(p)).sum() for p in P]))


def _flip(parquet: str) -> float:
    from ccrl.train.hypotheses import nuisance_flips
    d = pd.read_parquet(parquet)
    d = d[d["parse_ok"].astype(bool) & d["probs"].notna()]
    return float(nuisance_flips(d)["nuisance_flip"].mean())


def check(run_dir: str, pilot: str, base: str, at_step: int = 50) -> dict:
    rows = [json.loads(x) for x in Path(run_dir, "train_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    steps = [r for r in rows if "sec" in r and at_step - 10 < r["step"] <= at_step]
    fired = any(r.get("event") == "early_stop_M_collapse" for r in rows)
    sums = [r["sum100_frac"] for r in steps if r.get("sum100_frac") is not None]
    sum100 = float(np.mean(sums)) if sums else float("nan")
    e_new, e_base = _entropy(pilot), _entropy(base)
    f_new, f_base = _flip(pilot), _flip(base)
    crit = {"sum100 >= 0.8": bool(sum100 >= 0.8), "safeguard not fired": not fired,
            "entropy <= base + 0.5": bool(e_new <= e_base + 0.5), "flip <= base + 0.05": bool(f_new <= f_base + 0.05)}
    res = {"at_step": at_step, "sum100": sum100, "entropy": e_new, "entropy_base": e_base, "flip": f_new,
           "flip_base": f_base, "criteria": crit, "go": all(crit.values())}
    Path(run_dir, "gonogo.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res
