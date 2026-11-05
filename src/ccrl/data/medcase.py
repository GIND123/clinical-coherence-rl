"""MedCaseReasoning: case selection for preprocessing (deterministic).

MedCaseReasoning (Wu et al., arXiv:2505.11733; Hugging Face
``zou-lab/MedCaseReasoning``, license MIT per the dataset card at the pinned
revision) holds diagnostic cases derived from open-access PubMed Central case
reports. Each row has a ``case_prompt`` (the presentation, written to stop
before the diagnosis is revealed), clinician-style ``diagnostic_reasoning``
and the ``final_diagnosis``.

This module only *selects* which cases enter preprocessing and in which
order. The LLM-assisted extraction of atomic findings, time classes and the
10-candidate differential happens in ``ccrl.prep.medcase`` (paid, on Bedrock).

More cases are sent to preprocessing than are used (``*_prep`` vs
``*_cases`` in study.yaml): a case that fails verification is replaced by the
next case in the frozen order, so the final sample is a deterministic
function of the seed and the verification outcomes — never of a judgement
made after seeing model answers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ccrl import config, paths
from ccrl.util import rng, write_json, write_jsonl

SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "val": "val-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


def load_split(split: str, raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir or paths.DATA_MCR)
    p = raw_dir / SPLIT_FILES[split]
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `ccrl data download medcase` (or set CCRL_DATA)")
    df = pd.read_parquet(p)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    return df


def _filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = df.copy()
    s["case_prompt"] = s["case_prompt"].astype(str).str.strip()
    s["final_diagnosis"] = s["final_diagnosis"].astype(str).str.strip()
    n = s["case_prompt"].str.len()
    s = s[(n >= cfg["min_prompt_chars"]) & (n <= cfg["max_prompt_chars"])]
    s = s[s["final_diagnosis"].str.len() > 1]
    s = s.drop_duplicates(subset=["pmcid"], keep="first")
    return s.sort_values("pmcid").reset_index(drop=True)


def _record(row: pd.Series, split: str, role: str, order: int) -> dict:
    return {
        "case_id": f"mcr-{split}-{row['pmcid']}",
        "dataset": "medcase",
        "split": split,
        "role": role,
        "prep_order": int(order),
        "pmcid": str(row["pmcid"]),
        "title": str(row.get("title", "")),
        "journal": str(row.get("journal", "")),
        "publication_date": str(row.get("publication_date", "")),
        "article_link": str(row.get("article_link", "")),
        "case_prompt": str(row["case_prompt"]),
        "final_diagnosis_raw": str(row["final_diagnosis"]),
    }


def select(raw_dir: Path | None = None, out_dir: Path | None = None, verbose: bool = True) -> dict:
    st = config.study()
    cfg = st["medcase"]
    seed = int(st["seed"])
    out_dir = Path(out_dir or paths.FROZEN_MCR)

    test = _filter(load_split("test", raw_dir), cfg)
    train = _filter(load_split("train", raw_dir), cfg)
    train = train[~train["pmcid"].isin(set(test["pmcid"]))].reset_index(drop=True)

    recs: list[dict] = []
    perm = rng(seed, "medcase", "test").permutation(len(test))
    n_pilot, n_panel = cfg["pilot_prep"], cfg["panel_prep"]
    for k, i in enumerate(perm[:n_pilot]):
        recs.append(_record(test.iloc[int(i)], "test", "pilot", k))
    for k, i in enumerate(perm[n_pilot:n_pilot + n_panel]):
        recs.append(_record(test.iloc[int(i)], "test", "panel", k))
    perm = rng(seed, "medcase", "train").permutation(len(train))
    for k, i in enumerate(perm[: cfg["train_prep"]]):
        recs.append(_record(train.iloc[int(i)], "train", "train", k))

    write_jsonl(out_dir / "cases_selected.jsonl", recs)
    report = {
        "seed": seed,
        "eligible": {"test": len(test), "train": len(train)},
        "selected": {r: sum(1 for x in recs if x["role"] == r) for r in ("pilot", "panel", "train")},
        "hf_repo": cfg["hf_repo"], "hf_revision": cfg["hf_revision"],
        "license": "MIT (dataset card at pinned revision)",
    }
    write_json(out_dir / "selection_report.json", report)
    if verbose:
        print("[select-medcase]", report)
    return report
