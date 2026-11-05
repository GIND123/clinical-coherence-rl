"""Filesystem layout. Every other module imports locations from here.

Three roots, each overridable by environment variable so the same checkout
can run on the PI's laptop (panel) and on a GPU workstation (training):

    CCRL_DATA  raw corpora (DDXPlus zips, MedCaseReasoning parquet) — never committed
    CCRL_RUNS  ledger database, raw responses, exports — never committed; archived
    (repo)     config/ and artifacts/frozen/ — committed, hashed by `ccrl freeze`
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIG = ROOT / "config"
PROMPTS = CONFIG / "prompts"
ARTIFACTS = ROOT / "artifacts"
FROZEN = ARTIFACTS / "frozen"
FROZEN_DDX = FROZEN / "ddxplus"
FROZEN_MCR = FROZEN / "medcase"
FROZEN_BANKS = FROZEN / "banks"
FROZEN_ITEMS = FROZEN / "items"
FREEZE_FILE = FROZEN / "FREEZE.json"
MODELS_LOCK = CONFIG / "models.lock.yaml"
MODELS_RESOLVED = CONFIG / "models.resolved.yaml"

DATA = Path(os.environ.get("CCRL_DATA", str(ROOT / "data"))).expanduser()
DATA_DDX = DATA / "ddxplus"
DATA_MCR = DATA / "medcase"

RUNS = Path(os.environ.get("CCRL_RUNS", str(ROOT / "runs"))).expanduser()


def runs_dir(dry_run: bool = False) -> Path:
    """Dry runs get their own tree so they can never touch the real ledger."""
    p = RUNS / "dryrun" if dry_run else RUNS
    p.mkdir(parents=True, exist_ok=True)
    return p


def ledger_path(dry_run: bool = False) -> Path:
    return runs_dir(dry_run) / "ledger.sqlite"
