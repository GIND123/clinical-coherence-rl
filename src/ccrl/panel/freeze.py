"""Freeze: hash everything that determines what a model is asked.

    ccrl freeze            write artifacts/frozen/FREEZE.json
    ccrl freeze --check    verify nothing changed since

Covered: config/*.yaml (including models.lock.yaml; excluding the advisory
models.resolved.yaml), config/prompts/*.txt, the committed DDXPlus and
MedCaseReasoning inputs, the phrase bank, the prepared MedCaseReasoning
cases and the built items. Paid panel jobs refuse to start when the check
fails, so the design registered on OSF is exactly the design that runs.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ccrl import paths
from ccrl.util import now_iso, read_json, sha256_file, write_json
from ccrl.workspace import Workspace

EXCLUDE_NAMES = {"models.resolved.yaml", "FREEZE.json", "build_summary.json"}


def covered_files(ws: Workspace) -> dict[str, Path]:
    files: dict[str, Path] = {}

    def add(base: Path, pattern: str, label: str) -> None:
        if not base.exists():
            return
        for p in sorted(base.glob(pattern)):
            if p.is_file() and p.name not in EXCLUDE_NAMES and not p.name.endswith(".tmp"):
                files[f"{label}/{p.relative_to(base).as_posix()}"] = p

    add(paths.CONFIG, "*.yaml", "config")
    add(paths.PROMPTS, "*.txt", "config/prompts")
    add(paths.FROZEN_DDX, "*", "frozen/ddxplus")
    add(paths.FROZEN_MCR, "cases_selected.jsonl", "frozen/medcase")
    add(ws.banks, "*", "frozen/banks")
    add(ws.mcr_dir, "prepared.jsonl", "frozen/medcase")
    add(ws.items_dir, "*", "frozen/items")
    return files


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def freeze(ws: Workspace) -> dict:
    files = covered_files(ws)
    if not any(k.startswith("frozen/items/") for k in files):
        raise SystemExit("No built items to freeze: run `ccrl build items` first.")
    doc = {"created_at": now_iso(), "git_commit": git_commit(), "dry_run": ws.dry_run,
           "files": {k: sha256_file(p) for k, p in sorted(files.items())}}
    write_json(ws.freeze_file, doc)
    return doc


def check_freeze(ws: Workspace) -> tuple[bool, str]:
    if not ws.freeze_file.exists():
        return False, f"no freeze file at {ws.freeze_file}"
    doc = read_json(ws.freeze_file)
    want = doc["files"]
    have = {k: sha256_file(p) for k, p in covered_files(ws).items()}
    changed = sorted(k for k in want if k in have and have[k] != want[k])
    missing = sorted(k for k in want if k not in have)
    added = sorted(k for k in have if k not in want)
    if changed or missing or added:
        parts = []
        if changed:
            parts.append(f"changed: {changed}")
        if missing:
            parts.append(f"missing: {missing}")
        if added:
            parts.append(f"new: {added}")
        return False, "; ".join(parts)
    return True, f"OK ({len(want)} files, frozen {doc['created_at']})"
