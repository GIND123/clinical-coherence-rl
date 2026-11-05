"""Where a command reads and writes, for real runs and for dry runs.

A dry run (`--dry-run`) exercises every stage end to end with the offline mock
client. It reads the committed inputs (KB, selected cases, config) but writes
EVERYTHING it produces — phrase bank, prepared cases, items, ledger, raw
responses, exports — under runs/dryrun/, so a rehearsal can never overwrite a
frozen artifact or touch the real ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ccrl import paths


@dataclass(frozen=True)
class Workspace:
    dry_run: bool = False

    @property
    def runs(self) -> Path:
        return paths.runs_dir(self.dry_run)

    @property
    def frozen(self) -> Path:
        p = (paths.RUNS / "dryrun" / "frozen") if self.dry_run else paths.FROZEN
        return p

    @property
    def banks(self) -> Path:
        return self.frozen / "banks"

    @property
    def mcr_dir(self) -> Path:
        return self.frozen / "medcase"

    @property
    def items_dir(self) -> Path:
        return self.frozen / "items"

    @property
    def phrasebank(self) -> Path:
        return self.banks / "ddx_phrasebank.json"

    @property
    def mcr_prepared(self) -> Path:
        return self.mcr_dir / "prepared.jsonl"

    @property
    def ledger(self) -> Path:
        return self.runs / "ledger.sqlite"

    @property
    def raw(self) -> Path:
        return self.runs / "raw"

    @property
    def export(self) -> Path:
        return self.runs / "export"

    @property
    def reports(self) -> Path:
        return self.runs / "reports"

    @property
    def freeze_file(self) -> Path:
        return self.frozen / "FREEZE.json"

    def ensure(self) -> "Workspace":
        for p in (self.runs, self.banks, self.mcr_dir, self.items_dir, self.raw, self.export, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get(dry_run: bool = False) -> Workspace:
    return Workspace(dry_run=dry_run).ensure()
