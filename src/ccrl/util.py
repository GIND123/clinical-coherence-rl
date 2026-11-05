"""Small shared helpers: hashing, deterministic RNG, JSON/JSONL I/O, time."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


# ---------------------------------------------------------------- hashing
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable serialisation: sorted keys, no whitespace, UTF-8 preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(*parts: Any, n: int = 16) -> str:
    """Short deterministic hex id from arbitrary JSON-able parts."""
    return sha256_bytes(canonical_json(list(parts)).encode("utf-8"))[:n]


# ---------------------------------------------------------------- randomness
def derive_seed(master: int, *labels: Any) -> int:
    """Independent, reproducible seed for one named random draw.

    Every random decision in the study (case sampling, permutations, cue
    choice, ...) gets its own seed derived from the master seed and a label,
    so adding a new draw never shifts an existing one.
    """
    digest = hashlib.sha256(canonical_json([master, *labels]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def rng(master: int, *labels: Any) -> np.random.Generator:
    return np.random.default_rng(derive_seed(master, *labels))


def py_rng(master: int, *labels: Any) -> random.Random:
    return random.Random(derive_seed(master, *labels))


# ---------------------------------------------------------------- JSON I/O
def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any, indent: int | None = 2) -> None:
    """Atomic write: a crash never leaves a half-written file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent, sort_keys=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:  # pragma: no cover - corruption guard
                raise ValueError(f"{path}:{i}: invalid JSON line: {e}") from e
    return out


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Atomic JSONL write with sorted keys, so committed files diff cleanly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True))
                f.write("\n")
                n += 1
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return n


def append_jsonl(path: Path, row: dict) -> None:
    """Durable append (flush + fsync) for the raw-response mirror."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------- time
def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_local(ts: str) -> _dt.datetime:
    """Parse a naive ISO timestamp from config as local time."""
    d = _dt.datetime.fromisoformat(ts)
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.datetime.now().astimezone().tzinfo)
    return d


def local_now() -> _dt.datetime:
    return _dt.datetime.now().astimezone()


# ---------------------------------------------------------------- misc
def two_letter_codes(n: int) -> list[str]:
    """AA, AB, ..., AZ, BA, ... — compact, single-token-ish candidate codes."""
    if n > 26 * 26:
        raise ValueError("too many candidates for two-letter codes")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [letters[i // 26] + letters[i % 26] for i in range(n)]


def chunked(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
