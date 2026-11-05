"""Robust parsing of a stated posterior from a model response.

Design decisions, each from a documented failure in the previous study
(Probabilistic Coherence, instrument calibration C1-C11):

* The LAST JSON object containing the answer key is used (reasoning models
  may echo drafts first); ``raw_decode`` is used so trailing text is harmless
  (C1). ``<think>...</think>`` blocks and Markdown fences are stripped.
* An all-zero vector is a valid answer that expresses no preference: it is
  parsed as UNIFORM and flagged non-informative, not rejected (C8).
* Sums other than 100 are accepted and renormalised (the transform is
  monotone, it preserves what the model expressed); the raw sum is kept.
* Missing codes count as 0 and are counted; unknown codes are dropped and
  counted. Neither is fatal, so formatting quirks cannot silently remove
  items from one transformation arm more than another (C4).
* A response is a parse FAILURE only if no object with the answer key exists,
  or values are non-numeric/negative/non-finite, or nothing maps to a code.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field

import numpy as np

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_OPEN = re.compile(r"^.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?", re.I)
_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*%?\s*$")


@dataclass
class ParseResult:
    ok: bool
    probs: list[float] | None = None        # aligned to `codes`, sums to 1
    dx: str | None = None                   # stated leading diagnosis code (validated)
    raw_sum: float | None = None
    n_missing: int = 0
    n_unknown: int = 0
    all_zero: bool = False
    informative: bool = False               # not exactly uniform
    error: str | None = None
    flags: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    @property
    def vector(self) -> np.ndarray | None:
        return None if self.probs is None else np.asarray(self.probs, dtype=np.float64)


def clean_text(text: str) -> str:
    t = text or ""
    t = _THINK.sub(" ", t)
    if "</think>" in t.lower():          # opening tag consumed by the template
        t = _THINK_OPEN.sub(" ", t)
    t = _FENCE.sub(" ", t)
    return t


def json_objects(text: str) -> list[dict]:
    """All top-level JSON objects decodable from the text, in order."""
    dec = json.JSONDecoder()
    out = []
    i = 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = end
    return out


def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        m = _NUM.match(v)
        if not m:
            return None
        x = float(m.group(1))
    else:
        return None
    return x if math.isfinite(x) else None


def parse_posterior(text: str, codes: list[str], key: str = "p",
                    names: dict[str, str] | None = None) -> ParseResult:
    """Parse a stated distribution over `codes` from a response text.

    key: 'p' (percentages, the main schema) or 's' (0-100 plausibility scores,
    the pilot schema variant). Both are normalised the same way.
    """
    objs = [o for o in json_objects(clean_text(text)) if isinstance(o.get(key), dict)]
    if not objs:
        alt = "probabilities" if key == "p" else None
        objs = [o for o in json_objects(clean_text(text)) if alt and isinstance(o.get(alt), dict)]
        if objs:
            key = alt
    if not objs:
        return ParseResult(ok=False, error=f"no JSON object with key {key!r}")
    obj = objs[-1]
    raw = obj[key]
    code_set = set(codes)
    upper = {str(k).strip().upper(): v for k, v in raw.items()}
    # Accept full names as keys if the model used them instead of codes.
    if names and not (set(upper) & code_set):
        by_name = {n.strip().lower(): c for c, n in names.items()}
        upper = {by_name.get(str(k).strip().lower(), str(k)): v for k, v in raw.items()}
    vals = np.zeros(len(codes))
    seen = 0
    unknown = 0
    for k, v in upper.items():
        if k not in code_set:
            unknown += 1
            continue
        x = _num(v)
        if x is None:
            return ParseResult(ok=False, error=f"non-numeric value for {k}: {v!r}")
        if x < 0:
            return ParseResult(ok=False, error=f"negative value for {k}: {x}")
        vals[codes.index(k)] = x
        seen += 1
    if seen == 0:
        return ParseResult(ok=False, error="no candidate code found among keys", n_unknown=unknown)
    s = float(vals.sum())
    res = ParseResult(ok=True, raw_sum=s, n_missing=len(codes) - seen, n_unknown=unknown)
    if s <= 0:
        res.all_zero = True
        res.flags.append("all_zero_as_uniform")
        p = np.full(len(codes), 1.0 / len(codes))
    else:
        p = vals / s
    res.probs = [float(x) for x in p]
    res.informative = not bool(np.allclose(p, 1.0 / len(codes), atol=1e-9))
    if key == "p" and abs(s - 100.0) > 1.0:
        res.flags.append("sum_not_100")
    if res.n_missing:
        res.flags.append("missing_codes")
    if res.n_unknown:
        res.flags.append("unknown_codes")
    dx = obj.get("dx")
    if isinstance(dx, str):
        d = dx.strip().upper()
        if d in code_set:
            res.dx = d
        elif names:
            by_name = {n.strip().lower(): c for c, n in names.items()}
            res.dx = by_name.get(dx.strip().lower())
        if res.dx is None:
            res.flags.append("dx_invalid")
    else:
        res.flags.append("dx_missing")
    return res


def parse_json_object(text: str, required_keys: tuple[str, ...] = ()) -> dict | None:
    """Last JSON object in the text that has all required keys (prep tasks)."""
    objs = json_objects(clean_text(text))
    for o in reversed(objs):
        if all(k in o for k in required_keys):
            return o
    return None
