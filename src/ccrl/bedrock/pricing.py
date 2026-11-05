"""Cost accounting from returned usage, and the worst-case reservation.

Two numbers per call:
  reported cost  — usage x listed price, Flex discount applied when the
                   response says it was served on Flex (what we expect AWS
                   to bill; reconciled against Cost Explorer);
  guard cost     — usage x listed price x safety factor (unverified rows),
                   never discounted. The budget guard uses only this.

Cache semantics: Bedrock reports cacheRead/WriteInputTokens separately.
Whether `inputTokens` already includes them is detected per response from
`totalTokens`; when it cannot be decided the conservative reading (cached
tokens billed on top) is used.
"""
from __future__ import annotations

from dataclasses import dataclass

from ccrl import config

M = 1_000_000.0


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float
    verified: bool

    @property
    def guard_factor(self) -> float:
        return 1.0 if self.verified else float(config.prices().get("unverified_safety_factor", 1.25))


def price_for(price_key: str) -> Price:
    p = config.prices()["prices"][price_key]
    return Price(float(p["input"]), float(p["output"]), float(p.get("cache_read", p["input"])),
                 float(p.get("cache_write", p["input"])), bool(p.get("verified", False)))


def usage_tokens(usage: dict) -> dict:
    """Normalise a Converse usage block into uncached/cached/output counts."""
    inp = int(usage.get("inputTokens", 0) or 0)
    out = int(usage.get("outputTokens", 0) or 0)
    cr = int(usage.get("cacheReadInputTokens", 0) or 0)
    cw = int(usage.get("cacheWriteInputTokens", 0) or 0)
    total = usage.get("totalTokens")
    semantics = "no_cache"
    uncached = inp
    if cr or cw:
        if total is not None and int(total) == inp + out:
            semantics = "inclusive"          # inputTokens already contains cached tokens
            uncached = max(0, inp - cr - cw)
        else:
            semantics = "exclusive"          # cached tokens are extra (also the conservative default)
    return {"uncached_input": uncached, "cache_read": cr, "cache_write": cw, "output": out,
            "input_reported": inp, "semantics": semantics}


def cost(usage: dict, price_key: str, served_tier: str | None) -> tuple[float, float]:
    """(reported_cost, guard_cost) in USD for one response."""
    p = price_for(price_key)
    u = usage_tokens(usage)
    base = (u["uncached_input"] * p.input + u["cache_read"] * p.cache_read
            + u["cache_write"] * p.cache_write + u["output"] * p.output) / M
    mult = float(config.prices().get("flex_multiplier", 0.5)) if (served_tier or "").lower() == "flex" else 1.0
    return base * mult, base * p.guard_factor


def worst_case(input_tokens_est: int, max_tokens: int, price_key: str) -> float:
    """Upper bound reserved before sending: every input token uncached at full
    price (ignores caching and Flex), every allowed output token used."""
    p = price_for(price_key)
    return (input_tokens_est * p.input + max_tokens * p.output) / M * p.guard_factor


def estimate_tokens(text: str) -> int:
    """Conservative local token estimate (no tokenizer needed): ~3 chars/token
    for English clinical text, rounded up, +5% — over-estimates, which is the
    safe direction for a reservation."""
    return int(len(text) / 3.0 * 1.05) + 16
