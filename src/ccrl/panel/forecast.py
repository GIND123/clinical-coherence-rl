"""Cost forecast for planned tasks (no API calls).

Input tokens are estimated locally from the exact prompt text (conservative,
~3 chars/token); cached system prefixes are priced at the cache-read rate for
models whose lock enables caching. Output tokens come from:
  * the pilot, once it exists: per (model) mean (expected) and p90 (high)
    output tokens of completed pilot calls; medium/high effort are scaled by
    the model's measured/assumed ratio at its default effort;
  * otherwise the assumptions in config/budget.yaml (low / expected / high).
The Flex discount is applied to the low and expected scenarios only; the high
scenario (and the budget guard) assume Standard pricing.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ccrl import config
from ccrl.bedrock import models as M
from ccrl.bedrock import pricing
from ccrl.bedrock.ledger import Ledger
from ccrl.workspace import Workspace

SCEN = ("low", "expected", "high")


def pilot_output_stats(ws: Workspace) -> dict[str, dict]:
    if not ws.ledger.exists():
        return {}
    led = Ledger(ws.ledger)
    try:
        rows = led.q("SELECT r.model_key AS m, a.output_tokens AS o, a.input_tokens AS i FROM attempts a "
                     "JOIN requests r USING(request_id) WHERE r.job='pilot' AND a.status IN ('ok','truncated') "
                     "AND a.output_tokens IS NOT NULL")
    finally:
        led.close()
    by: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by[r["m"]].append(int(r["o"]))
    out = {}
    for m, v in by.items():
        if len(v) >= 10:
            a = np.asarray(v, dtype=float)
            out[m] = {"n": len(v), "mean": float(a.mean()), "median": float(np.median(a)),
                      "p90": float(np.quantile(a, 0.9)), "low": float(np.quantile(a, 0.25))}
    return out


def _assumed(effort: str | None, strategy_none: bool) -> dict[str, float]:
    ft = config.budget()["forecast_tokens"]["output"]
    key = "provider_default" if strategy_none else (effort or "low")
    return {k: float(v) for k, v in ft.get(key, ft["low"]).items()}


def output_tokens(model_key: str, effort: str | None, strategy_none: bool, pilot: dict) -> dict[str, float]:
    a = _assumed(effort, strategy_none)
    p = pilot.get(model_key)
    if not p:
        return a
    base = _assumed(None if strategy_none else "low", strategy_none)
    if effort in (None, "low") or strategy_none:
        return {"low": p["low"], "expected": p["mean"], "high": p["p90"]}
    ratio = p["mean"] / max(1.0, base["expected"])
    return {k: v * ratio for k, v in a.items()}


def forecast_tasks(tasks: list, ws: Workspace, specs: dict | None = None) -> dict:
    if specs is None:
        specs = M.load_lock(ws.dry_run)
    pilot = pilot_output_stats(ws)
    retry = config.budget()["forecast_tokens"]["retry_overhead"]
    flex = float(config.prices().get("flex_multiplier", 0.5))
    by_model: dict[str, dict] = defaultdict(lambda: {"calls": 0, "low": 0.0, "expected": 0.0, "high": 0.0})
    sys_tok_cache: dict[str, int] = {}
    for t in tasks:
        spec = specs[t.model_key]
        p = pricing.price_for(spec.price_key)
        effort = spec.resolve_effort(t.effort)
        sys_t = sys_tok_cache.setdefault(t.system, pricing.estimate_tokens(t.system))
        n_turns = len(t.turns)
        user_t = sum(pricing.estimate_tokens(u) for u in t.turns)
        out = output_tokens(t.model_key, effort, spec.effort_strategy == "none", pilot)
        for s in SCEN:
            # every turn re-sends the system block and the conversation so far
            in_unc = user_t * n_turns + (out[s] * (n_turns - 1))
            sys_cost = sys_t * n_turns * (p.cache_read if spec.prompt_cache else p.input)
            c = (in_unc * p.input + sys_cost + out[s] * n_turns * p.output) / 1e6
            c *= 1.0 + float(retry[s])
            if spec.service_tier == "flex" and s != "high":   # high scenario: assume Flex not honoured
                c *= flex
            by_model[t.model_key][s] += c
        by_model[t.model_key]["calls"] += n_turns
    tot = {s: sum(v[s] for v in by_model.values()) for s in SCEN}
    tot["calls"] = sum(v["calls"] for v in by_model.values())
    return {"by_model": dict(by_model), "total": tot, "pilot_measured": sorted(pilot)}
