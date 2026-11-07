"""Load and validate the YAML configuration in ``config/``.

Validation is strict and runs before any paid command: a typo in a model key,
a job pointing at a model with no price, or a cap above the credit is an
error *before* a single request is sent, not a surprise halfway through.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ccrl import paths

EFFORT_STRATEGIES = ("output_config", "reasoning_field", "none")
SERVICE_TIERS = ("default", "flex", "priority")
ITEM_SETS = ("pilot", "panel", "core", "sequential", "distill", "panel_ext")


class ConfigError(ValueError):
    pass


def load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def study() -> dict:
    return load_yaml(paths.CONFIG / "study.yaml")


@lru_cache(maxsize=None)
def budget() -> dict:
    return load_yaml(paths.CONFIG / "budget.yaml")


@lru_cache(maxsize=None)
def prices() -> dict:
    return load_yaml(paths.CONFIG / "prices.yaml")


@lru_cache(maxsize=None)
def jobs() -> dict:
    return load_yaml(paths.CONFIG / "jobs.yaml")["jobs"]


@lru_cache(maxsize=None)
def cues() -> dict:
    return load_yaml(paths.CONFIG / "cues.yaml")


@lru_cache(maxsize=None)
def _models_raw() -> dict:
    return load_yaml(paths.CONFIG / "models.yaml")


def clear_caches() -> None:
    for fn in (study, budget, prices, jobs, cues, _models_raw):
        fn.cache_clear()


@dataclass
class ModelSpec:
    key: str
    display: str
    vendor: str
    group: str
    id_candidates: list[str]
    match: str
    effort: dict
    service_tier: str
    prompt_cache: bool
    optional_fields: dict
    max_tokens: dict
    concurrency: int
    region: str
    read_timeout_s: int
    connect_timeout_s: int
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def effort_strategy(self) -> str:
        return self.effort.get("strategy", "none")

    @property
    def default_effort(self) -> str | None:
        return self.effort.get("default") if self.effort_strategy != "none" else None

    def resolve_effort(self, effort: str) -> str | None:
        """Map a job's effort label ('default', 'low', ...) to what is sent."""
        if self.effort_strategy == "none":
            return None
        return self.default_effort if effort == "default" else effort

    def max_tokens_for(self, effort: str | None) -> int:
        key = effort if effort in self.max_tokens else "default"
        return int(self.max_tokens[key])


def models() -> dict[str, ModelSpec]:
    raw = _models_raw()
    d = raw.get("defaults", {})
    out: dict[str, ModelSpec] = {}
    for key, m in raw["models"].items():
        mt = dict(d.get("max_tokens", {}))
        mt.update(m.get("max_tokens", {}))
        out[key] = ModelSpec(
            key=key,
            display=m["display"],
            vendor=m["vendor"],
            group=m["group"],
            id_candidates=list(m.get("id_candidates", [])),
            match=m.get("match", ""),
            effort=dict(m.get("effort", {"strategy": "none"})),
            service_tier=m.get("service_tier", d.get("service_tier", "default")),
            prompt_cache=bool(m.get("prompt_cache", d.get("prompt_cache", False))),
            optional_fields=copy.deepcopy(m.get("optional_fields", {})),
            max_tokens=mt,
            concurrency=int(m.get("concurrency", d.get("concurrency", 4))),
            region=m.get("region", d.get("region", "us-east-1")),
            read_timeout_s=int(m.get("read_timeout_s", d.get("read_timeout_s", 1200))),
            connect_timeout_s=int(m.get("connect_timeout_s", d.get("connect_timeout_s", 30))),
            raw=m,
        )
    return out


def panel_models() -> list[str]:
    return list(_models_raw()["panel"])


def optional_models() -> list[str]:
    """Registry entries marked `optional: true` (partner models the account
    cannot call). They are probed or called only when named explicitly."""
    return [k for k, m in _models_raw()["models"].items() if m.get("optional")]


def study_models() -> list[str]:
    """Models the study depends on: everything in the registry except optional ones."""
    opt = set(optional_models())
    return [k for k in _models_raw()["models"] if k not in opt]


def role_model(role: str) -> str:
    return _models_raw()["roles"][role]


def job_models(job_name: str) -> list[str]:
    j = jobs()[job_name]
    m = j["models"]
    return panel_models() if m == "panel" else list(m)


def validate_all() -> list[str]:
    """Return a list of problems (empty = OK). Never raises on content."""
    problems: list[str] = []
    ms = models()
    pr = prices()["prices"]
    raw = _models_raw()
    for k, m in ms.items():
        if m.effort_strategy not in EFFORT_STRATEGIES:
            problems.append(f"models.{k}: unknown effort.strategy {m.effort_strategy!r}")
        for fb in m.effort.get("fallbacks", []) or []:
            if fb not in EFFORT_STRATEGIES:
                problems.append(f"models.{k}: unknown effort fallback {fb!r}")
        if m.service_tier not in SERVICE_TIERS:
            problems.append(f"models.{k}: unknown service_tier {m.service_tier!r}")
        if k not in pr:
            problems.append(f"models.{k}: no row in prices.yaml")
        if not m.id_candidates and not m.match:
            problems.append(f"models.{k}: needs id_candidates or match")
    opt = set(optional_models())
    for k in raw.get("panel", []):
        if k not in ms:
            problems.append(f"panel: unknown model {k!r}")
        elif k in opt:
            problems.append(f"panel: {k!r} is marked optional; the panel must use available models")
    for role, k in raw.get("roles", {}).items():
        if k not in ms:
            problems.append(f"roles.{role}: unknown model {k!r}")
        elif k in opt:
            problems.append(f"roles.{role}: {k!r} is marked optional")
    rk = raw.get("roles", {})
    if rk.get("preprocessor") in ms and rk.get("verifier") in ms and             ms[rk["preprocessor"]].vendor == ms[rk["verifier"]].vendor:
        problems.append("roles: verifier must be from a different vendor than the preprocessor")
    st = study()
    tsets = st["transform_sets"]
    for name, j in jobs().items():
        if j["item_set"] not in ITEM_SETS:
            problems.append(f"jobs.{name}: unknown item_set {j['item_set']!r}")
        if j["transform_set"] not in tsets and j["transform_set"] != "sequential":
            problems.append(f"jobs.{name}: unknown transform_set {j['transform_set']!r}")
        for mk in job_models(name):
            if mk not in ms:
                problems.append(f"jobs.{name}: unknown model {mk!r}")
        for e in j.get("efforts", ["default"]):
            if e not in ("default", "low", "medium", "high", "xhigh", "max"):
                problems.append(f"jobs.{name}: unknown effort {e!r}")
            if e != "default":
                for mk in job_models(name):
                    if mk in ms and ms[mk].effort_strategy == "none":
                        problems.append(f"jobs.{name}: effort {e!r} requested for {mk}, "
                                        "which has effort.strategy none")
        if j.get("priority") not in ("A", "B", "C"):
            problems.append(f"jobs.{name}: priority must be A, B or C")
        if j.get("priority") in ("A", "B") and any(mk in opt for mk in job_models(name)):
            problems.append(f"jobs.{name}: priority {j['priority']} job uses an optional model")
        for dep in j.get("requires_completed", []) or []:
            if dep not in jobs():
                problems.append(f"jobs.{name}: requires unknown job {dep!r}")
        if name not in budget()["job_caps_usd"]:
            problems.append(f"jobs.{name}: no entry in budget.job_caps_usd")
    b = budget()
    if b["hard_cap_usd"] > b["credit_usd"]:
        problems.append("budget: hard_cap_usd exceeds credit_usd")
    if sum(v for k, v in b["job_caps_usd"].items() if k != "probe") < 0:
        problems.append("budget: negative caps")
    return problems
