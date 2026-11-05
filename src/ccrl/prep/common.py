"""Shared helpers for the paid preprocessing jobs."""
from __future__ import annotations

import json

from ccrl import config
from ccrl.bedrock import models as M
from ccrl.bedrock.ledger import Ledger
from ccrl.bedrock.runner import BudgetGuard, Task, preflight_time, run_tasks
from ccrl.prompts.parse import parse_json_object
from ccrl.workspace import Workspace


def json_validator(required: tuple[str, ...], check=None):
    """Validator for JSON-returning prep tasks: (ok, informative, parsed_json)."""
    def v(text: str):
        obj = parse_json_object(text, required)
        if obj is None:
            return False, False, None
        if check is not None:
            err = check(obj)
            if err:
                return False, False, json.dumps({"_error": err, "_obj": obj}, ensure_ascii=False)
        return True, True, json.dumps(obj, ensure_ascii=False)
    return v


def outputs(ws: Workspace, job: str, model_key: str | None = None) -> dict[str, dict]:
    """item_id -> parsed object of the FINAL attempt of each finished request."""
    if not ws.ledger.exists():
        return {}
    led = Ledger(ws.ledger)
    try:
        sql = ("SELECT r.item_id, r.model_key, r.parse_ok, a.parsed_json, a.turn FROM requests r JOIN attempts a "
               "ON a.request_id=r.request_id AND a.attempt=r.final_attempt WHERE r.job=? AND r.status='done'")
        args: tuple = (job,)
        if model_key:
            sql += " AND r.model_key=?"
            args = (job, model_key)
        rows = led.q(sql, args)
    finally:
        led.close()
    out = {}
    for r in rows:
        if r["parse_ok"] and r["parsed_json"]:
            obj = json.loads(r["parsed_json"])
            if "_error" not in obj:
                out[r["item_id"]] = obj
    return out


def run_prep(tasks: list[Task], ws: Workspace, job: str, profile: str | None = None,
             max_spend: float | None = None) -> dict:
    if not tasks:
        print(f"[{job}] nothing to do")
        return {}
    probs = config.validate_all()
    if probs:
        raise SystemExit("Config problems:\n  " + "\n  ".join(probs))
    if not ws.dry_run:
        preflight_time()
    specs = M.load_lock(ws.dry_run)
    led = Ledger(ws.ledger)
    try:
        guard = BudgetGuard(led, job, run_cap=max_spend)
        factory = M.make_client_factory(ws.dry_run, profile)
        clients = {mk: factory(specs[mk]) for mk in {t.model_key for t in tasks}}
        return run_tasks(tasks, led, clients, specs, ws, guard, label=job)
    finally:
        led.close()


def role_spec(ws: Workspace, role: str):
    key = config.role_model(role)
    specs = M.load_lock(ws.dry_run)
    if key not in specs:
        raise SystemExit(f"{role} model {key} is not in the lock (or failed its probe); run `ccrl models probe`.")
    return key, specs[key]
