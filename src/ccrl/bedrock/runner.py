"""Concurrent, idempotent, budget-guarded execution of paid tasks.

A Task is one logical request (possibly two sequential turns). The executor:

  * skips tasks whose ledger status is final (done / refused);
  * admits a call only if the BudgetGuard accepts its worst-case reservation
    — the check and the ledger reservation happen under one lock, so
    concurrent workers can never jointly overshoot a cap;
  * runs per-model concurrency limits, case-major ordering (a partial run
    leaves complete 10-call families), round-robin across models (a budget
    stop leaves every model equally far along);
  * applies a bounded retry policy whose every attempt is recorded and billed;
  * mirrors every attempt to runs/raw/<job>/<model>.jsonl (append + fsync) as
    a second copy independent of SQLite;
  * stops cleanly on Ctrl-C (in-flight calls finish and are recorded).
"""
from __future__ import annotations

import json
import signal
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable

from ccrl import config
from ccrl.bedrock import pricing
from ccrl.bedrock.client import CallResult, CallSpec, build_request, request_json, reservation_for
from ccrl.bedrock.ledger import Ledger
from ccrl.prompts.parse import parse_posterior
from ccrl.util import append_jsonl, local_now, now_iso, stable_hash
from ccrl.workspace import Workspace

MAX_PARSE_RETRIES = 1
MAX_TRUNCATION_RETRIES = 1
MAX_UNCERTAIN_RETRIES = 1
MAX_UNBILLED_RETRIES = 2
MAX_TOKENS_CEILING = 65536


@dataclass
class Task:
    request_id: str
    job: str
    priority: str
    model_key: str
    item_id: str
    case_id: str
    transform: str
    dataset: str
    effort: str
    system: str
    turns: list[str]
    kind: str = "posterior"                       # posterior | scores | json
    codes: list[str] | None = None
    names: dict | None = None
    validator: Callable[[str], tuple[bool, bool, str | None]] | None = None
    replicate: int = 0
    order_key: tuple = field(default_factory=tuple)

    def validate(self, text: str) -> tuple[bool, bool, str | None]:
        """(parse_ok, informative, parsed_json)."""
        if self.validator is not None:
            return self.validator(text)
        res = parse_posterior(text, self.codes or [], key="s" if self.kind == "scores" else "p", names=self.names)
        return res.ok, res.informative, json.dumps(res.to_json())


def task_id(job: str, item_id: str, model_key: str, model_id: str, effort: str, replicate: int,
            system: str, turns: list[str]) -> str:
    return stable_hash(job, item_id, model_key, model_id, effort, replicate,
                       stable_hash(system, *turns, n=64), n=24)


# ------------------------------------------------------------------ budget
class BudgetStop(Exception):
    pass


class BudgetGuard:
    def __init__(self, ledger: Ledger, job: str, model_caps: dict[str, float] | None = None,
                 job_cap: float | None = None, hard_cap: float | None = None, run_cap: float | None = None):
        b = config.budget()
        self.ledger = ledger
        self.job = job
        self.lock = threading.Lock()
        self.model_caps = model_caps or {}
        self.job_cap = job_cap if job_cap is not None else float(b["job_caps_usd"].get(job, 0.0))
        self.hard_cap = hard_cap if hard_cap is not None else float(b["hard_cap_usd"])
        self.run_cap = run_cap
        self.run_start = ledger.committed()
        self.alerts = sorted(float(x) for x in b.get("alert_levels_usd", []))
        self._alerted: set[float] = {a for a in self.alerts if self.run_start >= a}

    def check(self, model_key: str, reservation: float) -> tuple[bool, str]:
        total = self.ledger.committed()
        if total + reservation > self.hard_cap:
            return False, f"hard cap ${self.hard_cap:.2f} (committed ${total:.2f})"
        if self.run_cap is not None and (total - self.run_start) + reservation > self.run_cap:
            return False, f"--max-spend ${self.run_cap:.2f} for this run"
        if self.job_cap:
            j = self.ledger.committed(job=self.job)
            if j + reservation > self.job_cap:
                return False, f"job cap ${self.job_cap:.2f} for {self.job} (committed ${j:.2f})"
        cap = self.model_caps.get(model_key)
        if cap:
            m = self.ledger.committed(job=self.job, model_key=model_key)
            if m + reservation > cap:
                return False, f"model cap ${cap:.2f} for {model_key} in {self.job} (committed ${m:.2f})"
        return True, ""

    def maybe_alert(self) -> str | None:
        total = self.ledger.committed()
        for a in self.alerts:
            if total >= a and a not in self._alerted:
                self._alerted.add(a)
                return f"ALERT: committed spend crossed ${a:.0f} (now ${total:.2f})"
        return None


# ------------------------------------------------------------------ one call
@dataclass
class Executor:
    ledger: Ledger
    clients: dict[str, object]          # model_key -> client with .converse(req, tier)
    specs: dict[str, CallSpec]
    ws: Workspace
    guard: BudgetGuard

    def call(self, task: Task, attempt: int, turn: int, messages: list[dict],
             max_tokens: int | None = None) -> CallResult | None:
        spec = self.specs[task.model_key]
        req = build_request(spec, task.system, messages, task.effort, max_tokens=max_tokens)
        res_cost = reservation_for(req, spec.price_key)
        with self.guard.lock:
            ok, why = self.guard.check(task.model_key, res_cost)
            if not ok:
                raise BudgetStop(why)
            self.ledger.open_attempt(task.request_id, attempt, turn, res_cost, request_json(req))
        res: CallResult = self.clients[task.model_key].converse(req, spec.service_tier)
        rep = gc = 0.0
        u = pricing.usage_tokens(res.usage) if res.usage else {}
        if res.billed:
            rep, gc = pricing.cost(res.usage, spec.price_key, res.served_tier)
        self.ledger.close_attempt(
            task.request_id, attempt, turn, status=res.status, cost_reported=rep, cost_guard=gc,
            input_tokens=u.get("input_reported"), output_tokens=u.get("output"), cache_read=u.get("cache_read"),
            cache_write=u.get("cache_write"), usage_semantics=u.get("semantics"), stop_reason=res.stop_reason,
            served_tier=res.served_tier, latency_ms=res.latency_ms, throttles=res.throttles,
            error_code=res.error_code, error_msg=res.error_msg,
            response_json=json.dumps(res.response, ensure_ascii=False) if res.response else None,
            text=res.text, reasoning=res.reasoning)
        append_jsonl(self.ws.raw / task.job / f"{task.model_key}.jsonl", {
            "ts": now_iso(), "request_id": task.request_id, "attempt": attempt, "turn": turn,
            "item_id": task.item_id, "model_key": task.model_key, "model_id": spec.model_id,
            "effort": task.effort, "status": res.status, "cost_reported": rep, "request": req,
            "response": res.response, "error_code": res.error_code, "error_msg": res.error_msg})
        return res

    def run_task(self, task: Task, stop: threading.Event, fatal_models: set, fatal_run: list) -> str:
        """Execute one logical task with the retry policy. Returns final status."""
        tries = Counter()
        max_tokens = None
        while not stop.is_set():
            if task.model_key in fatal_models or fatal_run:
                return "skipped"
            attempt = self.ledger.next_attempt_no(task.request_id)
            messages: list[dict] = []
            final: CallResult | None = None
            parsed = (False, False, None)
            for turn, user in enumerate(task.turns, 1):
                messages.append({"role": "user", "text": user})
                res = self.call(task, attempt, turn, messages, max_tokens)
                final = res
                if res.status not in ("ok", "truncated"):
                    break
                parsed = task.validate(res.text)
                self.ledger.close_attempt(task.request_id, attempt, turn, status=res.status, parsed_json=parsed[2])
                if turn < len(task.turns):
                    messages.append({"role": "assistant", "text": res.text if res.text.strip() else "{}"})
            assert final is not None
            st = final.status
            if st == "refused":
                self.ledger.finish_request(task.request_id, "refused", attempt, False, False, final.stop_reason)
                return "refused"
            if st == "unbilled_error":
                if final.fatal == "run":
                    fatal_run.append(f"{task.model_key}: {final.error_code}: {final.error_msg}")
                    stop.set()
                    return "fatal"
                if final.fatal == "model":
                    fatal_models.add(task.model_key)
                    self.ledger.event("model_halted", {"code": final.error_code, "msg": final.error_msg},
                                      task.job, task.model_key)
                    return "fatal"
                tries["unbilled"] += 1
                if tries["unbilled"] > MAX_UNBILLED_RETRIES:
                    self.ledger.finish_request(task.request_id, "failed", attempt, note=final.error_code)
                    return "failed"
                continue
            if st == "uncertain":
                tries["uncertain"] += 1
                if tries["uncertain"] > MAX_UNCERTAIN_RETRIES:
                    self.ledger.finish_request(task.request_id, "failed", attempt, note=final.error_code)
                    return "failed"
                continue
            if st == "truncated":
                tries["trunc"] += 1
                spec = self.specs[task.model_key]
                cur = max_tokens or spec.max_tokens_for(spec.resolve_effort(task.effort))
                if tries["trunc"] <= MAX_TRUNCATION_RETRIES and cur < MAX_TOKENS_CEILING:
                    max_tokens = min(MAX_TOKENS_CEILING, cur * 2)
                    continue
                self.ledger.finish_request(task.request_id, "done", attempt, parsed[0], parsed[1], "truncated")
                return "done"
            # ok
            if not parsed[0]:
                tries["parse"] += 1
                if tries["parse"] <= MAX_PARSE_RETRIES:
                    continue
            self.ledger.finish_request(task.request_id, "done", attempt, parsed[0], parsed[1])
            return "done"
        return "stopped"


# ------------------------------------------------------------------ driver
def preflight_time() -> None:
    hs = config.study()["dates"]["hard_stop"]
    from ccrl.util import parse_local
    if local_now() > parse_local(hs):
        raise SystemExit(f"Hard stop {hs} has passed: no new paid calls (study.yaml dates.hard_stop).")


def run_tasks(tasks: list[Task], ledger: Ledger, clients: dict, specs: dict[str, CallSpec], ws: Workspace,
              guard: BudgetGuard, label: str, progress_every: int = 25, quiet: bool = False) -> dict:
    """Run tasks to completion (or until a cap/stop). Returns a summary dict."""
    ledger.recover()
    ledger.ensure_requests([{
        "request_id": t.request_id, "job": t.job, "priority": t.priority, "model_key": t.model_key,
        "model_id": specs[t.model_key].model_id, "item_id": t.item_id, "case_id": t.case_id,
        "transform": t.transform, "dataset": t.dataset, "effort": t.effort, "replicate": t.replicate}
        for t in tasks])
    status = {r["request_id"]: r["status"] for r in ledger.q(
        "SELECT request_id,status FROM requests WHERE job=?", (tasks[0].job,))} if tasks else {}
    todo = [t for t in tasks if status.get(t.request_id) not in ("done", "refused")]
    # Case-major order within each model (a partial run leaves complete case
    # families); every model has its own worker pool, so a slow model never
    # holds threads another model could use.
    by_model: dict[str, list[Task]] = defaultdict(list)
    for t in sorted(todo, key=lambda t: (t.order_key, t.item_id, t.effort)):
        by_model[t.model_key].append(t)
    ex = Executor(ledger, clients, specs, ws, guard)
    stop = threading.Event()
    fatal_models: set[str] = set()
    fatal_run: list[str] = []
    budget_reason: list[str] = []
    outcome = Counter()
    t0 = time.time()
    done_n = 0
    n_queue = len(todo)
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    say(f"[{label}] {len(tasks)} tasks, {len(tasks) - len(todo)} already final, {n_queue} to run; "
        f"committed so far ${ledger.committed():.2f}")

    def work(t: Task) -> str:
        if stop.is_set():
            return "stopped"
        try:
            return ex.run_task(t, stop, fatal_models, fatal_run)
        except BudgetStop as e:
            budget_reason.append(str(e))
            stop.set()
            return "budget_stop"

    prev = signal.getsignal(signal.SIGINT)

    def _sigint(_s, _f):
        if not stop.is_set():
            print(f"\n[{label}] Ctrl-C: no new calls will start; waiting for in-flight calls to finish "
                  "(press Ctrl-C again to force-quit; forced calls are recorded as uncertain).", flush=True)
            stop.set()
        else:
            signal.signal(signal.SIGINT, prev)
            raise KeyboardInterrupt

    in_main = threading.current_thread() is threading.main_thread()
    if in_main:
        signal.signal(signal.SIGINT, _sigint)
    pools = {mk: ThreadPoolExecutor(max_workers=max(1, specs[mk].concurrency), thread_name_prefix=mk)
             for mk in by_model}
    iters = {mk: iter(v) for mk, v in by_model.items()}
    exhausted: set[str] = set()
    pending: dict = {}
    try:
        while True:
            if not stop.is_set():
                for mk, pool in pools.items():
                    if mk in exhausted:
                        continue
                    if mk in fatal_models:
                        rest = sum(1 for _ in iters[mk])
                        outcome["skipped"] += rest
                        exhausted.add(mk)
                        continue
                    cap = 2 * max(1, specs[mk].concurrency)
                    while sum(1 for v in pending.values() if v == mk) < cap:
                        t = next(iters[mk], None)
                        if t is None:
                            exhausted.add(mk)
                            break
                        pending[pool.submit(work, t)] = mk
            if not pending:
                break
            fin, _ = wait(list(pending), timeout=5, return_when=FIRST_COMPLETED)
            for f in fin:
                pending.pop(f, None)
                outcome[f.result()] += 1
                done_n += 1
                if done_n % progress_every == 0:
                    el = time.time() - t0
                    say(f"[{label}] {done_n}/{n_queue} finished | {dict(outcome)} | "
                        f"committed ${ledger.committed():.2f} (job ${ledger.committed(job=tasks[0].job):.2f}) | "
                        f"{done_n / max(el, 1e-9) * 60:.1f}/min")
                msg = guard.maybe_alert()
                if msg:
                    say(f"[{label}] {msg}")
                    ledger.event("spend_alert", msg, tasks[0].job)
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
        if in_main:
            signal.signal(signal.SIGINT, prev)
    summary = {"label": label, "tasks": len(tasks), "ran": done_n, "outcome": dict(outcome),
               "fatal_models": sorted(fatal_models), "fatal_run": fatal_run, "budget_stop": budget_reason[:1],
               "committed_total": round(ledger.committed(), 4),
               "committed_job": round(ledger.committed(job=tasks[0].job), 4) if tasks else 0.0,
               "reported_job": round(ledger.reported(job=tasks[0].job), 4) if tasks else 0.0,
               "elapsed_s": round(time.time() - t0, 1)}
    ledger.event("run_summary", summary, tasks[0].job if tasks else None)
    say(f"[{label}] finished: {json.dumps(summary)}")
    return summary
