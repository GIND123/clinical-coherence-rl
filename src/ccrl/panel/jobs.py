"""Plan and run the paid panel jobs defined in config/jobs.yaml."""
from __future__ import annotations

from ccrl import config
from ccrl.bedrock import models as M
from ccrl.bedrock.ledger import Ledger
from ccrl.bedrock.runner import BudgetGuard, Task, preflight_time, run_tasks, task_id
from ccrl.panel import items as I
from ccrl.panel.freeze import check_freeze
from ccrl.util import local_now, parse_local
from ccrl.workspace import Workspace

PRIORITY_RANK = {"A": 0, "B": 1, "C": 2}


def job_transforms(job: dict) -> list[str]:
    ts = job["transform_set"]
    if ts == "sequential":
        return ["seqA", "seqB"]
    return list(config.study()["transform_sets"][ts])


def plan_tasks(job_name: str, ws: Workspace, specs: dict, models_filter: list[str] | None = None,
               limit_cases: int | None = None) -> list[Task]:
    job = config.jobs()[job_name]
    items = I.load_items(ws)
    sets = I.load_sets(ws)
    systems = I.load_systems(ws)
    wanted = set(job_transforms(job))
    item_ids = [i for i in sets[job["item_set"]] if items[i]["transform"] in wanted]
    if limit_cases:
        keep: list[str] = []
        for i in item_ids:
            cid = items[i]["case_id"]
            if cid not in keep:
                keep.append(cid)
        # take an equal number of DDXPlus and MedCaseReasoning cases where both exist
        ddx = [c for c in keep if c.startswith("ddx")][: (limit_cases + 1) // 2]
        mcr = [c for c in keep if c.startswith("mcr")][: limit_cases // 2]
        chosen = set(ddx + mcr) if mcr else set(keep[:limit_cases])
        item_ids = [i for i in item_ids if items[i]["case_id"] in chosen]
    case_order = {}
    for i in item_ids:
        case_order.setdefault(items[i]["case_id"], len(case_order))
    tasks: list[Task] = []
    model_keys = config.job_models(job_name)
    if models_filter:
        model_keys = [m for m in model_keys if m in models_filter]
    if job.get("skip_unavailable"):
        model_keys = [m for m in model_keys if m in specs]
        if not model_keys:
            raise SystemExit(f"{job_name}: none of its models has a passing probe "
                             f"({config.job_models(job_name)}); nothing to run.")
    for mk in model_keys:
        if mk not in specs:
            raise SystemExit(f"model {mk} is not in the lock (or failed its probe); run `ccrl models probe`.")
        spec = specs[mk]
        if any(e != "default" for e in job.get("efforts", ["default"])) and spec.effort_strategy == "none":
            raise SystemExit(f"{job_name}: {mk} has no working reasoning-effort control in the lock "
                             "(the probe found it ignored or rejected), so its effort levels would be "
                             "identical requests. Not run.")
        for effort in job.get("efforts", ["default"]):
            for iid in item_ids:
                it = items[iid]
                system = systems[it["system"]]
                cands = I.candidates_of(it)
                rid = task_id(job_name, iid, mk, spec.model_id, effort, 0, system, it["turns"])
                tasks.append(Task(
                    request_id=rid, job=job_name, priority=job["priority"], model_key=mk, item_id=iid,
                    case_id=it["case_id"], transform=it["transform"], dataset=it["dataset"], effort=effort,
                    system=system, turns=it["turns"], kind=it["kind"], codes=[c for c, _ in cands],
                    names=dict(cands), order_key=(case_order[it["case_id"]],)))
    return tasks


def job_complete(ledger: Ledger, job_name: str) -> tuple[bool, int, int]:
    rows = ledger.q("SELECT status, COUNT(*) AS n FROM requests WHERE job=? GROUP BY status", (job_name,))
    total = sum(r["n"] for r in rows)
    final = sum(r["n"] for r in rows if r["status"] in ("done", "refused"))
    return (total > 0 and final == total), final, total


def check_preconditions(job_name: str, ws: Workspace, ledger: Ledger, force_order: bool, allow_c: bool) -> None:
    probs = config.validate_all()
    if probs:
        raise SystemExit("Config problems:\n  " + "\n  ".join(probs))
    if not ws.dry_run:
        preflight_time()
    ok, msg = check_freeze(ws)
    if not ok:
        raise SystemExit(f"Freeze check failed: {msg}\nRun `ccrl freeze` after reviewing the change.")
    job = config.jobs()[job_name]
    if not force_order:
        for dep in job.get("requires_completed", []) or []:
            done, final, total = job_complete(ledger, dep)
            if not done:
                raise SystemExit(f"{job_name} requires {dep} to be complete ({final}/{total} final). "
                                 "Finish it first or pass --force-order.")
    if job["priority"].startswith("C"):
        b = config.budget()
        gate = float(b["priority_c_gate_usd"])
        spent = ledger.committed()
        if spent > gate and not allow_c:
            raise SystemExit(f"Priority {job['priority']} gate: committed ${spent:.2f} > ${gate:.2f}. "
                             "Not launched (pass --override-c-gate only with the PI's approval).")
        dec = config.study()["dates"].get("priority_c_decision")
        dd = parse_local(dec + "T00:00:00") if dec else None
        if dd is not None and local_now() < dd and not allow_c:
            raise SystemExit(f"Priority C jobs wait for the decision date {dd.date()} (or --override-c-gate).")


def run_job(job_name: str, ws: Workspace, models_filter: list[str] | None = None, limit_cases: int | None = None,
            max_spend: float | None = None, force_order: bool = False, allow_c: bool = False,
            profile: str | None = None) -> dict:
    if job_name not in config.jobs():
        raise SystemExit(f"unknown job {job_name!r}; known: {sorted(config.jobs())}")
    ledger = Ledger(ws.ledger)
    try:
        check_preconditions(job_name, ws, ledger, force_order, allow_c)
        specs = M.load_lock(ws.dry_run)
        tasks = plan_tasks(job_name, ws, specs, models_filter, limit_cases)
        if not tasks:
            raise SystemExit("nothing to run")
        guard = BudgetGuard(ledger, job_name, model_caps=model_caps(job_name, tasks, ws), run_cap=max_spend)
        factory = M.make_client_factory(ws.dry_run, profile)
        clients = {mk: factory(specs[mk]) for mk in {t.model_key for t in tasks}}
        summary = run_tasks(tasks, ledger, clients, specs, ws, guard, label=job_name)
        stamp = local_now().strftime("%Y%m%d-%H%M%S")
        ledger.backup(ws.runs / "backups" / f"ledger-{job_name}-{stamp}.sqlite")
        return summary
    finally:
        ledger.close()


def model_caps(job_name: str, tasks: list[Task], ws: Workspace) -> dict[str, float]:
    """Per-model cap = max(multiplier x expected, high scenario) of the forecast
    (pilot-measured when available), never below $1."""
    from ccrl.panel.forecast import forecast_tasks
    mult = float(config.budget().get("per_model_cap_multiplier", 1.3))
    fc = forecast_tasks(tasks, ws)
    return {mk: max(1.0, mult * v["expected"], v["high"]) for mk, v in fc["by_model"].items()}
