"""Amendment 6 (docs/amendment_6.md): does the instability survive an open-ended format?

The panel asks for a probability over a given candidate list. A reviewer can fairly ask whether
the instability is an artefact of that format. This arm asks the same models, on the SAME
vignettes (identical text, extracted from the frozen panel items), for a free-text differential:
five diagnoses, most likely first, with no candidate list and no probabilities.

Cases: the 75 DDXPlus core cases, the 300 MedCaseReasoning panel cases and every post-freeze real
case (Amendment 3) once its items exist. Renderings: base, retest (identical request), perm1
(finding order) and para (paraphrase), as in the panel.

    ccrl openended plan          forecast ($0)
    ccrl openended run           job `openended` (provider-default decoding, 6 models) and
                                 job `openended_t0` (temperature 0, the 5 models that accept it)
    ccrl openended judge         job `openended_judge`: two judges (GLM-5, DeepSeek V3.2) each map
                                 every free-text diagnosis to the case's candidate codes, or to
                                 X-labels shared within the request; one call per case x model x
                                 decoding, the 4 answers shown in a seeded random order with
                                 neutral ids, so the judge cannot tell which is the base rendering
    ccrl openended analyze       tables under runs/reports/analysis/openended/

Endpoints (per case and model, under each judge; the primary judge for a model is the judge that
is not that model): top-1 flip vs the base rendering for retest / order / paraphrase, top-1 and
top-5 accuracy (mapped code == gold), top-5 Jaccard overlap with the base answer, and the
judges' agreement on the mapped top-1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ccrl import config
from ccrl.bedrock import models as M
from ccrl.bedrock.ledger import Ledger
from ccrl.bedrock.runner import BudgetGuard, Task, preflight_time, run_tasks, task_id
from ccrl.prompts import templates as T
from ccrl.prompts.parse import parse_json_object
from ccrl.util import stable_hash
from ccrl.workspace import Workspace

JOB, JOB_T0, JOB_JUDGE = "openended", "openended_t0", "openended_judge"
TRANSFORMS = ("base", "retest", "perm1", "para")
JUDGES = ("glm-5", "deepseek-v3.2")
N_DX = 5
_VIGNETTE = re.compile(r"Vignette:\n(.*)\n\nGive your answer as the JSON object described above\.\s*$", re.S)
_XLAB = re.compile(r"^X\d+$")


def job_models(job: str) -> list[str]:
    """openended: the six panel models; openended_t0: the five that accept temperature 0
    (Bedrock rejects the temperature field for Kimi K3, as in Amendment 2)."""
    m = config.panel_models()
    return [k for k in m if k != "kimi-k3"] if job == JOB_T0 else m


# ------------------------------------------------------------------ cases
def vignette_of(turn: str) -> str:
    m = _VIGNETTE.search(turn)
    if not m:
        raise ValueError("not a single-turn panel item")
    return m.group(1)


def cases(ws: Workspace) -> list[dict]:
    """DDXPlus core, MedCaseReasoning panel and post-freeze cases, with the 4 renderings' vignettes."""
    from ccrl.panel import items as I
    items, sets = I.load_items(ws), I.load_sets(ws)
    pools = [("core", "ddxplus"), ("panel", "medcase"), ("fresh", "medcase")]
    out, seen = [], set()
    for set_name, ds in pools:
        by_case: dict[str, dict] = {}
        for iid in sets.get(set_name, []):
            it = items[iid]
            if it["dataset"] != ds or it["transform"] not in TRANSFORMS:
                continue
            c = by_case.setdefault(it["case_id"], {"case_id": it["case_id"], "dataset": ds, "source": set_name,
                                                   "gold": it.get("gold"), "candidates": I.candidates_of(it),
                                                   "vignettes": {}})
            c["vignettes"][it["transform"]] = vignette_of(it["turns"][0])
        for c in by_case.values():
            if c["case_id"] not in seen and all(t in c["vignettes"] for t in TRANSFORMS):
                seen.add(c["case_id"])
                out.append(c)
    return out


# ------------------------------------------------------------------ generation
def open_validator(text: str):
    obj = parse_json_object(text, ("differential",))
    if obj is None:
        return False, False, None
    d = obj.get("differential")
    if not isinstance(d, list):
        return False, False, None
    d = [str(x).strip() for x in d if isinstance(x, (str, int, float)) and str(x).strip()]
    if not d:
        return False, False, None
    d = d[:N_DX]
    return True, len(d) == N_DX, json.dumps({"differential": d}, ensure_ascii=False)


def gen_tasks(ws: Workspace, specs: dict, job: str, models: list[str] | None = None,
              limit_cases: int | None = None) -> list[Task]:
    cs = cases(ws)
    if limit_cases:
        cs = _limit(cs, limit_cases)
    temperature = 0.0 if job == JOB_T0 else None
    keys = [k for k in job_models(job) if not models or k in models]
    system = T.load("open_system")
    tasks = []
    for k, c in enumerate(cs):
        for mk in keys:
            if mk not in specs:
                raise SystemExit(f"model {mk} is not in the lock (or failed its probe)")
            spec = specs[mk]
            for ti, tr in enumerate(TRANSFORMS):
                user = T.user_prompt(c["vignettes"][tr], None)
                iid = f"{c['case_id']}::open_{tr}"
                rid = task_id(job, iid, mk, spec.model_id, "default", 0, system, [user], temperature)
                tasks.append(Task(request_id=rid, job=job, priority="B", model_key=mk, item_id=iid,
                                  case_id=c["case_id"], transform=tr, dataset=c["dataset"], effort="default",
                                  system=system, turns=[user], kind="open", validator=open_validator,
                                  order_key=(k, ti), temperature=temperature))
    return tasks


def _limit(cs: list[dict], n: int) -> list[dict]:
    out = []
    for src in ("core", "panel", "fresh"):
        out += [c for c in cs if c["source"] == src][: max(1, n // 3)]
    return out


# ------------------------------------------------------------------ judge
def _answers(ws: Workspace, job: str) -> dict[tuple, dict[str, list[str]]]:
    """(case_id, model_key) -> {transform: differential} for finished, parsed generations."""
    if not ws.ledger.exists():
        return {}
    led = Ledger(ws.ledger)
    try:
        rows = led.q("SELECT r.case_id, r.model_key, r.transform, a.parsed_json FROM requests r JOIN attempts a "
                     "ON a.request_id=r.request_id AND a.attempt=r.final_attempt WHERE r.job=? AND r.status='done' "
                     "AND r.parse_ok=1", (job,))
    finally:
        led.close()
    out: dict[tuple, dict] = {}
    for r in rows:
        try:
            d = json.loads(r["parsed_json"])["differential"]
        except (TypeError, ValueError, KeyError):
            continue
        out.setdefault((r["case_id"], r["model_key"]), {})[r["transform"]] = d
    return out


def answer_ids(case_id: str, model_key: str, job: str) -> dict[str, str]:
    """transform -> neutral id (A-D), in a seeded order that hides the base rendering."""
    order = sorted(TRANSFORMS, key=lambda t: stable_hash("open-judge-order", job, case_id, model_key, t))
    return {t: "ABCD"[i] for i, t in enumerate(order)}


def judge_check(ids: dict[str, list[str]], codes: list[str]):
    allowed = set(codes)

    def check(obj: dict) -> str | None:
        maps = obj.get("maps")
        if not isinstance(maps, dict):
            return "maps must be an object"
        for aid, dx in ids.items():
            got = maps.get(aid)
            if not isinstance(got, list) or len(got) != len(dx):
                return f"maps[{aid}] must list {len(dx)} entries"
            for g in got:
                if not isinstance(g, str) or not (g.strip() in allowed or _XLAB.match(g.strip())):
                    return f"maps[{aid}] entry {g!r} is neither a candidate code nor an X-label"
        return None
    return check


def judge_tasks(ws: Workspace, specs: dict, judges: tuple[str, ...] = JUDGES,
                limit_cases: int | None = None) -> list[Task]:
    from ccrl.prep.common import json_validator
    cs = {c["case_id"]: c for c in cases(ws)}
    if limit_cases:
        cs = {c["case_id"]: c for c in _limit(list(cs.values()), limit_cases)}
    system = T.load("open_judge_system")
    tasks = []
    for gi, gen_job in enumerate((JOB, JOB_T0)):
        for k, ((cid, mk), ans) in enumerate(sorted(_answers(ws, gen_job).items())):
            c = cs.get(cid)
            if c is None or not all(t in ans for t in TRANSFORMS):
                continue
            lab = answer_ids(cid, mk, gen_job)
            ids = {lab[t]: ans[t] for t in TRANSFORMS}
            payload = json.dumps(dict(sorted(ids.items())), ensure_ascii=False, indent=1)
            user = T.fill(T.load("open_judge_user"), candidate_block=T.candidate_block(c["candidates"]),
                          answers_json=payload)
            codes = [x for x, _ in c["candidates"]]
            for jk in judges:
                if jk not in specs:
                    raise SystemExit(f"judge {jk} is not in the lock")
                iid = f"{cid}::{gen_job}::{mk}"
                rid = task_id(JOB_JUDGE, iid, jk, specs[jk].model_id, "default", 0, system, [user])
                tasks.append(Task(request_id=rid, job=JOB_JUDGE, priority="B", model_key=jk, item_id=iid,
                                  case_id=cid, transform=f"{gen_job}:{mk}", dataset=c["dataset"], effort="default",
                                  system=system, turns=[user], kind="json", order_key=(gi, k),
                                  validator=json_validator(("maps",), judge_check(ids, codes))))
    return tasks


# ------------------------------------------------------------------ plan / run
def _remaining(ws: Workspace, tasks: list[Task]) -> list[Task]:
    if not ws.ledger.exists() or not tasks:
        return tasks
    led = Ledger(ws.ledger)
    try:
        st = {}
        for j in {t.job for t in tasks}:
            st.update(led.statuses(j))
    finally:
        led.close()
    return [t for t in tasks if st.get(t.request_id) not in ("done", "refused")]


def plan(ws: Workspace, models: list[str] | None = None, limit_cases: int | None = None) -> dict:
    from ccrl.panel.forecast import forecast_tasks
    specs = M.load_lock(ws.dry_run)
    n_cases = len(cases(ws))
    out = {}
    for job in (JOB, JOB_T0):
        tasks = gen_tasks(ws, specs, job, models, limit_cases)
        rem = _remaining(ws, tasks)
        fc = forecast_tasks(rem, ws, specs)["total"] if rem else {"low": 0, "expected": 0, "high": 0}
        cap = float(config.budget()["job_caps_usd"].get(job, 0))
        print(f"[{job}] {n_cases} cases, {len(tasks)} calls, {len(rem)} remaining; expected ${fc['expected']:.2f} "
              f"(high ${fc['high']:.2f}); job cap ${cap:.2f}")
        out[job] = fc
    n_judge = len(JUDGES) * n_cases * (len(job_models(JOB)) + len(job_models(JOB_T0)))
    print(f"[{JOB_JUDGE}] after generation: about {n_judge} calls (one per case x model x decoding x judge); "
          f"job cap ${float(config.budget()['job_caps_usd'].get(JOB_JUDGE, 0)):.2f}")
    return out


def _run(ws: Workspace, tasks: list[Task], job: str, max_spend: float | None, profile: str | None) -> dict:
    from ccrl.panel.freeze import check_freeze
    from ccrl.panel.jobs import backup_ledger, check_disk
    probs = config.validate_all()
    if probs:
        raise SystemExit("Config problems:\n  " + "\n  ".join(probs))
    ok, msg = check_freeze(ws)
    if not ok:
        raise SystemExit(f"Freeze check failed: {msg}\nThe open-ended prompts are part of the freeze (Amendment 6).")
    if not ws.dry_run:
        preflight_time()
    check_disk(ws)
    if not tasks:
        print(f"[{job}] nothing to send")
        return {"ran": 0}
    specs = M.load_lock(ws.dry_run)
    led = Ledger(ws.ledger)
    try:
        guard = BudgetGuard(led, job, run_cap=max_spend)
        factory = M.make_client_factory(ws.dry_run, profile)
        clients = {mk: factory(specs[mk]) for mk in {t.model_key for t in tasks}}
        summary = run_tasks(tasks, led, clients, specs, ws, guard, label=job)
        if summary.get("ran"):
            backup_ledger(ws, led, job)
        return summary
    finally:
        led.close()


def run(ws: Workspace, models: list[str] | None = None, limit_cases: int | None = None,
        max_spend: float | None = None, profile: str | None = None) -> dict:
    specs = M.load_lock(ws.dry_run)
    out = {}
    for job in (JOB, JOB_T0):
        print(f"[{job}] {len(cases(ws))} cases")
        out[job] = _run(ws, gen_tasks(ws, specs, job, models, limit_cases), job, max_spend, profile)
    return out


def judge(ws: Workspace, limit_cases: int | None = None, max_spend: float | None = None,
          profile: str | None = None) -> dict:
    specs = M.load_lock(ws.dry_run)
    tasks = judge_tasks(ws, specs, limit_cases=limit_cases)
    print(f"[{JOB_JUDGE}] {len(tasks)} judge calls")
    return _run(ws, tasks, JOB_JUDGE, max_spend, profile)


# ------------------------------------------------------------------ analysis
def _judged(ws: Workspace) -> pd.DataFrame:
    led = Ledger(ws.ledger)
    try:
        rows = led.q("SELECT r.item_id, r.model_key, a.parsed_json FROM requests r JOIN attempts a "
                     "ON a.request_id=r.request_id AND a.attempt=r.final_attempt WHERE r.job=? AND r.status='done' "
                     "AND r.parse_ok=1", (JOB_JUDGE,))
    finally:
        led.close()
    out = []
    for r in rows:
        cid, gen_job, mk = r["item_id"].split("::")
        maps = json.loads(r["parsed_json"])["maps"]
        lab = answer_ids(cid, mk, gen_job)
        for t in TRANSFORMS:
            out.append({"case_id": cid, "job": gen_job, "model_key": mk, "judge": r["model_key"], "transform": t,
                        "mapped": [m.strip() for m in maps[lab[t]]]})
    return pd.DataFrame(out)


def per_case(ws: Workspace) -> pd.DataFrame:
    """One row per case x model x decoding x judge."""
    cs = {c["case_id"]: c for c in cases(ws)}
    j = _judged(ws)
    rows = []
    for (cid, job, mk, jk), g in j.groupby(["case_id", "job", "model_key", "judge"]):
        v = dict(zip(g["transform"], g["mapped"]))
        if "base" not in v:
            continue
        c = cs[cid]
        base = v["base"]
        row = {"case_id": cid, "dataset": c["dataset"], "source": c["source"], "job": job, "model_key": mk,
               "judge": jk, "primary_judge": jk != mk if mk in JUDGES else jk == JUDGES[0],
               "top1": float(base[0] == c["gold"]), "top5": float(c["gold"] in base),
               "base_top1": base[0]}
        for t in ("retest", "perm1", "para"):
            if t in v:
                row[f"flip_{t}"] = float(v[t][0] != base[0])
                a, b = set(v[t]), set(base)
                row[f"jacc_{t}"] = len(a & b) / len(a | b) if a | b else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(ws: Workspace) -> Path:
    from ccrl.analysis.stats import cluster_bootstrap
    out = ws.reports / "analysis" / "openended"
    out.mkdir(parents=True, exist_ok=True)
    pc = per_case(ws)
    pc.to_csv(out / "per_case.csv", index=False)
    rows = []
    prim = pc[pc["primary_judge"]]
    for (ds, src, job, mk), g in prim.groupby(["dataset", "source", "job", "model_key"]):
        r = {"dataset": ds, "source": src, "job": job, "model_key": mk, "n_cases": len(g)}
        cid = g["case_id"].to_numpy()
        for col in ("top1", "top5", "flip_retest", "flip_perm1", "flip_para", "jacc_retest"):
            if col in g:
                x = g[col].to_numpy(float)
                p, lo, hi = cluster_bootstrap(cid, lambda i, x=x: float(np.nanmean(x[i])), 2000, 0)
                r.update({col: p, f"{col}_lo": lo, f"{col}_hi": hi})
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    # judge agreement on the mapped top-1 of each answer
    j = _judged(ws)
    if not j.empty:
        j["top1"] = j["mapped"].map(lambda m: m[0] if m else None)
        w = j.pivot_table(index=["case_id", "job", "model_key", "transform"], columns="judge", values="top1",
                          aggfunc="first")
        if all(x in w for x in JUDGES):
            w = w.dropna()
            agree = float((w[JUDGES[0]] == w[JUDGES[1]]).mean()) if len(w) else np.nan
            pd.DataFrame([{"n_answers": len(w), "top1_agreement": agree}]).to_csv(out / "judge_agreement.csv",
                                                                                index=False)
    return out
