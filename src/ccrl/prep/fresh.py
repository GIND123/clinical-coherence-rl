"""Amendment 3 (docs/amendment_3.md): a real-case replication on case reports published after
every panel model was frozen (7-27 Nov 2026; selection in ccrl/data/fresh.py).

    ccrl fresh plan         forecast ($0)
    ccrl fresh run          all stages, each resumable (re-running never re-sends a finished call):
      1. cut        Kimi K3 marks where the case presentation must stop before the diagnosis is
                    revealed (by sentence number: the vignette is the authors' own sentences)
      2. screen     deterministic: one patient, a diagnostic case, diagnosis not named before the
                    cut (lexical leak check), >= 4 sentences and >= 400 characters
      3. extract    Kimi K3, the same prompt as the MedCaseReasoning panel (findings, distractors)
      4. verify     GLM-5, the same prompt as the MedCaseReasoning panel
      5. finalize   the same deterministic rules as the panel (ccrl.prep.medcase.finalize_case);
                    the first 150 passing cases in the frozen order are used, later ones are reserve
      6. build      items with the panel's renderer (10 calls per case); their SHA-256 is written to
                    artifacts/frozen/fresh/FREEZE_fresh.json BEFORE any panel call
      7. panel      job `fresh_panel`: 10 calls x 6 models per case, provider default decoding

Outputs live under artifacts/frozen/fresh/ (not part of the main freeze, so nothing the earlier
jobs ran under changes).
"""
from __future__ import annotations

import json
import re

from ccrl import config, paths
from ccrl.bedrock.runner import Task, task_id
from ccrl.prep import medcase as MC
from ccrl.prep.common import json_validator, outputs, role_spec, run_prep
from ccrl.prompts import templates as T
from ccrl.util import now_iso, read_jsonl, sha256_file, write_json, write_jsonl
from ccrl.workspace import Workspace

JOB_CUT, JOB_EXTRACT, JOB_VERIFY, PANEL_JOB = "prep_fresh_cut", "prep_fresh_extract", "prep_fresh_verify", "fresh_panel"
N_USE = None                  # all passing cases (up to the 238 selected); was 150 (changed 27 Nov 2026, before any call)
MIN_USE = 60                  # fewer usable cases than this: stop before the panel (not worth the cost)
MIN_SENTENCES, MIN_CHARS = 4, 400


def out_dir(ws: Workspace):
    d = ws.frozen / "fresh"
    d.mkdir(parents=True, exist_ok=True)
    return d


def selected(limit: int | None = None) -> list[dict]:
    rows = sorted(read_jsonl(paths.FROZEN / "fresh" / "cases_selected.jsonl"), key=lambda r: r["prep_order"])
    return rows[:limit] if limit else rows


# ------------------------------------------------------------------ 1. cut
def _cut_check(n: int):
    def check(obj: dict) -> str | None:
        k = obj.get("last_sentence")
        if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k <= n:
            return f"last_sentence must be an integer in 0..{n}"
        if not isinstance(obj.get("final_diagnosis"), str):
            return "final_diagnosis must be a string"
        for b in ("single_patient", "diagnostic_case"):
            if not isinstance(obj.get(b), bool):
                return f"{b} must be true or false"
        return None
    return check


def cut_tasks(ws: Workspace, limit: int | None = None) -> list[Task]:
    key, spec = role_spec(ws, "preprocessor")
    system = T.load("prep_fresh_cut_system")
    tasks = []
    for k, r in enumerate(selected(limit)):
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(r["sentences"]))
        user = T.fill(T.load("prep_fresh_cut_user"), numbered_sentences=numbered)
        rid = task_id(JOB_CUT, r["case_id"], key, spec.model_id, "default", 0, system, [user])
        tasks.append(Task(rid, JOB_CUT, "B", key, r["case_id"], r["case_id"], "cut", "medcase", "default", system, [user],
                          kind="json", order_key=(k,),
                          validator=json_validator(("single_patient", "final_diagnosis", "diagnostic_case",
                                                    "last_sentence"), _cut_check(len(r["sentences"])))))
    return tasks


# ------------------------------------------------------------------ 2. screen
def _tokens(s: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def screen(ws: Workspace, limit: int | None = None) -> list[dict]:
    key, _ = role_spec(ws, "preprocessor")
    cut = outputs(ws, JOB_CUT, key)
    rows = []
    for r in selected(limit):
        o = cut.get(r["case_id"])
        if o is None:
            continue
        k = int(o["last_sentence"])
        dx = str(o.get("final_diagnosis") or "").strip()
        prompt = " ".join(r["sentences"][:k])
        why = []
        if not o.get("single_patient"):
            why.append("more than one patient")
        if not o.get("diagnostic_case"):
            why.append("diagnosis known at presentation")
        if not dx:
            why.append("no final diagnosis")
        if k < MIN_SENTENCES or len(prompt) < MIN_CHARS:
            why.append(f"vignette too short ({k} sentences, {len(prompt)} characters)")
        if dx and MC.leak_words(dx) & _tokens(prompt):
            why.append("diagnosis named in the vignette")
        rows.append({"case_id": r["case_id"], "role": "fresh", "prep_order": r["prep_order"], "pmcid": r["pmcid"],
                     "split": "fresh", "case_prompt": prompt, "final_diagnosis_raw": dx, "cut_last_sentence": k,
                     "n_sentences": len(r["sentences"]), "cut_reason": o.get("reason"), "title": r["title"],
                     "journal": r["journal"], "publication_date": r["publication_date"], "license": r["license"],
                     "article_link": r["article_link"], "status": "rejected" if why else "ok", "reject_reasons": why})
    write_jsonl(out_dir(ws) / "cases_cut.jsonl", rows)
    return rows


# ------------------------------------------------------------------ 3-4. extract, verify
def _usable(ws: Workspace) -> list[dict]:
    p = out_dir(ws) / "cases_cut.jsonl"
    return [r for r in read_jsonl(p) if r["status"] == "ok"] if p.exists() else []


def extract_tasks(ws: Workspace) -> list[Task]:
    key, spec = role_spec(ws, "preprocessor")
    system = T.load("prep_mcr_extract_system")
    tasks = []
    for k, r in enumerate(_usable(ws)):
        user = T.fill(T.load("prep_mcr_extract_user"), case_prompt=r["case_prompt"],
                      final_diagnosis=r["final_diagnosis_raw"])
        rid = task_id(JOB_EXTRACT, r["case_id"], key, spec.model_id, "default", 0, system, [user])
        tasks.append(Task(rid, JOB_EXTRACT, "B", key, r["case_id"], r["case_id"], "extract", "medcase", "default",
                          system, [user], kind="json", order_key=(k,),
                          validator=json_validator(("findings", "distractors", "final_diagnosis"), MC._extract_check)))
    return tasks


def verify_tasks(ws: Workspace) -> list[Task]:
    key, spec = role_spec(ws, "verifier")
    pre_key, _ = role_spec(ws, "preprocessor")
    ex = outputs(ws, JOB_EXTRACT, pre_key)
    system = T.load("prep_mcr_verify_system")
    tasks = []
    for k, r in enumerate(_usable(ws)):
        e = ex.get(r["case_id"])
        if not e:
            continue
        user = T.fill(T.load("prep_mcr_verify_user"), case_prompt=r["case_prompt"],
                      final_diagnosis_raw=r["final_diagnosis_raw"],
                      extraction_json=json.dumps(MC._numbered(e), ensure_ascii=False, indent=1))
        rid = task_id(JOB_VERIFY, r["case_id"], key, spec.model_id, "default", 0, system, [user])
        tasks.append(Task(rid, JOB_VERIFY, "B", key, r["case_id"], r["case_id"], "verify", "medcase", "default",
                          system, [user], kind="json", order_key=(k,),
                          validator=json_validator(("unsupported", "final_diagnosis_ok"), MC._verify_check)))
    return tasks


# ------------------------------------------------------------------ 5. finalize
def finalize(ws: Workspace) -> dict:
    pre_key, _ = role_spec(ws, "preprocessor")
    ver_key, _ = role_spec(ws, "verifier")
    ex, ver = outputs(ws, JOB_EXTRACT, pre_key), outputs(ws, JOB_VERIFY, ver_key)
    seed = int(config.study()["seed"])
    k_cand = int(config.study()["medcase"]["k_candidates"])
    out, n_ok = [], 0
    for r in _usable(ws):
        e, v = ex.get(r["case_id"]), ver.get(r["case_id"])
        if e is None or v is None:
            continue
        rec = MC.finalize_case(r, e, v, seed, k_cand)
        rec.update({k: r[k] for k in ("title", "journal", "publication_date", "license", "article_link")})
        if rec["status"] == "ok":
            n_ok += 1
            if N_USE is not None and n_ok > N_USE:
                rec["status"] = "reserve"
        out.append(rec)
    write_jsonl(out_dir(ws) / "prepared_fresh.jsonl", out)
    counts: dict[str, int] = {}
    for rec in out:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    return counts


# ------------------------------------------------------------------ 6. build + record
def build_items(ws: Workspace) -> dict:
    from ccrl.panel import items as I
    from ccrl.panel.cases import mcr_prepared

    st = config.study()
    seed = int(st["seed"])
    systems = I._system_texts()
    rows = [r for r in read_jsonl(out_dir(ws) / "prepared_fresh.jsonl") if r["status"] == "ok"]
    items, index, ids = [], [], []
    for r in sorted(rows, key=lambda x: x["prep_order"]):
        case = mcr_prepared(r)
        its = I.items_for_case(case, list(st["transform_sets"]["all10"]), seed, systems)
        items.extend(its)
        ids.extend(it["item_id"] for it in its)
        index.append({"case_id": case.case_id, "dataset": case.dataset, "role": "fresh", "flags": case.flags,
                      "gold": case.gold, "codes": case.codes, "n_findings": len(case.findings),
                      "final_diagnosis": r["final_diagnosis"], "pmcid": r["pmcid"],
                      "publication_date": r["publication_date"], "candidates": [list(x) for x in case.candidates]})
    d = out_dir(ws) / "items"
    d.mkdir(parents=True, exist_ok=True)
    write_jsonl(d / "items.jsonl", items)
    write_json(d / "sets.json", {"fresh": ids})
    write_jsonl(d / "case_index.jsonl", index)
    return {"cases": len(index), "items": len(items)}


def record_freeze(ws: Workspace) -> dict:
    from ccrl.panel.freeze import git_commit
    d = out_dir(ws)
    files = {p.relative_to(d).as_posix(): sha256_file(p)
             for p in sorted(d.rglob("*")) if p.is_file() and p.name != "FREEZE_fresh.json"}
    files["config/prompts/prep_fresh_cut_system.txt"] = sha256_file(paths.PROMPTS / "prep_fresh_cut_system.txt")
    files["config/prompts/prep_fresh_cut_user.txt"] = sha256_file(paths.PROMPTS / "prep_fresh_cut_user.txt")
    f = d / "FREEZE_fresh.json"
    if f.exists():
        prev = json.loads(f.read_text(encoding="utf-8"))
        if prev.get("files") == files:
            return prev
        if (ws.ledger.exists() and _panel_started(ws)):
            raise SystemExit("fresh items changed after fresh_panel calls were made; refusing to re-record")
    doc = {"created_at": now_iso(), "git_commit": git_commit(), "dry_run": ws.dry_run, "files": files}
    write_json(f, doc)
    return doc


def _panel_started(ws: Workspace) -> bool:
    from ccrl.bedrock.ledger import Ledger
    led = Ledger(ws.ledger)
    try:
        return bool(led.q("SELECT 1 FROM requests WHERE job=? LIMIT 1", (PANEL_JOB,)))
    finally:
        led.close()


# ------------------------------------------------------------------ plan / run
def plan(ws: Workspace, limit: int | None = None) -> None:
    from ccrl.bedrock import models as M
    from ccrl.panel.forecast import forecast_tasks
    specs = M.load_lock(ws.dry_run)
    fc = forecast_tasks(cut_tasks(ws, limit), ws, specs)["total"]
    n = len(selected(limit))
    print(f"[fresh] {n} selected case reports (published 7-27 Nov 2026)")
    print(f"  1. cut (Kimi K3): {n} calls, expected ${fc['expected']:.2f} (high ${fc['high']:.2f})")
    print("  3-4. extraction (Kimi K3) + verification (GLM-5): ~$0.07 per usable case")
    n_use = n if N_USE is None else N_USE
    print(f"  7. panel: every passing case (at most {n_use}) x 10 calls x 6 models; ~$0.40 per case (Kimi K3 ~95%)")
    print(f"  expected total about ${fc['expected'] + 0.07 * n * 0.9 + 0.40 * n_use * 0.7:.0f} "
          f"(if ~70% pass; at most ${fc['high'] + 0.07 * n + 0.40 * n_use:.0f})")


def run(ws: Workspace, profile: str | None = None, max_spend: float | None = None, limit: int | None = None) -> dict:
    from ccrl.panel import jobs
    jobs.check_disk(ws)
    run_prep(cut_tasks(ws, limit), ws, JOB_CUT, profile, max_spend)
    rows = screen(ws, limit)
    print(f"[fresh] screen: {sum(r['status'] == 'ok' for r in rows)} usable of {len(rows)}")
    run_prep(extract_tasks(ws), ws, JOB_EXTRACT, profile, max_spend)
    run_prep(verify_tasks(ws), ws, JOB_VERIFY, profile, max_spend)
    counts = finalize(ws)
    print(f"[fresh] finalize: {counts}")
    n_ok = counts.get("ok", 0)
    if n_ok < MIN_USE and not (ws.dry_run or limit):
        raise SystemExit(f"only {n_ok} usable cases (< {MIN_USE}); stopping before the panel. Send the archive.")
    print(f"[fresh] items: {build_items(ws)}")
    print(f"[fresh] recorded {len(record_freeze(ws)['files'])} files in FREEZE_fresh.json")
    return jobs.run_job(PANEL_JOB, ws, profile=profile, max_spend=max_spend)
