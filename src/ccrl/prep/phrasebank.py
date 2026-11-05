"""DDXPlus phrase bank: one independently worded phrase per evidence code.

    ccrl prep phrasebank          generate (preprocessor) then verify (verifier)
    ccrl prep finalize            choose phrases -> artifacts/frozen/banks/ddx_phrasebank.json

Items are the 223 base templates plus the yes/no and special-value phrases
(id "E_131#Y", "E_57#nowhere", ...). Value meanings are never paraphrased.
A phrase is accepted only if (a) the verifier marks it equivalent and (b) it
passes local checks (placeholder count, no second-person leakage). If no
alternative passes, the base phrase is used and the id is listed under
`fallback_ids` — the paraphrase arm then leaves that finding unchanged, which
weakens but never invalidates the manipulation.
"""
from __future__ import annotations

import csv
import json
import re

from ccrl.data import ddx_render as R
from ccrl.data import ddxplus as dx
from ccrl.bedrock.runner import Task, task_id
from ccrl.prep.common import json_validator, outputs, role_spec, run_prep
from ccrl.prompts import templates as T
from ccrl.util import chunked, now_iso, write_json
from ccrl.workspace import Workspace

JOB = "prep_phrasebank"
BATCH = 25


def phrase_items() -> list[dict]:
    kb = dx.load_kb()
    base = R.all_base_templates(kb)
    items = []
    for code in sorted(kb.evidences, key=lambda c: int(c.split("_")[1])):
        if code in R.YES_NO:
            continue
        items.append({"id": code, "base": base[code], "question": kb.evidences[code].question})
    for code, (yes, no) in sorted(R.YES_NO.items()):
        items.append({"id": f"{code}#Y", "base": yes, "question": kb.evidences[code].question + " (yes)"})
        items.append({"id": f"{code}#N", "base": no, "question": kb.evidences[code].question + " (no)"})
    for (code, val), phrase in sorted(R.SPECIAL_VALUES.items()):
        items.append({"id": f"{code}#{val}", "base": phrase, "question": kb.evidences[code].question + f" ({val})"})
    return items


def _batches() -> list[tuple[str, list[dict]]]:
    return [(f"pb:{k:02d}", b) for k, b in enumerate(chunked(phrase_items(), BATCH))]


def _gen_check(batch: list[dict]):
    ids = [b["id"] for b in batch]

    def check(obj: dict) -> str | None:
        got = obj.get("items")
        if not isinstance(got, list):
            return "items not a list"
        by = {g.get("id"): g for g in got if isinstance(g, dict)}
        for i in ids:
            g = by.get(i)
            if g is None:
                return f"missing id {i}"
            alts = g.get("alternatives")
            if not (isinstance(alts, list) and len(alts) >= 1 and all(isinstance(a, str) for a in alts)):
                return f"bad alternatives for {i}"
        return None
    return check


def _ver_check(batch: list[dict]):
    ids = [b["id"] for b in batch]

    def check(obj: dict) -> str | None:
        got = obj.get("items")
        if not isinstance(got, list):
            return "items not a list"
        by = {g.get("id"): g for g in got if isinstance(g, dict)}
        for i in ids:
            if i not in by or not isinstance(by[i].get("ok"), list):
                return f"missing verdict for {i}"
        return None
    return check


def local_ok(base: str, alt: str) -> bool:
    if not alt or len(alt) > 3 * len(base) + 60:
        return False
    if base.count("{value}") != alt.count("{value}"):
        return False
    if re.search(r"\b(you|your|yourself)\b", alt, re.I):
        return False
    if alt.strip().lower().startswith("the patient"):
        return False
    return alt.strip().rstrip(".") != base.strip()


def generation_tasks(ws: Workspace) -> list[Task]:
    key, spec = role_spec(ws, "preprocessor")
    system = T.load("prep_phrasebank_system")
    tasks = []
    for k, (bid, batch) in enumerate(_batches()):
        payload = json.dumps([{"id": b["id"], "base": b["base"]} for b in batch], ensure_ascii=False, indent=1)
        user = T.fill(T.load("prep_phrasebank_user"), items_json=payload)
        rid = task_id(JOB, f"{bid}:gen", key, spec.model_id, "default", 0, system, [user])
        tasks.append(Task(rid, JOB, "A", key, f"{bid}:gen", bid, "gen", "ddxplus", "default", system, [user],
                          kind="json", validator=json_validator(("items",), _gen_check(batch)), order_key=(k,)))
    return tasks


def verification_tasks(ws: Workspace) -> list[Task]:
    key, spec = role_spec(ws, "verifier")
    pre_key, _ = role_spec(ws, "preprocessor")
    gen = outputs(ws, JOB, pre_key)
    system = T.load("prep_phrasebank_verify_system")
    tasks = []
    for k, (bid, batch) in enumerate(_batches()):
        g = gen.get(f"{bid}:gen")
        if not g:
            continue
        by = {x["id"]: x for x in g["items"]}
        payload = json.dumps([{"id": b["id"], "original": b["base"],
                               "alternatives": (by[b["id"]]["alternatives"] + ["", "", ""])[:3]} for b in batch],
                             ensure_ascii=False, indent=1)
        user = T.fill(T.load("prep_phrasebank_verify_user"), items_json=payload)
        rid = task_id(JOB, f"{bid}:ver", key, spec.model_id, "default", 0, system, [user])
        tasks.append(Task(rid, JOB, "A", key, f"{bid}:ver", bid, "ver", "ddxplus", "default", system, [user],
                          kind="json", validator=json_validator(("items",), _ver_check(batch)), order_key=(k,)))
    return tasks


def run(ws: Workspace, profile: str | None = None, max_spend: float | None = None) -> None:
    run_prep(generation_tasks(ws), ws, JOB, profile, max_spend)
    run_prep(verification_tasks(ws), ws, JOB, profile, max_spend)


def finalize(ws: Workspace) -> dict | None:
    pre_key, _ = role_spec(ws, "preprocessor")
    ver_key, _ = role_spec(ws, "verifier")
    gen = outputs(ws, JOB, pre_key)
    ver = outputs(ws, JOB, ver_key)
    if not gen:
        print("[phrasebank] no generation outputs yet; skipped")
        return None
    templates, fallback, rows = {}, [], []
    n_batches = 0
    for bid, batch in _batches():
        g, v = gen.get(f"{bid}:gen"), ver.get(f"{bid}:ver")
        if not g or not v:
            fallback.extend(b["id"] for b in batch)
            for b in batch:
                rows.append([b["id"], b["question"], b["base"], b["base"], "", "", "missing batch output"])
            continue
        n_batches += 1
        gby = {x["id"]: x for x in g["items"]}
        vby = {x["id"]: x for x in v["items"]}
        for b in batch:
            alts = (gby[b["id"]]["alternatives"] + ["", "", ""])[:3]
            oks = (list(vby[b["id"]]["ok"]) + [False, False, False])[:3]
            chosen = next((a.strip() for a, ok in zip(alts, oks) if ok is True and local_ok(b["base"], a.strip())), None)
            if chosen is None:
                fallback.append(b["id"])
                chosen = b["base"]
            templates[b["id"]] = chosen.rstrip(".")
            rows.append([b["id"], b["question"], b["base"], chosen, " || ".join(alts),
                         " ".join("Y" if o is True else "N" for o in oks), ""])
    doc = {"created_at": now_iso(), "preprocessor": pre_key, "verifier": ver_key,
           "n_items": len(rows), "n_batches_complete": n_batches, "fallback_ids": sorted(fallback),
           "templates": templates}
    write_json(ws.phrasebank, doc)
    with open(ws.banks / "phrasebank_review.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "base_phrase", "chosen_paraphrase", "alternatives", "verifier_ok",
                    "note", "reviewer_1_ok", "reviewer_2_ok", "reviewer_notes"])
        for r in rows:
            w.writerow(r + ["", "", ""])
    print(f"[phrasebank] {len(templates)} phrases; {len(fallback)} fell back to the base wording")
    return doc
