"""MedCaseReasoning preprocessing: extraction, verification, finalisation.

    ccrl prep mcr --stage extract [--roles pilot,panel]   preprocessor model
    ccrl prep mcr --stage verify                          verifier model (pilot + panel only)
    ccrl prep finalize                                    deterministic -> medcase/prepared.jsonl

Finalisation rules (all deterministic, applied before any panel model is
called, so case inclusion can never depend on a panel answer):
  * drop findings flagged names_diagnosis by the extractor, or
    reveals_diagnosis / unsupported by the verifier;
  * drop findings that contain a distinctive word of the final diagnosis
    (lexical leak check, words >= 5 letters, stop-words excluded);
  * a paraphrase the verifier marks non-equivalent is replaced by the
    original sentence (the paraphrase arm is weakened, never wrong);
  * exactly one chief complaint in stage 0 (first stage-0 finding if the
    extractor marked none or several); no stage-0 finding -> reject;
  * candidates = final diagnosis + the first 9 valid distractors (verifier-
    invalid, duplicate or gold-overlapping removed); panel/pilot cases need
    all 10, training cases at least 6; order shuffled with a per-case seed;
  * 6-24 findings after the drops, else reject;
  * verifier final_diagnosis_ok false -> reject (pilot/panel).
Rejected cases are replaced by the next case in the frozen preprocessing order.
"""
from __future__ import annotations

import csv
import json
import re

from ccrl import config, paths
from ccrl.bedrock.runner import Task, task_id
from ccrl.prep.common import json_validator, outputs, role_spec, run_prep
from ccrl.prompts import templates as T
from ccrl.util import now_iso, py_rng, read_jsonl, two_letter_codes, write_jsonl
from ccrl.workspace import Workspace

JOB_EXTRACT = "prep_mcr_extract"
JOB_VERIFY = "prep_mcr_verify"
STOP = {"syndrome", "disease", "disorder", "acute", "chronic", "primary", "secondary", "induced", "associated",
        "infection", "with", "without", "type", "related", "reaction", "lesion", "tumor", "tumour", "cancer",
        "carcinoma", "benign", "malignant", "left", "right", "bilateral", "multiple", "severe", "mild"}


def selected(roles: list[str] | None = None) -> list[dict]:
    rows = read_jsonl(paths.FROZEN_MCR / "cases_selected.jsonl")
    if roles:
        rows = [r for r in rows if r["role"] in roles]
    return sorted(rows, key=lambda r: (["pilot", "panel", "train"].index(r["role"]), r["prep_order"]))


def _extract_check(obj: dict) -> str | None:
    f = obj.get("findings")
    if not isinstance(f, list) or not f:
        return "findings missing"
    for x in f:
        if not isinstance(x, dict) or not isinstance(x.get("text"), str) or not x["text"].strip():
            return "finding without text"
        if not isinstance(x.get("paraphrase"), str):
            return "finding without paraphrase"
        try:
            s = int(x.get("stage"))
        except (TypeError, ValueError):
            return "stage not an integer"
        if s < 0 or s > 3:
            return "stage out of range"
    if not isinstance(obj.get("distractors"), list) or len(obj["distractors"]) < 6:
        return "fewer than 6 distractors"
    if not isinstance(obj.get("final_diagnosis"), str) or not obj["final_diagnosis"].strip():
        return "final_diagnosis missing"
    if not isinstance(obj.get("patient"), str):
        return "patient missing"
    return None


def _verify_check(obj: dict) -> str | None:
    for k in ("unsupported", "paraphrase_not_equivalent", "reveals_diagnosis", "stage_errors", "distractors_invalid"):
        if not isinstance(obj.get(k), list):
            return f"{k} not a list"
    if not isinstance(obj.get("final_diagnosis_ok"), bool):
        return "final_diagnosis_ok not a bool"
    return None


def extraction_tasks(ws: Workspace, roles: list[str]) -> list[Task]:
    key, spec = role_spec(ws, "preprocessor")
    system = T.load("prep_mcr_extract_system")
    tasks = []
    for k, r in enumerate(selected(roles)):
        user = T.fill(T.load("prep_mcr_extract_user"), case_prompt=r["case_prompt"],
                      final_diagnosis=r["final_diagnosis_raw"])
        rid = task_id(JOB_EXTRACT, r["case_id"], key, spec.model_id, "default", 0, system, [user])
        tasks.append(Task(rid, JOB_EXTRACT, "A", key, r["case_id"], r["case_id"], "extract", "medcase", "default",
                          system, [user], kind="json",
                          validator=json_validator(("findings", "distractors", "final_diagnosis"), _extract_check),
                          order_key=(k,)))
    return tasks


def _numbered(ex: dict) -> dict:
    """The extraction as shown to the verifier: findings with ids f01.."""
    return {"patient": ex.get("patient"), "final_diagnosis": ex.get("final_diagnosis"),
            "findings": [{"id": f"f{i + 1:02d}", "text": f["text"], "paraphrase": f.get("paraphrase", ""),
                          "stage": int(f["stage"]), "chief_complaint": bool(f.get("chief_complaint"))}
                         for i, f in enumerate(ex["findings"])],
            "distractors": list(ex.get("distractors", []))}


def verification_tasks(ws: Workspace) -> list[Task]:
    key, spec = role_spec(ws, "verifier")
    pre_key, _ = role_spec(ws, "preprocessor")
    ex = outputs(ws, JOB_EXTRACT, pre_key)
    system = T.load("prep_mcr_verify_system")
    tasks = []
    for k, r in enumerate(selected(["pilot", "panel"])):
        e = ex.get(r["case_id"])
        if not e:
            continue
        user = T.fill(T.load("prep_mcr_verify_user"), case_prompt=r["case_prompt"],
                      final_diagnosis_raw=r["final_diagnosis_raw"],
                      extraction_json=json.dumps(_numbered(e), ensure_ascii=False, indent=1))
        rid = task_id(JOB_VERIFY, r["case_id"], key, spec.model_id, "default", 0, system, [user])
        tasks.append(Task(rid, JOB_VERIFY, "A", key, r["case_id"], r["case_id"], "verify", "medcase", "default",
                          system, [user], kind="json",
                          validator=json_validator(("unsupported", "final_diagnosis_ok"), _verify_check),
                          order_key=(k,)))
    return tasks


def run(ws: Workspace, stage: str, roles: list[str], profile: str | None = None,
        max_spend: float | None = None) -> None:
    if stage in ("extract", "all"):
        run_prep(extraction_tasks(ws, roles), ws, JOB_EXTRACT, profile, max_spend)
    if stage in ("verify", "all"):
        run_prep(verification_tasks(ws), ws, JOB_VERIFY, profile, max_spend)


# --------------------------------------------------------------------- finalize
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def leak_words(dx_name: str) -> set[str]:
    return {w for w in _norm(dx_name).split() if len(w) >= 5 and w not in STOP}


def finalize_case(sel: dict, ex: dict, ver: dict | None, seed: int, need_k: int) -> dict:
    reasons: list[str] = []
    num = _numbered(ex)
    drop: set[str] = set()
    for f, raw in zip(num["findings"], ex["findings"]):
        if raw.get("names_diagnosis"):
            drop.add(f["id"])
    if ver:
        drop |= set(map(str, ver.get("reveals_diagnosis", []))) | set(map(str, ver.get("unsupported", [])))
        if ver.get("final_diagnosis_ok") is False:
            reasons.append("verifier: final_diagnosis not ok")
    gold_name = (ex.get("final_diagnosis") or sel["final_diagnosis_raw"]).strip()
    lw = leak_words(gold_name) | leak_words(sel["final_diagnosis_raw"].replace("_", " "))
    bad_para = set(map(str, (ver or {}).get("paraphrase_not_equivalent", [])))
    findings = []
    for f in num["findings"]:
        if f["id"] in drop:
            continue
        toks = set(_norm(f["text"]).split()) | set(_norm(f["paraphrase"]).split())
        if lw & toks:
            continue
        para = f["paraphrase"].strip() if f["id"] not in bad_para and f["paraphrase"].strip() else f["text"].strip()
        findings.append({"fid": f["id"], "text": f["text"].strip(), "paraphrase": para, "stage": int(f["stage"]),
                         "chief_complaint": bool(f["chief_complaint"])})
    stage0 = [f for f in findings if f["stage"] == 0]
    if not stage0:
        reasons.append("no stage-0 finding")
    else:
        cc = [f for f in stage0 if f["chief_complaint"]]
        keep = cc[0] if cc else stage0[0]
        for f in findings:
            f["chief_complaint"] = f is keep
    mcfg = config.study()["medcase"]
    if not (mcfg["min_findings"] <= len(findings) <= mcfg["max_findings"]):
        reasons.append(f"{len(findings)} findings after drops")
    invalid = set()
    for i in (ver or {}).get("distractors_invalid", []):
        try:
            invalid.add(int(i))
        except (TypeError, ValueError):
            pass
    gnorm = _norm(gold_name)
    seen = {gnorm}
    dis = []
    for i, d in enumerate(ex.get("distractors", [])):
        if not isinstance(d, str) or i in invalid:
            continue
        dn = _norm(d)
        if not dn or dn in seen or dn in gnorm or gnorm in dn:
            continue
        seen.add(dn)
        dis.append(d.strip())
    dis = dis[: mcfg["k_candidates"] - 1]
    if len(dis) + 1 < need_k:
        reasons.append(f"only {len(dis)} valid distractors")
    names = [gold_name] + dis
    order = list(range(len(names)))
    py_rng(seed, sel["case_id"], "candidates").shuffle(order)
    codes = two_letter_codes(len(names))
    cands = [[codes[j], names[i]] for j, i in enumerate(order)]
    gold = codes[order.index(0)]
    return {"case_id": sel["case_id"], "dataset": "medcase", "role": sel["role"], "prep_order": sel["prep_order"],
            "pmcid": sel["pmcid"], "split": sel["split"], "patient": (ex.get("patient") or "an adult patient").strip(),
            "findings": findings, "candidates": cands, "gold": gold, "final_diagnosis": gold_name,
            "final_diagnosis_raw": sel["final_diagnosis_raw"], "verified": ver is not None,
            "n_dropped": len(num["findings"]) - len(findings),
            "status": "rejected" if reasons else "ok", "reject_reasons": reasons}


def finalize(ws: Workspace) -> dict | None:
    pre_key, _ = role_spec(ws, "preprocessor")
    ver_key, _ = role_spec(ws, "verifier")
    ex = outputs(ws, JOB_EXTRACT, pre_key)
    ver = outputs(ws, JOB_VERIFY, ver_key)
    if not ex:
        print("[medcase] no extraction outputs yet; skipped")
        return None
    seed = int(config.study()["seed"])
    k = int(config.study()["medcase"]["k_candidates"])
    out = []
    for sel in selected():
        e = ex.get(sel["case_id"])
        if e is None:
            continue
        needs_ver = sel["role"] in ("pilot", "panel")
        v = ver.get(sel["case_id"])
        if needs_ver and v is None:
            rec = {"case_id": sel["case_id"], "role": sel["role"], "prep_order": sel["prep_order"],
                   "status": "pending_verification", "reject_reasons": ["not verified yet"]}
        else:
            rec = finalize_case(sel, e, v, seed, k if needs_ver else 6)
        out.append(rec)
    write_jsonl(ws.mcr_prepared, out)
    # clinician-review export (pilot + panel)
    with open(ws.mcr_dir / "medcase_review.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "status", "row_type", "id", "stage", "chief_complaint", "text", "paraphrase",
                    "reviewer_ok", "reviewer_notes"])
        for r in out:
            if r["role"] not in ("pilot", "panel") or r["status"] == "pending_verification":
                continue
            for c in r["candidates"]:
                w.writerow([r["case_id"], r["status"], "candidate" + (" (gold)" if c[0] == r["gold"] else ""),
                            c[0], "", "", c[1], "", "", ""])
            for fd in r["findings"]:
                w.writerow([r["case_id"], r["status"], "finding", fd["fid"], fd["stage"], fd["chief_complaint"],
                            fd["text"], fd["paraphrase"], "", ""])
    cnt: dict = {}
    for r in out:
        cnt.setdefault(r["role"], {}).setdefault(r["status"], 0)
        cnt[r["role"]][r["status"]] += 1
    summary = {"created_at": now_iso(), "counts": cnt}
    print(f"[medcase] prepared: {json.dumps(cnt)}")
    return summary
