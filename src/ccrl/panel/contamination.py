"""Amendment 2 (docs/amendment_2.md): is the MedCaseReasoning panel contaminated?

The 300 real case reports in the confirmatory panel were published 2005-2025, before every
panel model's training cutoff, so a before/after-cutoff split is impossible. Two probes per
case and model instead (4 calls; provider-default decoding, like the panel):

  id       PMCID + journal + year and the case's 10 candidate diagnoses, clinical details
           withheld -> posterior JSON (same system prompt and parser as the panel)
  noid     the same candidates with the publication's identity withheld (baseline)
  guided   source identity + the first part of the case presentation -> "continue exactly
           as in the original publication"
  general  the first part only -> "continue it"

Memorisation shows as id > noid (recall of the PMCID -> diagnosis mapping) and as
guided > general in verbatim overlap with the withheld second part (Golchin & Surdeanu's
guided-instruction test). `analyze` then asks whether suspect cases inflate the panel's
real-case accuracy or stability.

    ccrl contamination plan       calls and cost forecast ($0)
    ccrl contamination run        the paid probe (~7,200 calls, ~$60, mostly Kimi K3)
    ccrl contamination analyze    tables under runs/reports/analysis/
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ccrl import config, paths
from ccrl.bedrock import models as M
from ccrl.bedrock.ledger import Ledger
from ccrl.bedrock.runner import BudgetGuard, Task, preflight_time, run_tasks, task_id
from ccrl.prompts import templates as T
from ccrl.util import read_jsonl, two_letter_codes
from ccrl.workspace import Workspace

JOB = "contamination"
TRANSFORMS = ("id", "noid", "guided", "general")
MC = ("id", "noid")
N_WORDS_MIN, N_WORDS_MAX = 40, 150
SUSPECT_SUBSTRING = 13            # tokens: a shared 13-gram counts as verbatim recall
N_BOOT = 2000
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_THINK = re.compile(r"<think>.*?</think>", re.S)


# ------------------------------------------------------------------ cases and prompts
def panel_cases(ws: Workspace) -> list[dict]:
    """The MedCaseReasoning cases of the confirmatory panel, in item order."""
    ci = read_jsonl(ws.items_dir / "case_index.jsonl")
    ids = [r["case_id"] for r in ci if r["dataset"] == "medcase" and r["role"] == "panel"]
    prep = {r["case_id"]: r for r in read_jsonl(ws.mcr_prepared)}
    sel = {r["case_id"]: r for r in read_jsonl(paths.FROZEN_MCR / "cases_selected.jsonl")}
    out = []
    for cid in ids:
        s, p = sel[cid], prep[cid]
        out.append({"case_id": cid, "pmcid": s["pmcid"], "journal": s.get("journal") or "a medical journal",
                    "year": str(s.get("publication_date") or "")[:4] or "year unknown",
                    "case_prompt": s["case_prompt"], "candidates": [tuple(c) for c in p["candidates"]],
                    "gold": p["gold"], "final_diagnosis": p["final_diagnosis"]})
    return out


def split_text(text: str) -> tuple[str, str]:
    """First part (whole sentences, >= half the characters) and the withheld remainder."""
    sents = [s for s in _SENT.split(text.strip()) if s.strip()]
    if len(sents) < 2:
        cut = len(text) // 2
        return text[:cut].strip(), text[cut:].strip()
    total = sum(len(s) for s in sents)
    acc, k = 0, 0
    while k < len(sents) - 1 and acc < total / 2:
        acc += len(sents[k])
        k += 1
    return " ".join(sents[:k]).strip(), " ".join(sents[k:]).strip()


def n_words(reference: str) -> int:
    n = len(reference.split())
    return int(min(N_WORDS_MAX, max(N_WORDS_MIN, round(n / 10) * 10)))


def text_validator(text: str):
    t = _THINK.sub("", text or "").strip()
    ok = len(t) >= 20
    return ok, ok, json.dumps({"n_chars": len(t)})


def build_tasks(ws: Workspace, specs: dict, models: list[str] | None = None,
                limit_cases: int | None = None) -> list[Task]:
    cases = panel_cases(ws)
    if limit_cases:
        cases = cases[:limit_cases]
    keys = models or config.panel_models()
    missing = [k for k in keys if k not in specs]
    if missing:
        raise SystemExit(f"not in the model lock (or failed the probe): {missing}")
    sys_mc = T.system_prompt("posterior", None, fixed=False)
    sys_c = T.load("contam_completion_system")
    tasks: list[Task] = []
    for k, c in enumerate(cases):
        cands = c["candidates"]
        cs = "Candidate diagnoses:\n" + T.candidate_block(cands) + "\n\n"
        first, ref = split_text(c["case_prompt"])
        nw = str(n_words(ref))
        users = {
            "id": T.fill(T.load("contam_id_user"), candidates_section=cs, pmcid=c["pmcid"], journal=c["journal"],
                         year=c["year"]),
            "noid": T.fill(T.load("contam_noid_user"), candidates_section=cs),
            "guided": T.fill(T.load("contam_guided_user"), journal=c["journal"], year=c["year"], pmcid=c["pmcid"],
                             n_words=nw, first_part=first),
            "general": T.fill(T.load("contam_general_user"), n_words=nw, first_part=first),
        }
        for mk in keys:
            spec = specs[mk]
            for ti, tr in enumerate(TRANSFORMS):
                system = sys_mc if tr in MC else sys_c
                iid = f"{c['case_id']}::contam_{tr}"
                rid = task_id(JOB, iid, mk, spec.model_id, "default", 0, system, [users[tr]])
                tasks.append(Task(
                    request_id=rid, job=JOB, priority="B", model_key=mk, item_id=iid, case_id=c["case_id"],
                    transform=tr, dataset="medcase", effort="default", system=system, turns=[users[tr]],
                    kind="posterior" if tr in MC else "text",
                    codes=[x for x, _ in cands] if tr in MC else None,
                    names=dict(cands) if tr in MC else None,
                    validator=None if tr in MC else text_validator, order_key=(k, ti)))
    return tasks


# ------------------------------------------------------------------ plan / run
def _remaining(ws: Workspace, tasks: list[Task]) -> list[Task]:
    if not ws.ledger.exists():
        return tasks
    led = Ledger(ws.ledger)
    try:
        st = led.statuses(JOB)
    finally:
        led.close()
    return [t for t in tasks if st.get(t.request_id) not in ("done", "refused")]


def plan(ws: Workspace, models: list[str] | None = None, limit_cases: int | None = None) -> dict:
    from ccrl.panel.forecast import forecast_tasks
    specs = M.load_lock(ws.dry_run)
    tasks = build_tasks(ws, specs, models, limit_cases)
    rem = _remaining(ws, tasks)
    fc = forecast_tasks(rem, ws, specs)
    cap = float(config.budget()["job_caps_usd"].get(JOB, 0))
    t = fc["total"]
    print(f"[{JOB}] {len(tasks)} calls in total, {len(rem)} remaining; forecast low ${t['low']:.2f} | "
          f"expected ${t['expected']:.2f} | high ${t['high']:.2f}; job cap ${cap:.2f}")
    for mk, v in sorted(fc["by_model"].items()):
        print(f"    {mk:<16} calls {v['calls']:>6}   expected ${v['expected']:.2f}")
    return fc


def run(ws: Workspace, models: list[str] | None = None, limit_cases: int | None = None,
        max_spend: float | None = None, profile: str | None = None) -> dict:
    from ccrl.panel.freeze import check_freeze
    from ccrl.panel.jobs import backup_ledger, check_disk
    probs = config.validate_all()
    if probs:
        raise SystemExit("Config problems:\n  " + "\n  ".join(probs))
    ok, msg = check_freeze(ws)
    if not ok:
        raise SystemExit(f"Freeze check failed: {msg}\nThe probe prompts are part of the freeze (Amendment 2).")
    if not ws.dry_run:
        preflight_time()
    check_disk(ws)
    specs = M.load_lock(ws.dry_run)
    tasks = build_tasks(ws, specs, models, limit_cases)
    plan(ws, models, limit_cases)
    led = Ledger(ws.ledger)
    try:
        guard = BudgetGuard(led, JOB, run_cap=max_spend)
        factory = M.make_client_factory(ws.dry_run, profile)
        clients = {mk: factory(specs[mk]) for mk in {t.model_key for t in tasks}}
        summary = run_tasks(tasks, led, clients, specs, ws, guard, label=JOB)
        if summary.get("ran"):
            backup_ledger(ws, led, JOB)
        return summary
    finally:
        led.close()


# ------------------------------------------------------------------ scoring
def _tokens(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


def _ngrams(tok: list[str], n: int) -> list[tuple]:
    return [tuple(tok[i:i + n]) for i in range(len(tok) - n + 1)]


def rouge_n_f1(cand: list[str], ref: list[str], n: int = 2) -> float:
    c, r = Counter(_ngrams(cand, n)), Counter(_ngrams(ref, n))
    if not c or not r:
        return 0.0
    hit = sum((c & r).values())
    p, q = hit / sum(c.values()), hit / sum(r.values())
    return 0.0 if hit == 0 else 2 * p * q / (p + q)


def ngram_overlap(cand: list[str], source: list[str], n: int = 8) -> float:
    """Share of the candidate's n-grams that occur anywhere in the source text."""
    cg = _ngrams(cand, n)
    if not cg:
        return 0.0
    src = set(_ngrams(source, n))
    return float(np.mean([g in src for g in cg]))


def longest_shared(cand: list[str], source: list[str], cap: int = 60) -> int:
    """Length (tokens) of the longest token sequence the candidate shares with the source."""
    best, n = 0, 1
    while n <= min(cap, len(cand)):
        src = set(_ngrams(source, n))
        if any(g in src for g in _ngrams(cand, n)):
            best = n
            n += 1
        else:
            break
    return best


def mentions(text: str, diagnosis: str) -> bool:
    from ccrl.prep.medcase import leak_words
    words = leak_words(diagnosis)
    tok = set(_tokens(text))
    if words:
        return words <= tok
    d = " ".join(_tokens(diagnosis))
    return bool(d) and d in " ".join(_tokens(text))


def _boot(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def per_case(ws: Workspace) -> pd.DataFrame:
    """One row per case x model with both probes scored."""
    cases = {c["case_id"]: c for c in panel_cases(ws)}
    led = Ledger(ws.ledger)
    try:
        rows = led.q("SELECT r.case_id, r.model_key, r.transform, r.parse_ok, a.text, a.parsed_json FROM requests r "
                     "JOIN attempts a ON a.request_id=r.request_id AND a.attempt=r.final_attempt AND a.turn=1 "
                     "WHERE r.job=? AND r.status='done'", (JOB,))
    finally:
        led.close()
    acc: dict[tuple, dict] = {}
    for r in rows:
        c = cases.get(r["case_id"])
        if c is None:
            continue
        d = acc.setdefault((r["case_id"], r["model_key"]), {"case_id": r["case_id"], "model_key": r["model_key"]})
        tr = r["transform"]
        if tr in MC:
            pj = json.loads(r["parsed_json"]) if r["parsed_json"] else {}
            probs = pj.get("probs")
            if r["parse_ok"] and probs:
                codes = two_letter_codes(len(probs))
                d[f"top1_{tr}"] = float(codes[int(np.argmax(probs))] == c["gold"])
                d[f"pgold_{tr}"] = float(probs[codes.index(c["gold"])]) if c["gold"] in codes else np.nan
                d[f"informative_{tr}"] = float(bool(pj.get("informative")))
        else:
            text = _THINK.sub("", r["text"] or "").strip()
            _, ref = split_text(c["case_prompt"])
            ct, rt, full = _tokens(text), _tokens(ref), _tokens(c["case_prompt"])
            d[f"rouge2_{tr}"] = rouge_n_f1(ct, rt, 2)
            d[f"ngram8_{tr}"] = ngram_overlap(ct, rt, 8)
            d[f"longest_{tr}"] = float(longest_shared(ct, full))
            d[f"dx_mention_{tr}"] = float(mentions(text, c["final_diagnosis"]))
    df = pd.DataFrame(list(acc.values()))
    if df.empty:
        return df
    for col in ("top1_id", "top1_noid", "longest_guided"):
        if col not in df:
            df[col] = np.nan
    df["suspect"] = (((df["top1_id"] == 1) & (df["top1_noid"] == 0))
                     | (df["longest_guided"] >= SUSPECT_SUBSTRING)).astype(float)
    return df


def summarize(pc: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    pairs = [("top1", "id", "noid"), ("pgold", "id", "noid"), ("rouge2", "guided", "general"),
             ("ngram8", "guided", "general"), ("longest", "guided", "general"),
             ("dx_mention", "guided", "general")]
    for mk, g in pc.groupby("model_key"):
        row = {"model_key": mk, "n_cases": len(g), "suspect_rate": float(g["suspect"].mean())}
        for m, a, b in pairs:
            ca, cb = f"{m}_{a}", f"{m}_{b}"
            if ca not in g or cb not in g:
                continue
            row[ca], row[cb] = float(g[ca].mean()), float(g[cb].mean())
            d, lo, hi = _boot((g[ca] - g[cb]).to_numpy(float), rng, n_boot)
            row.update({f"d_{m}": d, f"d_{m}_lo": lo, f"d_{m}_hi": hi})
        if "longest_guided" in g:
            row["verbatim13_guided"] = float((g["longest_guided"] >= SUSPECT_SUBSTRING).mean())
            row["verbatim13_general"] = float((g.get("longest_general", pd.Series(dtype=float)) >= SUSPECT_SUBSTRING).mean())
        out.append(row)
    return pd.DataFrame(out)


def link_to_panel(ws: Workspace, pc: pd.DataFrame) -> pd.DataFrame:
    """Main-panel top-1 and retest flip on suspect vs other cases, and top-1 without suspects."""
    from ccrl.analysis.secondary import per_case as panel_per_case
    from ccrl.archive.export import posterior_table
    post = posterior_table(ws)
    if post.empty:
        return pd.DataFrame()
    post = post[(post["job"] == "panel_main") & (post["dataset"] == "medcase") & (post["turn"] == 1)
                & post["parse_ok"]]
    rows = []
    for mk, g in post.groupby("model_key"):
        pp = panel_per_case(g)
        s = pc[pc["model_key"] == mk].set_index("case_id")["suspect"]
        if pp.empty or s.empty:
            continue
        pp = pp.join(s, how="inner")
        sus, rest = pp[pp["suspect"] == 1], pp[pp["suspect"] == 0]
        rows.append({"model_key": mk, "n_cases": len(pp), "n_suspect": len(sus),
                     "top1_all": float(pp["top1"].mean()),
                     "top1_suspect": float(sus["top1"].mean()) if len(sus) else np.nan,
                     "top1_without_suspect": float(rest["top1"].mean()) if len(rest) else np.nan,
                     "flip_suspect": float(sus["flip"].mean()) if len(sus) else np.nan,
                     "flip_without_suspect": float(rest["flip"].mean()) if len(rest) else np.nan})
    return pd.DataFrame(rows)


def analyze(ws: Workspace) -> Path:
    pc = per_case(ws)
    if pc.empty:
        raise SystemExit("no contamination results in the ledger")
    out = ws.reports / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pc.to_csv(out / "contamination_per_case.csv", index=False)
    s = summarize(pc)
    s.to_csv(out / "contamination_summary.csv", index=False)
    link = link_to_panel(ws, pc)
    if not link.empty:
        link.to_csv(out / "contamination_panel_link.csv", index=False)
    print(s.round(3).to_string(index=False))
    if not link.empty:
        print()
        print(link.round(3).to_string(index=False))
    return out
