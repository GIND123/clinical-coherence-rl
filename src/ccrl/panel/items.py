"""Build the frozen items: one model-independent prompt per (case, transform).

Transformations (study.yaml `calls`; docs/design.md §3):

  base       findings in a seeded random order within exchangeable blocks
             (never the release's code-sorted order — calibration C7 of the
             previous study: a grouped canonical order is easier and would
             contaminate the floor)
  perm1-3    three further distinct seeded orders (meaning-preserving)
  dup        one non-fixed finding repeated later, marked "As noted above"
             (set semantics: a re-statement, not a new observation)
  para       base order, every finding (and the context sentence) in the
             frozen paraphrase wording
  cue        one certified-irrelevant sentence inserted at a seeded position
  authority  base vignette + an attending's suggestion with stated
             reliability r (evaluation only; normative LR = r(K-1)/(1-r))
  ablate     base order with one finding removed (DDXPlus: a KB-informative
             unit, |log LR for the gold diagnosis| >= log 2)
  retest     byte-identical to base, sent as a separate call (noise floor)
  schema_scores  base vignette, 0-100 plausibility schema (pilot only)
  seqA/seqB  two-turn sequential evidence (E1 then E2, and E2 then E1)

Everything random is drawn from a seed derived from (master seed, case id,
transform), so rebuilding items is byte-identical.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from ccrl import config
from ccrl.data import ddxplus as dx
from ccrl.panel.cases import (PreparedCase, ddx_cases_raw, ddx_prepared, load_mcr_prepared,
                              load_phrasebank, mcr_prepared)
from ccrl.prompts import templates as T
from ccrl.util import py_rng, read_json, sha256_file, write_json, write_jsonl
from ccrl.workspace import Workspace

ALL_TRANSFORMS = ("base", "perm1", "perm2", "perm3", "dup", "para", "cue", "authority", "ablate",
                  "retest", "schema_scores", "seqA", "seqB")
DDX_REF = "ddx49"


class BuildError(RuntimeError):
    pass


# ----------------------------------------------------------------- ordering
def _ordered(case: PreparedCase, label: str, seed: int) -> list:
    r = py_rng(seed, case.case_id, "order", label)
    out = []
    for b in sorted({f.block for f in case.findings}):
        members = [f for f in case.findings if f.block == b]
        fixed = [f for f in members if f.fixed]
        free = [f for f in members if not f.fixed]
        r.shuffle(free)
        out.extend(fixed + free)
    return out


def orders(case: PreparedCase, seed: int, n_perm: int = 3) -> dict[str, list]:
    """base + n distinct permutations (distinct whenever the case admits it)."""
    base = _ordered(case, "base", seed)
    seen = {tuple(f.fid for f in base)}
    out = {"base": base}
    for k in range(1, n_perm + 1):
        cand = None
        for attempt in range(200):
            o = _ordered(case, f"perm{k}.{attempt}", seed)
            key = tuple(f.fid for f in o)
            if key not in seen:
                cand = o
                break
        if cand is None:  # tiny case: accept a repeat rather than fail
            cand = _ordered(case, f"perm{k}.0", seed)
        seen.add(tuple(f.fid for f in cand))
        out[f"perm{k}"] = cand
    return out


# ----------------------------------------------------------------- rendering
_SAFE_LOWER = {"The", "She", "He", "They", "Her", "His", "Their", "On", "A", "An", "In", "At",
               "This", "There", "It", "After", "During", "Over", "Initial", "Repeat", "Physical"}


def dup_sentence(s: str) -> str:
    first = s.split(" ", 1)[0]
    if first in _SAFE_LOWER:
        return "As noted above, " + s[0].lower() + s[1:]
    return "As noted above: " + s


def render(case: PreparedCase, ordered: list, variant: str = "base",
           inserts: list[tuple[int, str]] | None = None, with_context: bool = True) -> str:
    """Paragraphs: context + findings; a break before each block in
    case.paragraph_blocks. inserts: (index in `ordered`, sentence) placed
    right after that finding."""
    inserts = inserts or []
    seq: list[tuple[int, str]] = []
    for i, f in enumerate(ordered):
        seq.append((f.block, f.para if variant == "para" else f.base))
        for pos, s in inserts:
            if pos == i:
                seq.append((f.block, s))
    paras: list[list[str]] = []
    if with_context:
        paras.append([case.context_para if variant == "para" else case.context_base])
    prev = None
    for block, s in seq:
        if not paras or (prev is not None and block != prev and block in case.paragraph_blocks):
            paras.append([])
        paras[-1].append(s)
        prev = block
    return "\n\n".join(" ".join(p) for p in paras if p)


# ----------------------------------------------------------------- cues
def cue_ok(text: str, candidates: list[tuple[str, str]]) -> bool:
    low = text.lower()
    for _code, name in candidates:
        for part in [name] + [p for p in name.replace("/", " ").replace("(", " ").replace(")", " ").split()
                              if len(p) >= 6]:
            if part.lower() in low:
                return False
    return True


def pick_cue(case: PreparedCase, seed: int) -> dict:
    bank = config.cues()["irrelevant"]
    r = py_rng(seed, case.case_id, "cue")
    start = r.randrange(len(bank))
    for k in range(len(bank)):
        c = bank[(start + k) % len(bank)]
        if cue_ok(c["text"], case.candidates):
            return c
    raise BuildError(f"no cue compatible with candidates of {case.case_id}")


def authority_text(code: str, name: str, r: float) -> str:
    tpl = " ".join(config.cues()["authority"]["text"].split())
    return tpl.replace("{pct}", f"{round(100 * r):d}").replace("{name}", name).replace("{code}", code)


# ----------------------------------------------------------------- per case
def _ddx_ablation(case: PreparedCase, seed: int, thr: float) -> tuple[str, dict]:
    lik = dx.load_likelihood()
    gold_i = case.codes.index(case.gold)
    cands = []
    for f in case.findings:
        if f.fixed:
            continue
        llr = lik.unit_log_lr(f.meta["tokens"], gold_i)
        cands.append((f.fid, llr))
    informative = [c for c in cands if abs(c[1]) >= thr]
    r = py_rng(seed, case.case_id, "ablate")
    if informative:
        fid, llr = informative[r.randrange(len(informative))]
        rule = "random_informative"
    else:
        fid, llr = max(cands, key=lambda c: abs(c[1]))
        rule = "max_abs_llr_fallback"
    f = case.by_id()[fid]
    remaining = [t for g in case.findings if g.fid != fid for t in g.meta["tokens"]]
    q_full = lik.posterior(case.meta["tokens"])
    q_abl = lik.posterior(remaining)
    meta = {"ablated": fid, "ablated_tokens": f.meta["tokens"], "log_lr_gold": round(llr, 6),
            "selection": rule, "n_informative": len(informative),
            "unit_loglik": [round(float(x), 5) for x in lik.unit_loglik(f.meta["tokens"])],
            "kb_bayes_full": [round(float(x), 7) for x in q_full],
            "kb_bayes_ablated": [round(float(x), 7) for x in q_abl]}
    return fid, meta


def _authority_code(case: PreparedCase, seed: int) -> str:
    r = py_rng(seed, case.case_id, "authority")
    if case.dataset == "ddxplus":
        q = dx.load_likelihood().posterior(case.meta["tokens"])
        order = [case.codes[i] for i in np.argsort(-q)]
        pool = [c for c in order if c != case.gold][:5]
    else:
        pool = [c for c in case.codes if c != case.gold]
    return pool[r.randrange(len(pool))]


def items_for_case(case: PreparedCase, transforms: list[str], seed: int, systems: dict[str, str]) -> list[dict]:
    st = config.study()
    od = orders(case, seed)
    base = od["base"]
    out: list[dict] = []
    is_ddx = case.dataset == "ddxplus"
    cand_ref = DDX_REF if is_ddx else [list(x) for x in case.candidates]
    user_cands = None if case.candidates_fixed else case.candidates
    sys_post = "posterior_fixed_ddx49" if is_ddx else "posterior_case"
    sys_score = "scores_fixed_ddx49" if is_ddx else "scores_case"
    common = {"case_id": case.case_id, "dataset": case.dataset, "gold": case.gold, "candidates": cand_ref}
    base_fids = [f.fid for f in base]

    def item(transform: str, turns: list[str], kind: str = "posterior", system: str = sys_post, **meta):
        return {"item_id": f"{case.case_id}::{transform}", "transform": transform, "kind": kind,
                "system": system, "turns": turns, "meta": meta, **common}

    for t in transforms:
        if t in ("base", "retest"):
            out.append(item(t, [T.user_prompt(render(case, base), user_cands)], order=base_fids))
        elif t.startswith("perm"):
            o = od[t]
            out.append(item(t, [T.user_prompt(render(case, o), user_cands)], order=[f.fid for f in o]))
        elif t == "para":
            out.append(item(t, [T.user_prompt(render(case, base, "para"), user_cands)], order=base_fids))
        elif t == "dup":
            r = py_rng(seed, case.case_id, "dup")
            free = [i for i, f in enumerate(base) if not f.fixed]
            i = free[r.randrange(len(free))]
            later = list(range(i + 1, len(base))) or [i]
            pos = later[r.randrange(len(later))]
            s = dup_sentence(base[i].base)
            out.append(item(t, [T.user_prompt(render(case, base, inserts=[(pos, s)]), user_cands)],
                            order=base_fids, duplicated=base[i].fid, inserted_after=base[pos].fid))
        elif t == "cue":
            cue = pick_cue(case, seed)
            r = py_rng(seed, case.case_id, "cue_pos")
            pos = r.randrange(len(base))
            out.append(item(t, [T.user_prompt(render(case, base, inserts=[(pos, cue["text"])]), user_cands)],
                            order=base_fids, cue_id=cue["id"], inserted_after=base[pos].fid))
        elif t == "authority":
            code = _authority_code(case, seed)
            name = dict(case.candidates)[code]
            rel = float(st["authority_cue"]["stated_reliability"])
            K = len(case.candidates)
            text = authority_text(code, name, rel)
            out.append(item(t, [T.user_prompt(render(case, base), user_cands, authority=text)],
                            order=base_fids, suggested=code, stated_reliability=rel,
                            normative_log_lr=round(math.log(rel * (K - 1) / (1 - rel)), 6)))
        elif t == "ablate":
            if is_ddx:
                fid, meta = _ddx_ablation(case, seed, float(st["ddxplus"]["informative_log_lr"]))
            else:
                r = py_rng(seed, case.case_id, "ablate")
                free = [f for f in base if not f.fixed]
                fid = free[r.randrange(len(free))].fid
                meta = {"ablated": fid, "selection": "random_nonfixed"}
            o = [f for f in base if f.fid != fid]
            out.append(item(t, [T.user_prompt(render(case, o), user_cands)], order=[f.fid for f in o], **meta))
        elif t == "schema_scores":
            out.append(item(t, [T.user_prompt(render(case, base), user_cands)], kind="scores",
                            system=sys_score, order=base_fids))
        elif t in ("seqA", "seqB"):
            r = py_rng(seed, case.case_id, "seq_split")
            free = [f for f in base if not f.fixed]
            idx = list(range(len(free)))
            r.shuffle(idx)
            h = math.ceil(len(free) / 2)
            e1 = {free[i].fid for i in idx[:h]}
            first_ids = e1 if t == "seqA" else {f.fid for f in free} - e1
            fixed = [f for f in base if f.fixed]
            o1 = fixed + [f for f in base if f.fid in first_ids]
            o2 = [f for f in base if (not f.fixed) and f.fid not in first_ids]
            u1 = T.user_prompt(render(case, o1), user_cands, template="user_seq_first")
            u2 = T.fill(T.load("user_seq_second"), vignette=render(case, o2, with_context=False))
            out.append(item(t, [u1, u2], first=[f.fid for f in o1], second=[f.fid for f in o2]))
        else:
            raise BuildError(f"unknown transform {t!r}")
    return out


# ----------------------------------------------------------------- sets
def _system_texts() -> dict[str, str]:
    kb = dx.load_kb()
    c = kb.candidates()
    return {
        "posterior_fixed_ddx49": T.system_prompt("posterior", c, fixed=True),
        "scores_fixed_ddx49": T.system_prompt("scores", c, fixed=True),
        "posterior_case": T.system_prompt("posterior", None, fixed=False),
        "scores_case": T.system_prompt("scores", None, fixed=False),
    }


def build(ws: Workspace, allow_missing_para: bool = False, allow_short_mcr: bool = False,
          verbose: bool = True) -> dict:
    st = config.study()
    seed = int(st["seed"])
    tsets = st["transform_sets"]
    para = load_phrasebank(ws.phrasebank)
    if para is None and not allow_missing_para:
        raise BuildError(f"phrase bank not found at {ws.phrasebank}. Run `ccrl prep phrasebank` "
                         "and `ccrl prep finalize` first.")
    systems = _system_texts()

    # cases
    ddx = [ddx_prepared(r, para) for r in ddx_cases_raw()]
    mcr_rows = load_mcr_prepared(ws.mcr_prepared)
    mcfg = st["medcase"]
    mcr_pilot = sorted([r for r in mcr_rows if r["role"] == "pilot"], key=lambda r: r["prep_order"])[: mcfg["pilot_cases"]]
    mcr_panel = sorted([r for r in mcr_rows if r["role"] == "panel"], key=lambda r: r["prep_order"])[: mcfg["panel_cases"]]
    short = []
    if len(mcr_pilot) < mcfg["pilot_cases"]:
        short.append(f"MedCaseReasoning pilot: {len(mcr_pilot)}/{mcfg['pilot_cases']} prepared")
    if len(mcr_panel) < mcfg["panel_cases"]:
        short.append(f"MedCaseReasoning panel: {len(mcr_panel)}/{mcfg['panel_cases']} prepared")
    if short and not allow_short_mcr:
        raise BuildError("; ".join(short) + ". Run `ccrl prep mcr` + `ccrl prep finalize` "
                         "(or pass --allow-short-mcr for a partial build).")
    for k, r in enumerate(mcr_panel):
        r.setdefault("flags", {})["core"] = k < mcfg["core_cases"]
    mcr_p = [mcr_prepared(r) for r in mcr_pilot]
    mcr_m = [mcr_prepared(r) for r in mcr_panel]

    plan: dict[str, list[tuple[PreparedCase, list[str]]]] = {
        "pilot": [(c, tsets["pilot"]) for c in ddx if c.role == "pilot"] + [(c, tsets["pilot"]) for c in mcr_p],
        "panel": [(c, tsets["all10"]) for c in ddx if c.role == "panel"] + [(c, tsets["all10"]) for c in mcr_m],
        "core": [(c, tsets["effort6"]) for c in ddx if c.role == "panel" and c.flags.get("core")]
        + [(c, tsets["effort6"]) for c in mcr_m if c.flags.get("core")],
        "sequential": [(c, ["seqA", "seqB"]) for c in ddx if c.role == "panel" and c.flags.get("sequential")],
        "distill": [(c, tsets["base_only"]) for c in ddx if c.role == "distill"],
        "panel_ext": [(c, tsets["all10"]) for c in ddx if c.role == "panel_ext"],
    }
    needed: dict[str, tuple[PreparedCase, set[str]]] = {}
    sets: dict[str, list[str]] = {}
    for name, entries in plan.items():
        ids = []
        for case, ts in entries:
            slot = needed.setdefault(case.case_id, (case, set()))
            slot[1].update(ts)
            ids.extend(f"{case.case_id}::{t}" for t in ts)
        sets[name] = ids

    items: list[dict] = []
    for cid in sorted(needed):
        case, ts = needed[cid]
        ordered_ts = [t for t in ALL_TRANSFORMS if t in ts]
        items.extend(items_for_case(case, ordered_ts, seed, systems))
    by_id = {it["item_id"]: it for it in items}
    for name, ids in sets.items():
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise BuildError(f"set {name}: {len(missing)} items missing, e.g. {missing[:3]}")

    # case index for the analysis (everything known about each case, no model output)
    index = []
    for cid in sorted(needed):
        case = needed[cid][0]
        rec = {"case_id": cid, "dataset": case.dataset, "role": case.role, "flags": case.flags,
               "gold": case.gold, "codes": case.codes, "n_findings": len(case.findings)}
        if case.dataset == "ddxplus":
            rec.update(pathology=case.meta["pathology"], severity=case.meta["severity"],
                       kb_bayes=[round(float(x), 7) for x in dx.load_likelihood().posterior(case.meta["tokens"])])
        else:
            rec.update(final_diagnosis=case.meta["final_diagnosis"], pmcid=case.meta["pmcid"],
                       candidates=[list(x) for x in case.candidates])
        index.append(rec)

    d = ws.items_dir
    write_json(d / "systems.json", systems)
    n = write_jsonl(d / "items.jsonl", items)
    write_json(d / "sets.json", sets)
    write_jsonl(d / "case_index.jsonl", index)
    summary = {
        "n_items": n,
        "sets": {k: len(v) for k, v in sets.items()},
        "transforms": dict(Counter(it["transform"] for it in items)),
        "datasets": dict(Counter(it["dataset"] for it in items)),
        "phrasebank": str(ws.phrasebank) if para else None,
        "warnings": short,
        "items_sha256": sha256_file(d / "items.jsonl"),
    }
    write_json(d / "build_summary.json", summary)
    if verbose:
        print("[build-items]", {k: summary[k] for k in ("n_items", "sets", "warnings")})
    return summary


# ----------------------------------------------------------------- loading
def load_items(ws: Workspace) -> dict[str, dict]:
    from ccrl.util import read_jsonl
    p = ws.items_dir / "items.jsonl"
    if not p.exists():
        raise BuildError(f"{p} not found; run `ccrl build items`")
    return {it["item_id"]: it for it in read_jsonl(p)}


def load_sets(ws: Workspace) -> dict[str, list[str]]:
    return read_json(ws.items_dir / "sets.json")


def load_systems(ws: Workspace) -> dict[str, str]:
    return read_json(ws.items_dir / "systems.json")


def candidates_of(item: dict) -> list[tuple[str, str]]:
    if item["candidates"] == DDX_REF:
        return dx.load_kb().candidates()
    return [tuple(x) for x in item["candidates"]]
