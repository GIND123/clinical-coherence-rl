"""Prepared cases: the dataset-agnostic form every transformation works on.

A PreparedCase is a context sentence plus an ordered list of findings, each
with a base and a paraphrased sentence, an exchangeability block, and a
``fixed`` flag (the chief complaint stays first in its block). DDXPlus and
MedCaseReasoning differ only in how a PreparedCase is produced.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from ccrl import paths
from ccrl.data import ddx_render as R
from ccrl.data import ddxplus as dx
from ccrl.util import read_json, read_jsonl


@dataclass
class Finding:
    fid: str
    base: str
    para: str
    block: int
    fixed: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class PreparedCase:
    case_id: str
    dataset: str                       # ddxplus | medcase
    role: str
    flags: dict
    context_base: str
    context_para: str
    findings: list[Finding]
    candidates: list[tuple[str, str]]  # (code, name), presentation order
    gold: str
    candidates_fixed: bool             # True -> list lives in the (cacheable) system block
    paragraph_blocks: tuple[int, ...]  # a new paragraph starts before these blocks
    meta: dict = field(default_factory=dict)

    @property
    def codes(self) -> list[str]:
        return [c for c, _ in self.candidates]

    def by_id(self) -> dict[str, Finding]:
        return {f.fid: f for f in self.findings}

    def to_json(self) -> dict:
        d = asdict(self)
        d["candidates"] = [list(x) for x in self.candidates]
        return d


# --------------------------------------------------------------------- DDXPlus
@lru_cache(maxsize=1)
def ddx_cases_raw() -> tuple[dict, ...]:
    return tuple(read_jsonl(paths.FROZEN_DDX / "cases.jsonl"))


def load_phrasebank(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    d = read_json(path)
    return d.get("templates", d)


def ddx_prepared(rec: dict, para_templates: dict[str, str] | None) -> PreparedCase:
    kb = dx.load_kb()
    base_T = R.all_base_templates(kb)
    para_T = dict(base_T)
    if para_templates:
        para_T.update(para_templates)
    units = [dx.Unit.from_json(u) for u in rec["units"]]
    findings = []
    for u in units:
        findings.append(Finding(
            fid=u.uid,
            base=R.unit_sentence(kb, u, base_T),
            para=R.unit_sentence(kb, u, para_T),
            block=u.block,
            fixed=u.is_cc,
            meta={"tokens": u.tokens},
        ))
    return PreparedCase(
        case_id=rec["case_id"], dataset="ddxplus", role=rec["role"], flags=rec.get("flags", {}),
        context_base=R.context_sentence(rec["age"], rec["sex"], "base"),
        context_para=R.context_sentence(rec["age"], rec["sex"], "para"),
        findings=findings, candidates=kb.candidates(), gold=rec["gold"], candidates_fixed=True,
        paragraph_blocks=(2,),
        meta={"pathology": rec["pathology"], "severity": rec["severity"], "age": rec["age"],
              "sex": rec["sex"], "tokens": rec["tokens"], "row_id": rec["row_id"], "split": rec["split"]},
    )


def ddx_kb_bayes(tokens: list[str]) -> np.ndarray:
    return dx.load_likelihood().posterior(tokens)


# ------------------------------------------------------------ MedCaseReasoning
def mcr_prepared(rec: dict) -> PreparedCase:
    """From one row of prepared.jsonl (output of `ccrl prep finalize`)."""
    findings = [Finding(fid=f["fid"], base=f["text"], para=f["paraphrase"], block=int(f["stage"]),
                        fixed=bool(f.get("chief_complaint")), meta={})
                for f in rec["findings"]]
    patient = rec["patient"].strip().rstrip(".")
    return PreparedCase(
        case_id=rec["case_id"], dataset="medcase", role=rec["role"], flags=rec.get("flags", {}),
        context_base=f"The patient is {patient}.",
        context_para=f"This case concerns {patient}.",
        findings=findings, candidates=[tuple(x) for x in rec["candidates"]], gold=rec["gold"],
        candidates_fixed=False, paragraph_blocks=(1, 2, 3),
        meta={"final_diagnosis": rec["final_diagnosis"], "pmcid": rec.get("pmcid"),
              "prep_order": rec.get("prep_order")},
    )


def load_mcr_prepared(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [r for r in read_jsonl(path) if r.get("status") == "ok"]


def dumps(case: PreparedCase) -> str:
    return json.dumps(case.to_json(), ensure_ascii=False, sort_keys=True)
