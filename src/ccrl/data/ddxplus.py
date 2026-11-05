"""DDXPlus: knowledge base, corpus parsing, finding units, corpus likelihoods.

DDXPlus (Fansi Tchango et al., NeurIPS 2022 Datasets & Benchmarks; CC BY 4.0)
is ~1.3M synthetic patients over 49 pathologies and 223 questionnaire
evidences. Each patient carries a list of *positive* evidence tokens:

    "E_91"            binary evidence, present
    "E_56_@_7"        categorical value (pain intensity 7/10)
    "E_55_@_V_29"     multi-choice value (pain location: a body part)

Finding units
-------------
Several evidences are descriptors of a parent question (`code_question`):
pain location/character/intensity are children of "pain related to the
reason for consulting" (E_53), rash descriptors of E_129, and so on. A
*finding unit* is one parent evidence together with all of its children that
the patient has. Units are what the transformations permute, duplicate and
ablate, so a permutation never separates "the pain is burning" from "has
pain".

Exchangeability (study rule 3.2.1): the chief-complaint unit stays first;
symptom units are exchangeable among themselves, antecedent units among
themselves; the symptom block precedes the antecedent block.

Corpus likelihoods
------------------
The released knowledge base has no conditional probabilities (every slot in
release_conditions.json is empty). P(token | pathology) is therefore
estimated from the training split with a Jeffreys prior. The "KB-Bayes"
reference posterior is naive Bayes over the *presented* (positive) tokens:

    q*(d | E) ∝ π(d) · Π_{t ∈ E} P(t | d)

which is exactly the posterior a reasoner holding these likelihoods should
report for a vignette that lists those findings. It is an approximation to
the generator (conditional independence does not hold in DDXPlus); its fit
to the shipped differentials is measured and reported, never assumed.
"""
from __future__ import annotations

import ast
import csv
import io
import math
import re
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np

from ccrl import paths
from ccrl.util import read_json, two_letter_codes

TOKEN_RE = re.compile(r"^(E_\d+)(?:_@_(.+))?$")


def split_token(token: str) -> tuple[str, str | None]:
    m = TOKEN_RE.match(token)
    if not m:
        raise ValueError(f"unparseable DDXPlus evidence token: {token!r}")
    return m.group(1), m.group(2)


# --------------------------------------------------------------------------
# Knowledge base
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Evidence:
    code: str
    question: str
    is_antecedent: bool
    data_type: str            # B binary, C categorical/ordinal, M multi-choice
    parent: str               # code_question; == code for top-level questions
    value_meaning: dict       # value id -> English meaning (may be empty)
    default_value: str

    @property
    def is_child(self) -> bool:
        return self.parent != self.code


@dataclass
class KB:
    pathologies: list[str]                 # sorted English names (the candidate order)
    codes: list[str]                       # two-letter code per pathology, same order
    conditions: dict[str, dict]            # name -> {icd10, severity}
    evidences: dict[str, Evidence]
    children: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kids: dict[str, list[str]] = {}
        for c, e in self.evidences.items():
            if e.is_child:
                kids.setdefault(e.parent, []).append(c)
        self.children = {p: sorted(v, key=_code_num) for p, v in kids.items()}
        self.name_to_code = dict(zip(self.pathologies, self.codes))
        self.code_to_name = dict(zip(self.codes, self.pathologies))

    @property
    def K(self) -> int:
        return len(self.pathologies)

    def candidates(self) -> list[tuple[str, str]]:
        return list(zip(self.codes, self.pathologies))


def _code_num(code: str) -> int:
    return int(code.split("_")[1])


def kb_from_raw(conditions_path: Path, evidences_path: Path) -> KB:
    rc = read_json(conditions_path)
    re_ = read_json(evidences_path)
    evidences = {
        k: Evidence(
            code=k,
            question=v["question_en"].strip(),
            is_antecedent=bool(v["is_antecedent"]),
            data_type=v["data_type"],
            parent=v.get("code_question", k) or k,
            value_meaning={str(vk): vm["en"] for vk, vm in (v.get("value_meaning") or {}).items()},
            default_value=str(v.get("default_value")),
        )
        for k, v in re_.items()
    }
    names = sorted(v["cond-name-eng"] for v in rc.values())
    by_name = {v["cond-name-eng"]: v for v in rc.values()}
    conditions = {n: {"icd10": by_name[n]["icd10-id"], "severity": int(by_name[n]["severity"])}
                  for n in names}
    return KB(pathologies=names, codes=two_letter_codes(len(names)),
              conditions=conditions, evidences=evidences)


def kb_to_json(kb: KB) -> dict:
    return {
        "pathologies": kb.pathologies,
        "codes": kb.codes,
        "conditions": kb.conditions,
        "evidences": {k: {"question": e.question, "is_antecedent": e.is_antecedent,
                          "data_type": e.data_type, "parent": e.parent,
                          "value_meaning": e.value_meaning, "default_value": e.default_value}
                      for k, e in sorted(kb.evidences.items(), key=lambda kv: _code_num(kv[0]))},
    }


def kb_from_json(d: dict) -> KB:
    ev = {k: Evidence(code=k, **v) for k, v in d["evidences"].items()}
    return KB(pathologies=d["pathologies"], codes=d["codes"], conditions=d["conditions"], evidences=ev)


@lru_cache(maxsize=1)
def load_kb() -> KB:
    """The frozen, committed KB (no raw corpus needed)."""
    return kb_from_json(read_json(paths.FROZEN_DDX / "kb.json"))


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------
SPLIT_FILES = {
    "train": "release_train_patients",
    "validate": "release_validate_patients",
    "test": "release_test_patients",
}


def _open_split(raw_dir: Path, split: str) -> io.TextIOBase:
    name = SPLIT_FILES[split]
    extracted = raw_dir / "extracted" / name
    if extracted.exists():
        return open(extracted, "r", encoding="utf-8", newline="")
    z = raw_dir / f"{name}.zip"
    if not z.exists():
        raise FileNotFoundError(
            f"DDXPlus split {split!r} not found under {raw_dir}. "
            "Run `ccrl data download ddxplus` (or set CCRL_DATA).")
    zf = zipfile.ZipFile(z)
    member = [n for n in zf.namelist() if not n.endswith("/")][0]
    return io.TextIOWrapper(zf.open(member), encoding="utf-8", newline="")


def iter_split(split: str, raw_dir: Path | None = None) -> Iterator[dict]:
    """Yield one dict per patient. Row order is the release order (row_id)."""
    raw_dir = raw_dir or paths.DATA_DDX
    with _open_split(raw_dir, split) as f:
        for i, r in enumerate(csv.DictReader(f)):
            yield {
                "row_id": i,
                "split": split,
                "age": int(r["AGE"]),
                "sex": r["SEX"].strip(),
                "pathology": r["PATHOLOGY"].strip(),
                "initial_evidence": r["INITIAL_EVIDENCE"].strip(),
                "evidences": ast.literal_eval(r["EVIDENCES"]),
                "differential": [(n, float(p)) for n, p in ast.literal_eval(r["DIFFERENTIAL_DIAGNOSIS"])],
            }


# --------------------------------------------------------------------------
# Finding units
# --------------------------------------------------------------------------
@dataclass
class Unit:
    uid: str                      # the parent evidence code
    tokens: list[str]             # all tokens of the parent and its children
    values: dict[str, list[str]]  # code -> values (None for binary present)
    is_antecedent: bool
    is_cc: bool = False

    @property
    def block(self) -> int:
        """0 = chief complaint, 1 = symptoms, 2 = antecedents."""
        return 0 if self.is_cc else (2 if self.is_antecedent else 1)

    def to_json(self) -> dict:
        return {"uid": self.uid, "tokens": self.tokens, "values": self.values,
                "is_antecedent": self.is_antecedent, "is_cc": self.is_cc}

    @classmethod
    def from_json(cls, d: dict) -> "Unit":
        return cls(**d)


def build_units(kb: KB, tokens: list[str], initial_evidence: str) -> list[Unit]:
    """Group a patient's tokens into finding units (parent + children)."""
    by_code: dict[str, list] = {}
    order: list[str] = []
    for t in tokens:
        code, val = split_token(t)
        if code not in kb.evidences:
            raise KeyError(f"evidence {code} not in KB")
        if code not in by_code:
            by_code[code] = []
            order.append(code)
        by_code[code].append((t, val))
    present = set(by_code)
    cc_code = split_token(initial_evidence)[0]
    units: dict[str, Unit] = {}
    for code in order:
        ev = kb.evidences[code]
        head = ev.parent if (ev.is_child and ev.parent in present) else code
        u = units.get(head)
        if u is None:
            hev = kb.evidences[head]
            u = units[head] = Unit(uid=head, tokens=[], values={}, is_antecedent=hev.is_antecedent)
        for t, v in by_code[code]:
            u.tokens.append(t)
            u.values.setdefault(code, [])
            if v is not None:
                u.values[code].append(v)
    # The chief complaint is the unit containing the initial evidence.
    for u in units.values():
        if cc_code in u.values:
            u.is_cc = True
            break
    # Stable order: CC first, then the release order of the remaining heads.
    out = sorted(units.values(), key=lambda u: (not u.is_cc, _code_num(u.uid)))
    return out


# --------------------------------------------------------------------------
# Corpus likelihoods and the KB-Bayes reference posterior
# --------------------------------------------------------------------------
@dataclass
class Likelihood:
    pathologies: list[str]
    N: np.ndarray                    # (D,) patients per pathology in the train split
    tokens: list[str]
    counts: np.ndarray               # (T, D) patients with token t and pathology d
    alpha: float = 0.5

    def __post_init__(self) -> None:
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.log_prior = np.log(self.N / self.N.sum())
        self.loglik = np.log((self.counts + self.alpha) / (self.N[None, :] + 2 * self.alpha))
        # log-lik of a token never seen in training: smoothing mass only
        self.loglik_unseen = np.log(self.alpha / (self.N + 2 * self.alpha))

    @property
    def D(self) -> int:
        return len(self.pathologies)

    def token_loglik(self, token: str) -> np.ndarray:
        i = self.index.get(token)
        return self.loglik[i] if i is not None else self.loglik_unseen

    def unit_loglik(self, tokens: list[str]) -> np.ndarray:
        """log P(unit | d), naive within the unit. (D,)"""
        out = np.zeros(self.D)
        for t in tokens:
            out = out + self.token_loglik(t)
        return out

    tau: float = 1.0   # likelihood temperature of the reference posterior (fit_tau); 1 = naive Bayes

    def posterior(self, tokens: list[str], tau: float | None = None) -> np.ndarray:
        """Reference posterior  q(d|E) ∝ π(d) Π_t P(t|d)^tau  (generalised Bayes).

        tau = 1 is naive Bayes, which is badly over-confident on DDXPlus
        (findings are correlated but multiplied as if independent). The
        study's reference uses the tau fitted to the generator's own released
        differentials (`fit_tau`). Any tau > 0 keeps the posterior a function
        of the evidence SET, so Proposition 1 (order invariance) is unaffected.
        """
        t = self.tau if tau is None else tau
        lp = self.log_prior + t * self.unit_loglik(tokens)
        lp = lp - lp.max()
        p = np.exp(lp)
        return p / p.sum()

    def unit_log_lr(self, tokens: list[str], d: int) -> float:
        """log P(unit | d) - log P(unit | not d), prior-weighted over the rest."""
        ll = self.unit_loglik(tokens)
        w = np.exp(self.log_prior).copy()
        w[d] = 0.0
        w = w / w.sum()
        m = ll.max()
        not_d = m + math.log(float(np.sum(w * np.exp(ll - m))))
        return float(ll[d] - not_d)

    def to_json(self) -> dict:
        return {"pathologies": self.pathologies, "N": [int(x) for x in self.N],
                "alpha": self.alpha,
                "tokens": {t: [int(x) for x in self.counts[i]] for i, t in enumerate(self.tokens)}}

    @classmethod
    def from_json(cls, d: dict) -> "Likelihood":
        toks = list(d["tokens"])
        return cls(pathologies=d["pathologies"], N=np.asarray(d["N"], dtype=np.float64),
                   tokens=toks, counts=np.asarray([d["tokens"][t] for t in toks], dtype=np.float64),
                   alpha=float(d.get("alpha", 0.5)))


def fit_likelihood(rows: Iterator[dict], pathologies: list[str], alpha: float = 0.5) -> Likelihood:
    pidx = {p: i for i, p in enumerate(pathologies)}
    N = np.zeros(len(pathologies))
    counts: dict[str, np.ndarray] = {}
    for r in rows:
        d = pidx[r["pathology"]]
        N[d] += 1
        for t in set(r["evidences"]):
            a = counts.get(t)
            if a is None:
                a = counts[t] = np.zeros(len(pathologies))
            a[d] += 1
    toks = sorted(counts, key=lambda t: (_code_num(split_token(t)[0]), t))
    return Likelihood(pathologies=pathologies, N=N, tokens=toks,
                      counts=np.stack([counts[t] for t in toks]), alpha=alpha)


@lru_cache(maxsize=1)
def load_likelihood() -> Likelihood:
    lik = Likelihood.from_json(read_json(paths.FROZEN_DDX / "likelihood.json"))
    ref = paths.FROZEN_DDX / "reference.json"
    if ref.exists():
        lik.tau = float(read_json(ref)["tau"])
    return lik


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    a = p > 0
    b = q > 0
    return float(0.5 * np.sum(p[a] * np.log2(p[a] / m[a])) + 0.5 * np.sum(q[b] * np.log2(q[b] / m[b])))


def fit_tau(cases: list[dict], lik: Likelihood, grid: list[float] | None = None) -> dict:
    """Choose the reference temperature tau by the proper scoring rule: minimise
    the mean negative log-likelihood of the true pathology on TRAINING patients.

    Diagnostics reported alongside: top-1 accuracy, mean entropy, and the mean
    JSD to the generator's released differential (which is far flatter than any
    positives-only posterior, so it is reported, not used as the criterion —
    minimising it drives tau to the prior)."""
    grid = grid or [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    idx = {p: i for i, p in enumerate(lik.pathologies)}
    table = []
    for t in grid:
        nll, top1, ent, jsd = [], [], [], []
        for c in cases:
            q = lik.posterior(c["tokens"], t)
            gi = idx[c["pathology"]]
            nll.append(-np.log(max(q[gi], 1e-12)))
            top1.append(float(np.argmax(q) == gi))
            nz = q[q > 0]
            ent.append(float(-(nz * np.log(nz)).sum()))
            r = np.zeros(lik.D)
            for n, p in c["differential"]:
                r[idx[n]] = p
            if r.sum() > 0:
                jsd.append(_jsd(q, r / r.sum()))
        table.append({"tau": float(t), "nll": float(np.mean(nll)), "top1": float(np.mean(top1)),
                      "entropy": float(np.mean(ent)), "mean_jsd_to_differential": float(np.mean(jsd))})
    best = min(table, key=lambda r: r["nll"])
    return {"tau": best["tau"], "nll_at_tau": best["nll"], "top1_at_tau": best["top1"],
            "entropy_at_tau": best["entropy"], "n_cases": len(cases),
            "rule": "argmin mean NLL of the true pathology, train split (proper scoring rule)", "grid": table}
