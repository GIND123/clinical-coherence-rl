"""Training data for Track B: patient pools and paired-group prompts.

    ccrl train build-pool        DDXPlus train split (8 held-out pathologies excluded)
                                 -> $CCRL_DATA/train_pool/ddxplus_train.jsonl

A paired group for patient p (docs/theory.md):
    x   full evidence, random order within exchangeable blocks
    x'  T(x), T drawn from the TRAINED meaning-preserving transforms
        {perm, dup, para, cue} (authority cues are never trained)
    x-  contrast: the triage rendering (context + chief complaint), or the
        plan's single-finding ablation (contrast="single")
Prompts use the same system/user templates as the panel, so a trained model
is evaluated on exactly the prompt format it was trained on.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ccrl import config, paths
from ccrl.data import ddxplus as dx
from ccrl.panel import items as I
from ccrl.panel.cases import PreparedCase, ddx_prepared, load_phrasebank, mcr_prepared
from ccrl.prompts import templates as T
from ccrl.util import read_json, read_jsonl, rng as seeded_rng, write_jsonl

TRAIN_TRANSFORMS = ("perm", "dup", "para", "cue")


def heldout() -> set[str]:
    return set(read_json(paths.FROZEN_DDX / "heldout.json")["heldout"])


def build_pool(n: int = 60000, seed: int | None = None) -> str:
    """Stratified pool of DDXPlus TRAIN-split patients (held-out pathologies excluded)."""
    from collections import defaultdict
    from ccrl.data.build_ddx import _eligible, case_record

    st = config.study()
    seed = int(seed if seed is not None else st["seed"])
    kb = dx.load_kb()
    held = heldout()
    by_p: dict[str, list[dict]] = defaultdict(list)
    for r in dx.iter_split("train"):
        if r["pathology"] in held:
            continue
        ok, units = _eligible(kb, r, st["ddxplus"])
        if ok:
            by_p[r["pathology"]].append(case_record(kb, r, units, "train"))
    per = max(1, n // max(1, len(by_p)))
    out = []
    for p in sorted(by_p):
        lst = by_p[p]
        idx = seeded_rng(seed, "train_pool", p).permutation(len(lst))[:per]
        out.extend(lst[i] for i in sorted(idx))
    dest = paths.DATA / "train_pool" / "ddxplus_train.jsonl"
    write_jsonl(dest, out)
    print(f"[train-pool] {len(out)} patients from {len(by_p)} pathologies -> {dest}")
    return str(dest)


@dataclass
class GroupPrompts:
    case: PreparedCase
    system: str
    x: list[str]                 # user texts (G copies of the same rendering)
    xp: list[str]
    xm: list[str]
    transform: str
    gold: int | None
    added_loglik: np.ndarray | None
    codes: list[str] = field(default_factory=list)


class PairedSampler:
    def __init__(self, dataset: str, seed: int, transforms: dict[str, float] | None = None,
                 contrast: str = "triage", label_free: bool = False, pool_path: str | None = None,
                 phrasebank_path: str | None = None):
        self.dataset = dataset
        self.r = random.Random(seed)
        self.seed = seed
        self.contrast = contrast
        self.label_free = label_free
        tw = transforms or {t: 1.0 for t in TRAIN_TRANSFORMS}
        self.transforms = [t for t, w in tw.items() if w > 0]
        self.weights = [tw[t] for t in self.transforms]
        pb = load_phrasebank(paths.FROZEN_BANKS / "ddx_phrasebank.json") if phrasebank_path is None \
            else load_phrasebank(__import__("pathlib").Path(phrasebank_path))
        if pb is None and "para" in self.transforms:
            i = self.transforms.index("para")
            self.transforms.pop(i)
            self.weights.pop(i)
        if dataset == "ddxplus":
            src = pool_path or str(paths.DATA / "train_pool" / "ddxplus_train.jsonl")
            self.records = read_jsonl(__import__("pathlib").Path(src))
            self.make = lambda rec: ddx_prepared(rec, pb)
            self.lik = dx.load_likelihood()
            kb = dx.load_kb()
            self.system = T.system_prompt("posterior", kb.candidates(), fixed=True)
        else:
            rows = read_jsonl(paths.FROZEN_MCR / "prepared.jsonl")
            self.records = [r for r in rows if r.get("role") == "train" and r.get("status") == "ok"]
            self.make = mcr_prepared
            self.lik = None
            self.system = T.system_prompt("posterior", None, fixed=False)
        if not self.records:
            raise SystemExit(f"no training records for {dataset}")
        self.order: list[int] = []

    def _next(self) -> dict:
        if not self.order:
            self.order = list(range(len(self.records)))
            self.r.shuffle(self.order)
        return self.records[self.order.pop()]

    def _render(self, case: PreparedCase, label: str, transform: str | None) -> str:
        seed = self.r.randrange(2**31)
        o = I._ordered(case, f"{label}.{seed}", self.seed)
        user_c = None if case.candidates_fixed else case.candidates
        if transform is None:
            return T.user_prompt(I.render(case, o), user_c)
        if transform == "perm":
            o2 = I._ordered(case, f"{label}.perm.{seed}", self.seed)
            return T.user_prompt(I.render(case, o2), user_c)
        if transform == "para":
            return T.user_prompt(I.render(case, o, "para"), user_c)
        if transform == "dup":
            free = [i for i, f in enumerate(o) if not f.fixed]
            i = self.r.choice(free)
            pos = self.r.choice(list(range(i + 1, len(o))) or [i])
            return T.user_prompt(I.render(case, o, inserts=[(pos, I.dup_sentence(o[i].base))]), user_c)
        if transform == "cue":
            bank = [c for c in config.cues()["irrelevant"] if I.cue_ok(c["text"], case.candidates)]
            cue = self.r.choice(bank)
            return T.user_prompt(I.render(case, o, inserts=[(self.r.randrange(len(o)), cue["text"])]), user_c)
        raise ValueError(transform)

    def _contrast(self, case: PreparedCase) -> tuple[str, list[str]]:
        user_c = None if case.candidates_fixed else case.candidates
        o = I._ordered(case, f"contrast.{self.r.randrange(2**31)}", self.seed)
        if self.contrast == "triage":
            keep = [f for f in o if f.fixed]
        else:
            free = [f for f in o if not f.fixed]
            drop = self.r.choice(free).fid
            keep = [f for f in o if f.fid != drop]
        removed = [t for f in case.findings if f not in keep for t in f.meta.get("tokens", [])]
        return T.user_prompt(I.render(case, keep), user_c), removed

    def sample(self, n_patients: int, G: int, H: int) -> list[GroupPrompts]:
        out = []
        for _ in range(n_patients):
            rec = self._next()
            case = self.make(rec)
            t = self.r.choices(self.transforms, self.weights)[0]
            x = self._render(case, "x", None)
            xp = self._render(case, "xp", t)
            xm, removed = self._contrast(case)
            gold = None if self.label_free else case.codes.index(case.gold)
            added = None
            if self.lik is not None and removed:
                added = self.lik.tau * self.lik.unit_loglik(removed)
            out.append(GroupPrompts(case, self.system, [x] * G, [xp] * G, [xm] * H, t, gold, added, case.codes))
        return out
