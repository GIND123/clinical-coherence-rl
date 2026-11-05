"""Prompt assembly from the frozen templates in config/prompts/.

Templates are plain text with ``{placeholder}`` markers filled by literal
replacement (never ``str.format``: the templates contain JSON braces).
Everything model-independent is decided here, so every model in the panel
receives byte-identical text for a given item.

Layout for caching: for DDXPlus the 49-candidate list lives in the SYSTEM
block (identical for every DDXPlus request -> cacheable prefix above the
512-token minimum). MedCaseReasoning candidate lists are case-specific and
go in the user message.
"""
from __future__ import annotations

from functools import lru_cache

from ccrl import paths
from ccrl.util import sha256_bytes

PLACEHOLDER_KEYS = ("candidate_block", "candidates_section", "vignette", "authority_section",
                    "case_prompt", "final_diagnosis", "final_diagnosis_raw", "extraction_json", "items_json")


@lru_cache(maxsize=None)
def load(name: str) -> str:
    with open(paths.PROMPTS / f"{name}.txt", "r", encoding="utf-8") as f:
        return f.read().rstrip("\n")


def fill(template: str, **values: str) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", v)
    return out


def template_hashes() -> dict[str, str]:
    return {p.stem: sha256_bytes(p.read_bytes()) for p in sorted(paths.PROMPTS.glob("*.txt"))}


def candidate_block(candidates: list[tuple[str, str]]) -> str:
    return "\n".join(f"{code}  {name}" for code, name in candidates)


def system_prompt(kind: str, candidates: list[tuple[str, str]] | None, fixed: bool) -> str:
    """kind: 'posterior' or 'scores'. fixed=True puts the list in the system block."""
    if fixed:
        return fill(load(f"{kind}_system_fixed"), candidate_block=candidate_block(candidates or []))
    return load(f"{kind}_system_case")


def user_prompt(vignette: str, candidates: list[tuple[str, str]] | None = None,
                authority: str | None = None, template: str = "user_single") -> str:
    cand = ""
    if candidates:
        cand = "Candidate diagnoses:\n" + candidate_block(candidates) + "\n\n"
    auth = f"\n\n{authority}" if authority else ""
    return fill(load(template), candidates_section=cand, vignette=vignette, authority_section=auth)
