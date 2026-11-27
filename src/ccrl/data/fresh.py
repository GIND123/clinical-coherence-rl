"""Amendment 3: real case reports published AFTER every panel model was frozen.

Every panel model was live on Bedrock with the exact model IDs in config/models.lock.yaml by
6 Nov 2026 (the probe). A case report first published on or after 7 Nov 2026 therefore cannot be
in any panel model's training data, whatever the model's cutoff.

    ccrl data fresh-select      PubMed Central -> artifacts/frozen/fresh/cases_selected.jsonl ($0)

Source: PMC, publication type "Case Reports", open access under CC BY (redistributable with
attribution; each row keeps its licence and link), electronic publication date 7-27 Nov 2026.
From the full text we keep only the case-presentation section (verbatim). Selection is
deterministic: eligible articles in a seeded hash order, the first `n_prep` kept.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from ccrl import config, paths
from ccrl.util import write_json, write_jsonl

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
QUERY = '"case reports"[pt] AND "open access"[filter] AND "cc by license"[filter] AND {start}:{end}[pdat]'
FIRST_DAY = "2026-11-07"           # day after the probe: every panel model already frozen
LAST_DAY = "2026-11-27"
N_PREP = None                      # all eligible cases go to preprocessing; the first 150 that pass are used
N_USE = 150
MIN_CHARS, MAX_CHARS = 700, 6000
CASE_SEC = re.compile(r"(\bcase\b.*\b(presentation|report|description|history|summary|details)\b|^\s*case\s*(report|presentation|description)?\s*$|"
                      r"clinical\s+(presentation|case|findings)|patient\s+(information|presentation)|presentation of (the )?case)", re.I)
MULTI = re.compile(r"\b(cases|case series|two patients|three patients|patients|siblings|twins|family)\b", re.I)
MULTI_TEXT = re.compile(r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(cases|patients)\b"
                        r"|\bcases?\s+[1-9]\s+(and|to|-)\s+[1-9]\b"
                        r"|\b(case|patient)\s+(2|ii|two|b)\b", re.I)
SECOND_CASE = re.compile(r"^\s*(case|patient)\s*(2|ii|two|b)\b", re.I)
CITE = re.compile(r"\s*\[(\d+([,–\-]\s*\d+)*)\]")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\d])")
XLINK = "{http://www.w3.org/1999/xlink}href"


def _get(url: str, params: dict, tries: int = 4) -> bytes:
    q = urllib.parse.urlencode({**params, "tool": "ccrl"})
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": "ccrl/1.0"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    return b""


def search(start: str = FIRST_DAY, end: str = LAST_DAY) -> list[str]:
    term = QUERY.format(start=start.replace("-", "/"), end=end.replace("-", "/"))
    js = json.loads(_get(f"{EUTILS}/esearch.fcgi", {"db": "pmc", "term": term, "retmax": 5000, "retmode": "json"}))
    return js["esearchresult"]["idlist"]


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def _date(art) -> str | None:
    best = None
    for d in art.iter("pub-date"):
        kind = d.get("pub-type") or d.get("date-type") or ""
        fmt = d.get("publication-format") or ""
        if kind in ("epub", "pub") or fmt == "electronic" or kind == "":
            y, m, dd = (d.findtext(x) for x in ("year", "month", "day"))
            if y:
                iso = f"{int(y):04d}-{int(m or 1):02d}-{int(dd or 1):02d}"
                if kind == "epub" or fmt == "electronic":
                    return iso
                best = best or iso
    return best


def _license(art) -> str:
    out = []
    for lic in art.iter("license"):
        out.append(lic.get(XLINK) or "")
        out.extend(x.text or "" for x in lic.iter() if x.tag.endswith("license_ref"))
        out.append(lic.get("license-type") or "")
    return " ".join(out)


def _case_section(art):
    body = art.find("body")
    if body is None:
        return None, 0
    secs = list(body.iter("sec"))
    titles = [_text(s.find("title")) for s in secs]
    n_second = sum(bool(SECOND_CASE.match(t)) for t in titles)
    for s, t in zip(secs, titles):
        if CASE_SEC.search(t) and not re.search(r"discussion|conclusion|review", t, re.I):
            return s, n_second
    return None, n_second


def _section_text(sec) -> str:
    skip = set()
    for tag in ("fig", "table-wrap", "caption", "disp-formula", "supplementary-material"):
        for el in sec.iter(tag):
            skip.update(id(x) for x in el.iter())
    paras = [CITE.sub("", _text(p)) for p in sec.iter("p") if id(p) not in skip]
    return "\n\n".join(p for p in paras if p)


def parse(xml_bytes: bytes) -> list[dict]:
    out = []
    root = ET.fromstring(xml_bytes)
    for art in root.iter("article"):
        meta = art.find("front/article-meta")
        if meta is None:
            continue
        ids = {i.get("pub-id-type"): (i.text or "").strip() for i in meta.findall("article-id")}
        pmcid = ids.get("pmcid") or ids.get("pmc") or ""
        pmcid = pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"
        sec, n_second = _case_section(art)
        text = _section_text(sec) if sec is not None else ""
        out.append({
            "pmcid": pmcid, "title": _text(meta.find("title-group/article-title")),
            "journal": _text(art.find("front/journal-meta/journal-title-group/journal-title")),
            "publication_date": _date(meta), "license": _license(meta), "article_type": art.get("article-type"),
            "case_text": text, "n_second_case": n_second,
            "article_link": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"})
    return out


def eligible(r: dict) -> str | None:
    """None if eligible, else the reason."""
    if not r["publication_date"] or r["publication_date"] < FIRST_DAY:
        return "published before 7 Nov 2026"
    if "creativecommons.org/licenses/by/" not in r["license"] and "cc by" not in r["license"].lower():
        return "not CC BY"
    if not r["case_text"]:
        return "no case-presentation section"
    if MULTI.search(r["title"]) or r["n_second_case"] or MULTI_TEXT.search(r["case_text"]):
        return "more than one patient"
    if len(r["case_text"]) < MIN_CHARS:
        return f"case section shorter than {MIN_CHARS} characters"
    if len(r["case_text"]) > MAX_CHARS:
        return f"case section longer than {MAX_CHARS} characters"
    return None


def sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n\n"):
        out.extend(s.strip() for s in SENT.split(para) if s.strip())
    return out


def select(n_prep: int | None = N_PREP) -> dict:
    seed = int(config.study()["seed"])
    ids = search()
    rows = []
    for k in range(0, len(ids), 50):
        rows.extend(parse(_get(f"{EUTILS}/efetch.fcgi", {"db": "pmc", "id": ",".join(ids[k:k + 50]),
                                                         "retmode": "xml"})))
        time.sleep(0.4)
    reasons: dict[str, int] = {}
    ok = []
    for r in rows:
        why = eligible(r)
        if why:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            ok.append(r)
    ok.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['pmcid']}".encode()).hexdigest())
    chosen = []
    for k, r in enumerate(ok if n_prep is None else ok[:n_prep]):
        chosen.append({"case_id": f"fresh-{r['pmcid']}", "dataset": "medcase", "role": "fresh", "prep_order": k,
                       **{x: r[x] for x in ("pmcid", "title", "journal", "publication_date", "license",
                                            "article_link", "case_text")},
                       "sentences": sentences(r["case_text"])})
    d = paths.FROZEN / "fresh"
    d.mkdir(parents=True, exist_ok=True)
    write_jsonl(d / "cases_selected.jsonl", chosen)
    report = {"query": QUERY.format(start=FIRST_DAY, end=LAST_DAY), "window": [FIRST_DAY, LAST_DAY],
              "pmc_hits": len(ids), "parsed": len(rows), "eligible": len(ok), "selected": len(chosen),
              "excluded": reasons, "seed": seed, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(d / "selection_report.json", report)
    return report
