"""Download the raw corpora with integrity checks.

    ccrl data download ddxplus     # figshare article 22687585, md5-checked
    ccrl data download medcase     # HF zou-lab/MedCaseReasoning @ pinned revision, sha256-checked

Only needed to REBUILD the frozen inputs or to train (Track B). Running the
Bedrock panel needs neither: the frozen inputs are committed.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from ccrl import config, paths

FIGSHARE_URL = "https://ndownloader.figshare.com/files/{fid}"
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/data/{name}"
# Observed at the pinned revision 469a5365bc534b5b2b9cfbc52b2ef2a10f43a339.
MEDCASE_SHA256 = {
    "test-00000-of-00001.parquet": "353b33113cd504a870430f59d4149f1cc9e1c5561523ba745031719548b1c643",
    "train-00000-of-00001.parquet": "12b23b1d652caff931ea53012e8b84c2e74bfd3c6394c4fcdea63afa74158207",
    "val-00000-of-00001.parquet": "442a60139b8b8d0526690ce3773c47c654e19a979cb95e76750eabc0e0e53b4d",
}


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ccrl-data-downloader"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)


def download_ddxplus(dest: Path | None = None, extract: bool = True) -> Path:
    dest = Path(dest or paths.DATA_DDX)
    files = config.study()["ddxplus"]["files"]
    for name, (fid, md5) in files.items():
        p = dest / name
        if p.exists() and _hash(p, "md5") == md5:
            print(f"[ok] {name}")
            continue
        print(f"[get] {name}")
        _fetch(FIGSHARE_URL.format(fid=fid), p)
        got = _hash(p, "md5")
        if got != md5:
            raise ValueError(f"md5 mismatch for {name}: {got} != {md5}")
    if extract:
        out = dest / "extracted"
        out.mkdir(exist_ok=True)
        for name in files:
            if name.endswith(".zip"):
                target = out / name[:-4]
                if not target.exists():
                    print(f"[unzip] {name}")
                    with zipfile.ZipFile(dest / name) as z:
                        z.extractall(out)
    return dest


def download_medcase(dest: Path | None = None) -> Path:
    dest = Path(dest or paths.DATA_MCR)
    cfg = config.study()["medcase"]
    for name, sha in MEDCASE_SHA256.items():
        p = dest / name
        if not p.exists():
            print(f"[get] {name}")
            _fetch(HF_URL.format(repo=cfg["hf_repo"], rev=cfg["hf_revision"], name=name), p)
        got = _hash(p, "sha256")
        if sha and got != sha:
            raise ValueError(f"sha256 mismatch for {name}: {got} != {sha}")
        print(f"[ok] {name} sha256={got[:16]}...")
    return dest
