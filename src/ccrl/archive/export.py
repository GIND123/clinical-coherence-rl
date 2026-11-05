"""Export the ledger to analysis-ready, archivable files; pack the archive.

    ccrl export          runs/export/{requests,attempts,posteriors}.parquet + .jsonl.gz + MANIFEST.json
    ccrl archive pack    runs/archive/ccrl-archive-<timestamp>.tar.gz (+ .sha256)

The archive holds everything needed to redo every analysis offline, with no
further paid call: raw request and response JSON per attempt, usage and
cost, parsed posteriors, the frozen inputs, configs, the model lock, the
freeze manifest, and a SQLite backup of the ledger.
"""
from __future__ import annotations

import gzip
import json
import tarfile
from pathlib import Path

import pandas as pd

from ccrl import paths
from ccrl.bedrock.ledger import Ledger
from ccrl.panel import items as I
from ccrl.util import local_now, now_iso, sha256_file, write_json
from ccrl.workspace import Workspace

PANEL_JOBS = ("pilot", "panel_main", "effort_arm", "sequential", "distill", "astra_ceiling", "panel_ext")


def _df(ledger: Ledger, sql: str) -> pd.DataFrame:
    rows = ledger.q(sql)
    return pd.DataFrame([dict(r) for r in rows])


def posterior_table(ws: Workspace, ledger: Ledger | None = None) -> pd.DataFrame:
    """One row per (request, turn) of the FINAL attempt of every panel request."""
    own = ledger is None
    ledger = ledger or Ledger(ws.ledger)
    try:
        df = _df(ledger, "SELECT r.request_id, r.job, r.model_key, r.model_id, r.item_id, r.case_id, r.transform, "
                         "r.dataset, r.effort, r.status AS request_status, r.n_attempts, a.attempt, a.turn, "
                         "a.status AS attempt_status, a.stop_reason, a.served_tier, a.input_tokens, a.output_tokens, "
                         "a.cache_read, a.cost_reported, a.latency_ms, a.parsed_json FROM requests r JOIN attempts a "
                         "ON a.request_id=r.request_id AND a.attempt=r.final_attempt "
                         f"WHERE r.job IN ({','.join(repr(j) for j in PANEL_JOBS)})")
    finally:
        if own:
            ledger.close()
    if df.empty:
        return df
    parsed = df["parsed_json"].map(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    df["parse_ok"] = parsed.map(lambda p: bool(p.get("ok")))
    df["informative"] = parsed.map(lambda p: bool(p.get("informative")))
    df["probs"] = parsed.map(lambda p: p.get("probs"))
    df["dx"] = parsed.map(lambda p: p.get("dx"))
    df["raw_sum"] = parsed.map(lambda p: p.get("raw_sum"))
    df["n_missing"] = parsed.map(lambda p: p.get("n_missing"))
    df["flags"] = parsed.map(lambda p: ",".join(p.get("flags", [])))
    df["parse_error"] = parsed.map(lambda p: p.get("error"))
    try:
        items = I.load_items(ws)
        df["gold"] = df["item_id"].map(lambda i: items.get(i, {}).get("gold"))
    except Exception:
        df["gold"] = None
    return df.drop(columns=["parsed_json"])


def export(ws: Workspace) -> dict:
    out = ws.export
    out.mkdir(parents=True, exist_ok=True)
    led = Ledger(ws.ledger)
    try:
        req = _df(led, "SELECT * FROM requests")
        att = _df(led, "SELECT * FROM attempts")
        ev = _df(led, "SELECT * FROM events")
        post = posterior_table(ws, led)
        backup = led.backup(out / "ledger_backup.sqlite")
    finally:
        led.close()
    files = {}
    for name, df in (("requests", req), ("attempts", att), ("events", ev), ("posteriors", post)):
        p = out / f"{name}.parquet"
        if df.empty:
            df = pd.DataFrame({"empty": []})
        df.to_parquet(p, index=False)
        files[p.name] = p
        g = out / f"{name}.jsonl.gz"
        with gzip.open(g, "wt", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        files[g.name] = g
    files[backup.name] = backup
    manifest = {"created_at": now_iso(), "dry_run": ws.dry_run,
                "rows": {"requests": len(req), "attempts": len(att), "posteriors": len(post)},
                "files": {k: sha256_file(v) for k, v in sorted(files.items())}}
    write_json(out / "MANIFEST.json", manifest)
    print(f"[export] {manifest['rows']} -> {out}")
    return manifest


def pack(ws: Workspace) -> Path:
    export(ws)
    stamp = local_now().strftime("%Y%m%d-%H%M%S")
    dest_dir = ws.runs / "archive"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"ccrl-archive{'-dryrun' if ws.dry_run else ''}-{stamp}.tar.gz"
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(ws.export, arcname="export")
        if ws.raw.exists():
            tar.add(ws.raw, arcname="raw")
        tar.add(paths.CONFIG, arcname="config")
        tar.add(paths.FROZEN, arcname="frozen_committed")
        if ws.dry_run:
            tar.add(ws.frozen, arcname="frozen_workspace")
        if (ws.runs / "reports").exists():
            tar.add(ws.runs / "reports", arcname="reports")
    digest = sha256_file(dest)
    (dest.parent / (dest.name + ".sha256")).write_text(f"{digest}  {dest.name}\n", encoding="utf-8")
    print(f"[archive] {dest}\n[archive] sha256 {digest}")
    return dest
