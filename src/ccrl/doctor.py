"""`ccrl doctor`: everything that can be checked without spending money.

Exit status 0 = no blocking problem. Warnings do not block.
"""
from __future__ import annotations

import platform
import shutil
import sys

from ccrl import config, paths
from ccrl.util import local_now, parse_local, read_json
from ccrl.workspace import Workspace

PINNED = {"boto3": "1.43.124", "botocore": "1.43.124"}


def run(ws: Workspace, profile: str | None = None, region: str | None = None, offline: bool = False) -> int:
    ok_all = True

    def line(level: str, msg: str) -> None:
        nonlocal ok_all
        if level == "FAIL":
            ok_all = False
        print(f"[{level:4}] {msg}", flush=True)

    v = sys.version_info
    line("OK" if v >= (3, 10) else "FAIL", f"Python {platform.python_version()} ({sys.executable})")
    try:
        import importlib.metadata as md
        for pkg, want in PINNED.items():
            have = md.version(pkg)
            line("OK" if have == want else "WARN", f"{pkg} {have}" + ("" if have == want else
                 f" (tested with {want}; run: pip install -r requirements/panel.txt)"))
        for pkg in ("numpy", "pandas", "pyarrow", "PyYAML", "tqdm"):
            line("OK", f"{pkg} {md.version(pkg)}")
    except Exception as e:
        line("FAIL", f"package check: {e}")

    probs = config.validate_all()
    line("OK" if not probs else "FAIL", "config files valid" if not probs else "config problems: " + "; ".join(probs))

    # frozen inputs
    try:
        rep = read_json(paths.FROZEN_DDX / "build_report.json")["counts"]
        want = config.study()["ddxplus"]
        good = rep["panel"] == 300 and rep["panel_ext"] == 300 and rep["pilot"] == want["pilot_cases"] \
            and rep["distill"] == want["distill_cases"]
        line("OK" if good else "FAIL", f"DDXPlus frozen inputs: {rep}")
        for f in ("kb.json", "likelihood.json", "cases.jsonl", "heldout.json"):
            if not (paths.FROZEN_DDX / f).exists():
                line("FAIL", f"missing artifacts/frozen/ddxplus/{f}")
    except FileNotFoundError:
        line("FAIL", "DDXPlus frozen inputs missing (artifacts/frozen/ddxplus)")
    sel = paths.FROZEN_MCR / "cases_selected.jsonl"
    if sel.exists():
        from collections import Counter
        from ccrl.util import read_jsonl
        got = Counter(r["role"] for r in read_jsonl(sel))
        mc = config.study()["medcase"]
        want_m = {"pilot": mc["pilot_prep"], "panel": mc["panel_prep"], "train": mc["train_prep"]}
        line("OK" if dict(got) == want_m else "FAIL", f"MedCaseReasoning selection: {dict(got)}"
             + ("" if dict(got) == want_m else f" (study.yaml expects {want_m}: git pull)"))
    else:
        line("FAIL", "MedCaseReasoning selection MISSING")
    line("OK" if ws.phrasebank.exists() else "INFO",
         f"phrase bank {'present' if ws.phrasebank.exists() else 'not built yet (ccrl prep phrasebank)'}")
    line("OK" if ws.mcr_prepared.exists() else "INFO",
         f"MedCaseReasoning prepared cases {'present' if ws.mcr_prepared.exists() else 'not built yet (ccrl prep mcr)'}")
    items = ws.items_dir / "items.jsonl"
    line("OK" if items.exists() else "INFO", f"items {'built' if items.exists() else 'not built yet (ccrl build items)'}")
    from ccrl.panel.freeze import check_freeze
    fok, fmsg = check_freeze(ws)
    line("OK" if fok else "INFO", f"freeze: {fmsg}")

    # models / prices
    from ccrl.bedrock.models import lock_doc
    ld = lock_doc()
    if ld:
        from ccrl.bedrock.models import stale_lock_models
        okm = {k for k, m in ld["models"].items() if m.get("probe", {}).get("ok")}
        panel = config.panel_models()
        roles = set(config._models_raw()["roles"].values())
        need = [k for k in panel if k not in okm] + [k for k in sorted(roles) if k not in okm and k not in panel]
        line("OK" if not need else "FAIL", f"model lock: {len([k for k in panel if k in okm])}/{len(panel)} panel "
             f"models probed OK" + (f"; missing or failed: {need}" if need else ""))
        opt_ok = [k for k in config.optional_models() if k in okm]
        line("INFO", f"optional partner models available: {opt_ok or 'none'} (only the exploratory partner_panel uses them)")
        stale = stale_lock_models(ld)
        if stale:
            line("FAIL", f"probe results are stale for {stale} (registry changed): ccrl models probe --models {','.join(stale)}")
    else:
        line("INFO", "model lock not created yet (ccrl models resolve; ccrl models probe)")
    unv = [k for k, p in config.prices()["prices"].items() if not p.get("verified") and k != "mock"]
    line("WARN" if unv else "OK", f"prices verified for all models" if not unv else
         f"{len(unv)} price rows not verified against the AWS pricing page: {', '.join(unv)}")

    # time
    hs = parse_local(config.study()["dates"]["hard_stop"])
    left = hs - local_now()
    line("OK" if left.total_seconds() > 0 else "FAIL",
         f"hard stop {hs.isoformat(timespec='minutes')} ({left.days} days left)" if left.total_seconds() > 0
         else f"hard stop {hs} has passed: no paid calls allowed")

    free = shutil.disk_usage(str(paths.ROOT)).free / 2**30
    line("OK" if free > 5 else "WARN", f"free disk {free:.1f} GiB (raw archive needs ~1-3 GiB)")

    if ws.ledger.exists():
        from ccrl.bedrock.ledger import Ledger
        led = Ledger(ws.ledger)
        try:
            line("OK", f"ledger {ws.ledger}: committed ${led.committed():.2f}")
        finally:
            led.close()

    if offline or ws.dry_run:
        line("INFO", "AWS checks skipped (offline/dry-run)")
        return 0 if ok_all else 1
    region = region or config.models()[config.panel_models()[0]].region
    try:
        import boto3
        s = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)
        ident = s.client("sts").get_caller_identity()
        line("OK", f"AWS identity {ident['Arn']} (account {ident['Account']}), region {region}")
        n = len(s.client("bedrock").list_foundation_models().get("modelSummaries", []))
        line("OK", f"Bedrock reachable: {n} foundation models listed")
    except Exception as e:
        line("FAIL", f"AWS check failed: {type(e).__name__}: {e}")
    return 0 if ok_all else 1
