"""Command-line interface: `ccrl <command>` (or `python -m ccrl <command>`).

Every paid command accepts --dry-run, which uses the offline mock client and
writes only under runs/dryrun/. Start with `ccrl rehearse` — a complete dry
run of the whole pipeline in a few minutes, at zero cost.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

from ccrl import config, paths
from ccrl.workspace import get as get_ws


def _csv(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def cmd_doctor(a) -> int:
    from ccrl import doctor
    return doctor.run(get_ws(a.dry_run), a.profile, a.region, a.offline)


def cmd_data(a) -> int:
    if a.action == "download":
        from ccrl.data import download
        (download.download_ddxplus if a.dataset == "ddxplus" else download.download_medcase)()
    elif a.action == "build-ddxplus":
        from ccrl.data import build_ddx
        build_ddx.build()
    elif a.action == "fit-reference":
        from ccrl.data import build_ddx
        build_ddx.fit_reference()
    elif a.action == "select-medcase":
        from ccrl.data import medcase
        medcase.select()
    return 0


def cmd_models(a) -> int:
    from ccrl.bedrock import models as M
    ws = get_ws(a.dry_run)
    region = a.region or config.models()[config.panel_models()[0]].region
    if a.action == "list":
        if a.dry_run:
            print("dry run: no live listing")
            return 0
        live = M.list_live(region, a.profile)
        for m in sorted(live["foundation_models"], key=lambda x: (x["provider"] or "", x["id"])):
            print(f"model    {m['provider'] or '':<14} {m['id']:<55} {','.join(m['inference'] or [])} {m['status']}")
        for p in sorted(live["inference_profiles"], key=lambda x: x["id"]):
            print(f"profile  {'':<14} {p['id']:<55} {p['status']}")
        return 0
    if a.action == "resolve":
        if a.dry_run:
            print("dry run: models resolve to mock.<key>")
            return 0
        res = M.resolve(M.list_live(region, a.profile))
        path = M.write_resolved(res, region)
        print(f"{'key':<16} {'chosen id':<50} how       alternatives")
        for k, v in res.items():
            print(f"{k:<16} {str(v['model_id']):<50} {str(v['how']):<9} {', '.join(v['alternatives'][1:4])}")
        missing = [k for k, v in res.items() if not v["model_id"]]
        print(f"\nwrote {path}" + (f"\nUNRESOLVED: {missing} — check model access in the Bedrock console" if missing else ""))
        return 1 if any(k in missing for k in config.panel_models()) else 0
    if a.action == "probe":
        keys = _csv(a.models) or list(config.models())
        if a.dry_run:
            ids = {k: f"mock.{k}" for k in keys}
        else:
            res = M.load_resolved()
            if not res:
                raise SystemExit("run `ccrl models resolve` first")
            ids = {k: (res.get(k) or {}).get("model_id") for k in keys}
        doc = M.probe(keys, ws, M.make_client_factory(a.dry_run, a.profile), ids)
        bad = [k for k in keys if not doc["models"].get(k, {}).get("probe", {}).get("ok")]
        print(f"\nprobe: {len(keys) - len(bad)} OK, failed: {bad or 'none'}"
              + ("" if a.dry_run else f"\nwrote {paths.MODELS_LOCK}"))
        return 1 if any(k in bad for k in config.panel_models()) else 0
    return 0


def cmd_prep(a) -> int:
    ws = get_ws(a.dry_run)
    if a.action == "phrasebank":
        from ccrl.prep import phrasebank
        phrasebank.run(ws, a.profile, a.max_spend)
    elif a.action == "mcr":
        from ccrl.prep import medcase
        medcase.run(ws, a.stage, _csv(a.roles) or ["pilot", "panel", "train"], a.profile, a.max_spend)
    elif a.action == "finalize":
        from ccrl.prep import medcase, phrasebank
        phrasebank.finalize(ws)
        medcase.finalize(ws)
    return 0


def cmd_build(a) -> int:
    from ccrl.panel import items
    items.build(get_ws(a.dry_run), allow_missing_para=a.allow_missing_para, allow_short_mcr=a.allow_short_mcr)
    return 0


def cmd_freeze(a) -> int:
    from ccrl.panel import freeze
    ws = get_ws(a.dry_run)
    if a.check:
        ok, msg = freeze.check_freeze(ws)
        print(("OK: " if ok else "CHANGED: ") + msg)
        return 0 if ok else 1
    doc = freeze.freeze(ws)
    print(f"froze {len(doc['files'])} files -> {ws.freeze_file}")
    return 0


def cmd_plan(a) -> int:
    from ccrl.panel import qa
    print(qa.forecast_text(qa.forecast_all(get_ws(a.dry_run))))
    return 0


def cmd_run(a) -> int:
    from ccrl.panel import jobs
    s = jobs.run_job(a.job, get_ws(a.dry_run), _csv(a.models), a.limit_cases, a.max_spend, a.force_order,
                     a.override_c_gate, a.profile)
    return 0 if not s.get("fatal_run") else 2


def cmd_status(a) -> int:
    from ccrl.panel import qa
    print(qa.status(get_ws(a.dry_run)))
    return 0


def cmd_report(a) -> int:
    from ccrl.panel import qa
    ws = get_ws(a.dry_run)
    rep = qa.pilot_report(ws)
    print((ws.reports / "pilot_report.txt").read_text(encoding="utf-8"))
    return 0 if rep["go"] else 1


def cmd_export(a) -> int:
    from ccrl.archive import export
    export.export(get_ws(a.dry_run))
    return 0


def cmd_archive(a) -> int:
    from ccrl.archive import export
    export.pack(get_ws(a.dry_run))
    return 0


def cmd_billing(a) -> int:
    from ccrl import billing
    return billing.reconcile(get_ws(a.dry_run), a.start, a.end, a.profile)


def cmd_analyze(a) -> int:
    from ccrl.analysis import report
    out = report.run(get_ws(a.dry_run), jobs=_csv(a.jobs))
    print(f"analysis written to {out}")
    return 0


def cmd_rehearse(a) -> int:
    """Full zero-cost dry run of the whole Track A pipeline."""
    from ccrl.analysis import report
    from ccrl.archive import export
    from ccrl.bedrock import models as M
    from ccrl.panel import freeze, items, jobs, qa
    from ccrl.prep import medcase, phrasebank

    dry = paths.RUNS / "dryrun"
    if dry.exists():
        shutil.rmtree(dry)
    ws = get_ws(True)
    n = a.cases
    steps = []

    def step(name, fn):
        print(f"\n=== rehearse: {name} ===", flush=True)
        r = fn()
        steps.append(name)
        return r

    step("probe", lambda: M.probe(list(config.models()), ws, M.make_client_factory(True),
                                  {k: f"mock.{k}" for k in config.models()}, verbose=False))
    step("prep phrasebank", lambda: phrasebank.run(ws))
    step("prep mcr (pilot+panel)", lambda: medcase.run(ws, "all", ["pilot", "panel"]))
    step("prep finalize", lambda: (phrasebank.finalize(ws), medcase.finalize(ws)))
    step("build items", lambda: items.build(ws, allow_short_mcr=True))
    step("freeze", lambda: freeze.freeze(ws))
    step("plan", lambda: print(qa.forecast_text(qa.forecast_all(ws))))
    step("run pilot", lambda: jobs.run_job("pilot", ws, limit_cases=n))
    rep = step("report pilot", lambda: qa.pilot_report(ws))
    verdict = "GO" if rep["go"] else "NO-GO"
    print(f"pilot gates on MOCK data: {verdict} {rep['failures'][:5]}")
    print("(the mock deliberately injects ~1% truncations, timeouts and malformed answers, so a NO-GO")
    print(" here is expected: it shows the gates work. Only the real pilot's verdict matters.)")
    step("run panel_main (limited)", lambda: jobs.run_job("panel_main", ws, limit_cases=n, force_order=True))
    step("run effort_arm (limited)", lambda: jobs.run_job("effort_arm", ws, limit_cases=max(2, n // 4), force_order=True))
    step("run sequential (limited)", lambda: jobs.run_job("sequential", ws, limit_cases=max(2, n // 4), force_order=True))
    step("rerun panel_main (must be a no-op)", lambda: jobs.run_job("panel_main", ws, limit_cases=n, force_order=True))
    step("status", lambda: print(qa.status(ws)))
    step("analyze", lambda: report.run(ws))
    step("archive pack", lambda: export.pack(ws))
    print(f"\nREHEARSAL PASSED ({len(steps)} steps). Everything is under {dry} — delete it any time.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ccrl", description="Coherence-as-a-Reward study harness")
    p.add_argument("--dry-run", action="store_true", help="offline mock client; writes only under runs/dryrun/")
    p.add_argument("--profile", default=None, help="AWS named profile (default: standard credential chain)")
    p.add_argument("--region", default=None, help="override region for listing/doctor (models.yaml sets per-model)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="check environment, config, credentials (no cost)")
    s.add_argument("--offline", action="store_true")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("data", help="raw corpora (not needed to run the panel)")
    s.add_argument("action", choices=["download", "build-ddxplus", "fit-reference", "select-medcase"])
    s.add_argument("dataset", nargs="?", choices=["ddxplus", "medcase"], default="ddxplus")
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("models", help="list / resolve / probe Bedrock models")
    s.add_argument("action", choices=["list", "resolve", "probe"])
    s.add_argument("--models", help="comma-separated registry keys (probe)")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("prep", help="paid preprocessing (phrase bank, MedCaseReasoning) and finalisation")
    s.add_argument("action", choices=["phrasebank", "mcr", "finalize"])
    s.add_argument("--stage", choices=["extract", "verify", "all"], default="all")
    s.add_argument("--roles", help="mcr roles: pilot,panel,train (default all)")
    s.add_argument("--max-spend", type=float, default=None)
    s.set_defaults(fn=cmd_prep)

    s = sub.add_parser("build", help="build the frozen items (no cost)")
    s.add_argument("what", choices=["items"])
    s.add_argument("--allow-missing-para", action="store_true")
    s.add_argument("--allow-short-mcr", action="store_true")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("freeze", help="hash all inputs; --check to verify")
    s.add_argument("--check", action="store_true")
    s.set_defaults(fn=cmd_freeze)

    s = sub.add_parser("plan", help="cost forecast for every job (no cost)")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("run", help="run a paid job from config/jobs.yaml")
    s.add_argument("job")
    s.add_argument("--models", help="restrict to these registry keys")
    s.add_argument("--limit-cases", type=int, default=None, help="only the first N cases (testing)")
    s.add_argument("--max-spend", type=float, default=None, help="extra USD cap for this invocation")
    s.add_argument("--force-order", action="store_true", help="ignore requires_completed")
    s.add_argument("--override-c-gate", action="store_true", help="PI-approved override of the priority-C gate")
    s.set_defaults(fn=cmd_run)

    for name, fn, h in (("status", cmd_status, "spend and progress"), ("export", cmd_export, "ledger -> parquet"),
                        ("plan", cmd_plan, "forecast")):
        if name == "plan":
            continue
        s = sub.add_parser(name, help=h)
        s.set_defaults(fn=fn)

    s = sub.add_parser("report", help="pilot QA report and gates")
    s.add_argument("which", choices=["pilot"])
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("archive", help="pack the archive tarball")
    s.add_argument("action", choices=["pack"])
    s.set_defaults(fn=cmd_archive)

    s = sub.add_parser("billing", help="reconcile with AWS Cost Explorer")
    s.add_argument("action", choices=["reconcile"])
    s.add_argument("--start", required=True)
    s.add_argument("--end", default=None)
    s.set_defaults(fn=cmd_billing)

    s = sub.add_parser("analyze", help="offline estimands, tables and figures")
    s.add_argument("--jobs", help="comma-separated jobs (default: all panel jobs present)")
    s.set_defaults(fn=cmd_analyze)

    from ccrl.train.cli import add_parser as add_train
    add_train(sub)

    s = sub.add_parser("rehearse", help="complete zero-cost dry run of the pipeline")
    s.add_argument("--cases", type=int, default=12)
    s.set_defaults(fn=cmd_rehearse)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    if args.cmd == "rehearse":
        args.dry_run = True
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
