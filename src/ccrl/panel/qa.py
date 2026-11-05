"""Status, pilot QA gates and the re-forecast (no API calls).

    ccrl status             spend by job x model, request status counts, alerts
    ccrl report pilot       parse / refusal / truncation rates per model and transform,
                            measured token use, cost per call, GO/NO-GO gates, and the
                            forecast of every remaining job from pilot measurements
    ccrl plan               the forecast alone (works before the pilot, from assumptions)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ccrl import config
from ccrl.archive.export import posterior_table
from ccrl.bedrock import models as M
from ccrl.bedrock.ledger import Ledger
from ccrl.panel.forecast import forecast_tasks
from ccrl.panel.jobs import PRIORITY_RANK, plan_tasks
from ccrl.util import now_iso, write_json
from ccrl.workspace import Workspace

GATES = {"parse_rate_min": 0.98, "refusal_rate_max": 0.02, "truncation_rate_max": 0.01,
         "informative_rate_min": 0.90, "uncertain_rate_max": 0.02}


def _fmt_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "(empty)"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))
    return d.to_string(index=False)


def status(ws: Workspace) -> str:
    led = Ledger(ws.ledger)
    try:
        spend = pd.DataFrame(led.spend_table())
        st = pd.DataFrame(led.summary())
        total_guard = led.committed()
        total_rep = led.reported()
        alerts = led.q("SELECT ts, detail FROM events WHERE kind IN ('spend_alert','model_halted') ORDER BY ts")
    finally:
        led.close()
    b = config.budget()
    lines = [f"Ledger: {ws.ledger}",
             f"Committed (guard, conservative): ${total_guard:,.2f} of hard cap ${b['hard_cap_usd']:,.2f} "
             f"(credit ${b['credit_usd']:,.2f})",
             f"Reported (expected AWS bill):     ${total_rep:,.2f}", ""]
    if not st.empty:
        pv = st.pivot_table(index=["job", "model_key"], columns="status", values="n", fill_value=0).reset_index()
        lines += ["Requests by status:", pv.to_string(index=False), ""]
    if not spend.empty:
        lines += ["Spend by job x model (USD):", _fmt_table(spend, "{:.4f}"), ""]
    for a in alerts:
        lines.append(f"[{a['ts']}] {a['detail']}")
    return "\n".join(lines)


def pilot_report(ws: Workspace, write: bool = True) -> dict:
    post = posterior_table(ws)
    post = post[post["job"] == "pilot"] if not post.empty else post
    if post.empty:
        raise SystemExit("No pilot results in the ledger yet: run `ccrl run pilot` first.")
    led = Ledger(ws.ledger)
    try:
        att = pd.DataFrame([dict(r) for r in led.q(
            "SELECT r.model_key, r.transform, a.status, a.output_tokens, a.input_tokens, a.cache_read, "
            "a.cost_reported, a.latency_ms FROM attempts a JOIN requests r USING(request_id) WHERE r.job='pilot'")])
    finally:
        led.close()
    rows = []
    for mk, g in post.groupby("model_key"):
        a = att[att["model_key"] == mk]
        n = len(g)
        billed = a[a["status"].isin(["ok", "truncated", "refused"])]
        rows.append({
            "model": mk, "requests": n,
            "parse_rate": float(g["parse_ok"].mean()),
            "informative_rate": float(g.loc[g["parse_ok"], "informative"].mean()) if g["parse_ok"].any() else 0.0,
            "refusal_rate": float((g["request_status"] == "refused").mean()),
            "truncation_rate": float((a["status"] == "truncated").mean()) if len(a) else 0.0,
            "uncertain_rate": float((a["status"] == "uncertain").mean()) if len(a) else 0.0,
            "attempts_per_request": float(len(a) / max(1, n)),
            "in_tok_median": float(billed["input_tokens"].median()) if len(billed) else np.nan,
            "out_tok_mean": float(billed["output_tokens"].mean()) if len(billed) else np.nan,
            "out_tok_p90": float(billed["output_tokens"].quantile(0.9)) if len(billed) else np.nan,
            "cost_per_request": float(a["cost_reported"].fillna(0).sum() / max(1, n)),
            "latency_s_median": float(billed["latency_ms"].median() / 1000) if len(billed) else np.nan,
        })
    per_model = pd.DataFrame(rows)
    by_tf = post.groupby(["model_key", "transform"])["parse_ok"].mean().unstack(fill_value=np.nan)
    # gates
    fails = []
    for r in rows:
        if r["parse_rate"] < GATES["parse_rate_min"]:
            fails.append(f"{r['model']}: parse rate {r['parse_rate']:.3f} < {GATES['parse_rate_min']}")
        if r["refusal_rate"] > GATES["refusal_rate_max"]:
            fails.append(f"{r['model']}: refusal rate {r['refusal_rate']:.3f}")
        if r["truncation_rate"] > GATES["truncation_rate_max"]:
            fails.append(f"{r['model']}: truncation rate {r['truncation_rate']:.3f} (raise max_tokens)")
        if r["uncertain_rate"] > GATES["uncertain_rate_max"]:
            fails.append(f"{r['model']}: uncertain/timeout rate {r['uncertain_rate']:.3f}")
        if r["informative_rate"] < GATES["informative_rate_min"]:
            fails.append(f"{r['model']}: informative rate {r['informative_rate']:.3f} (uniform answers)")
    # arm-dependent parse failure (calibration C4 of the previous study)
    for mk, s in by_tf.iterrows():
        if s.max() - s.min() > 0.05:
            fails.append(f"{mk}: parse rate differs across transforms by {s.max() - s.min():.3f}")
    fc = forecast_all(ws)
    b = config.budget()
    spent = fc["spent_so_far"]
    ab_high = spent + fc["cumulative"]["A+B"]["high"]
    if ab_high > b["hard_cap_usd"]:
        fails.append(f"forecast A+B (high) ${ab_high:.2f} exceeds hard cap ${b['hard_cap_usd']:.2f}")
    report = {"created_at": now_iso(), "gates": GATES, "go": not fails, "failures": fails,
              "per_model": rows, "parse_rate_by_transform": json.loads(by_tf.to_json()), "forecast": fc}
    if write:
        write_json(ws.reports / "pilot_report.json", report)
        with open(ws.reports / "pilot_report.txt", "w", encoding="utf-8") as f:
            f.write(render_pilot(report, per_model, by_tf))
    return report


def render_pilot(report: dict, per_model: pd.DataFrame, by_tf: pd.DataFrame) -> str:
    out = [f"PILOT REPORT {report['created_at']}", "",
           "Per model:", _fmt_table(per_model), "",
           "Parse rate by transform:", _fmt_table(by_tf.reset_index()), "",
           forecast_text(report["forecast"]), "",
           ("GO: all pilot gates passed." if report["go"] else "NO-GO. Fix before production:\n  - "
            + "\n  - ".join(report["failures"]))]
    return "\n".join(out)


def forecast_all(ws: Workspace) -> dict:
    try:
        specs = M.load_lock(ws.dry_run)
    except SystemExit:
        specs = {}
    led = Ledger(ws.ledger) if ws.ledger.exists() else None
    spent = led.committed() if led else 0.0
    jobs = {}
    try:
        for name, j in config.jobs().items():
            try:
                tasks = plan_tasks(name, ws, specs)
            except (SystemExit, Exception) as e:      # items not built yet, model not probed, ...
                jobs[name] = {"error": str(e).splitlines()[0] if str(e) else type(e).__name__}
                continue
            if led is not None:
                st = led.statuses(name)
                remaining = [t for t in tasks if st.get(t.request_id) not in ("done", "refused")]
                done_cost = led.committed(job=name)
            else:
                remaining, done_cost = tasks, 0.0
            fc = forecast_tasks(remaining, ws, specs)
            jobs[name] = {"priority": j["priority"], "calls_total": sum(len(t.turns) for t in tasks),
                          "calls_remaining": fc["total"]["calls"], "committed": done_cost,
                          **{s: fc["total"][s] for s in ("low", "expected", "high")},
                          "by_model": fc["by_model"], "pilot_measured": fc["pilot_measured"]}
    finally:
        if led is not None:
            led.close()
    cum = {}
    for label, maxrank in (("A", 0), ("A+B", 1), ("A+B+C1", 2), ("A+B+C1+C2", 3)):
        sel = [v for v in jobs.values() if "priority" in v and PRIORITY_RANK[v["priority"]] <= maxrank]
        cum[label] = {s: sum(v[s] for v in sel) for s in ("low", "expected", "high")}
    return {"spent_so_far": spent, "jobs": jobs, "cumulative": cum}


def forecast_text(fc: dict) -> str:
    b = config.budget()
    lines = [f"FORECAST of remaining calls (USD; committed so far ${fc['spent_so_far']:.2f}):",
             f"{'job':<16}{'prio':<6}{'calls left':>11}{'low':>10}{'expected':>10}{'high':>10}{'job cap':>10}"]
    for name, v in fc["jobs"].items():
        if "error" in v:
            lines.append(f"{name:<16}  (not plannable yet: {v['error'][:70]})")
            continue
        cap = b["job_caps_usd"].get(name, 0)
        lines.append(f"{name:<16}{v['priority']:<6}{v['calls_remaining']:>11}{v['low']:>10.2f}{v['expected']:>10.2f}"
                     f"{v['high']:>10.2f}{cap:>10.2f}")
    lines.append("")
    for k, v in fc["cumulative"].items():
        lines.append(f"cumulative {k:<11} spent+remaining: low ${fc['spent_so_far'] + v['low']:.2f} | "
                     f"expected ${fc['spent_so_far'] + v['expected']:.2f} | high ${fc['spent_so_far'] + v['high']:.2f}"
                     f"   (hard cap ${b['hard_cap_usd']:.2f})")
    return "\n".join(lines)
