"""Reconcile the ledger with AWS Cost Explorer (the actual bill).

    ccrl billing reconcile --start 2026-11-03 [--end 2026-11-20]

Prints daily cost by service (Bedrock and Marketplace model lines) and by
record type (Usage vs Credit), next to the ledger's reported spend per day.
Cost Explorer lags ~24 h and costs $0.01 per API request (two requests here).
Needs IAM permission ce:GetCostAndUsage. The key check: every model line
should be matched by a Credit line, i.e. the credit — not the card — pays.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

from ccrl.bedrock.ledger import Ledger
from ccrl.workspace import Workspace


def reconcile(ws: Workspace, start: str, end: str | None = None, profile: str | None = None) -> int:
    import boto3

    end = end or (dt.date.today() + dt.timedelta(days=1)).isoformat()
    s = boto3.Session(profile_name=profile) if profile else boto3.Session()
    ce = s.client("ce", region_name="us-east-1")
    try:
        by_service = ce.get_cost_and_usage(TimePeriod={"Start": start, "End": end}, Granularity="DAILY",
                                           Metrics=["UnblendedCost"], GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}])
        by_type = ce.get_cost_and_usage(TimePeriod={"Start": start, "End": end}, Granularity="DAILY",
                                        Metrics=["UnblendedCost"],
                                        GroupBy=[{"Type": "DIMENSION", "Key": "RECORD_TYPE"}])
    except Exception as e:
        print(f"Cost Explorer query failed ({type(e).__name__}: {e}). Check ce:GetCostAndUsage permission, "
              "or use the console: Billing -> Cost Explorer, group by Service and by Charge type.")
        return 1
    print("Daily cost by service (Bedrock / model lines only):")
    tot = defaultdict(float)
    for day in by_service["ResultsByTime"]:
        d = day["TimePeriod"]["Start"]
        for g in day["Groups"]:
            name = g["Keys"][0]
            amt = float(g["Metrics"]["UnblendedCost"]["Amount"])
            if amt and any(k in name.lower() for k in ("bedrock", "claude", "anthropic", "openai", "gpt", "grok",
                                                       "kimi", "moonshot", "deepseek", "glm", "qwen", "marketplace")):
                print(f"  {d}  {name:<60} {amt:>10.4f}")
                tot[name] += amt
    print("\nDaily totals by record type (Usage should be offset by Credit):")
    for day in by_type["ResultsByTime"]:
        d = day["TimePeriod"]["Start"]
        parts = [f"{g['Keys'][0]}={float(g['Metrics']['UnblendedCost']['Amount']):.4f}" for g in day["Groups"]]
        print(f"  {d}  " + "  ".join(parts))
    if ws.ledger.exists():
        led = Ledger(ws.ledger)
        try:
            rows = led.q("SELECT substr(finished_at,1,10) AS d, SUM(cost_reported) AS c FROM attempts "
                         "WHERE status IN ('ok','truncated','refused') GROUP BY d ORDER BY d")
        finally:
            led.close()
        print("\nLedger reported spend per local day (compare with the model lines above; CE uses UTC days):")
        for r in rows:
            print(f"  {r['d']}  {float(r['c'] or 0):>10.4f}")
    print(f"\nModel-line total in Cost Explorer: ${sum(tot.values()):.4f}")
    return 0


def reprice_flex(ws: Workspace, apply: bool = False) -> int:
    """Re-count finished Flex calls at the Flex rate in the budget guard.

        ccrl billing reprice-flex            # show what would change
        ccrl billing reprice-flex --apply    # write it (needs guard_counts_flex: true)

    Until `guard_counts_flex` is set, the guard records every finished Flex
    call at the full Standard price, so committed spend runs about twice the
    real spend. Once AWS has confirmed Flex billing (Credits page or
    `ccrl billing reconcile`: AWS total ~ the ledger's reported cost), this
    lowers those past guard costs to reported x safety factor. Only finished
    Flex attempts change; reservations and uncertain attempts keep their
    worst-case value. The change is recorded as a ledger event.
    """
    from ccrl import config
    from ccrl.bedrock import pricing

    led = Ledger(ws.ledger)
    try:
        rows = led.q("SELECT a.request_id, a.attempt, a.turn, a.cost_reported, a.cost_guard, r.model_key "
                     "FROM attempts a JOIN requests r USING(request_id) "
                     "WHERE a.served_tier='flex' AND a.status IN ('ok','truncated','refused')")
        prices = config.prices()["prices"]
        updates, before, after = [], 0.0, 0.0
        for r in rows:
            pk = r["model_key"] if r["model_key"] in prices else None
            if pk is None:
                continue
            new = float(r["cost_reported"] or 0.0) * pricing.price_for(pk).guard_factor
            old = float(r["cost_guard"] or 0.0)
            before += old
            after += new
            if new < old - 1e-12:
                updates.append((new, r["request_id"], r["attempt"], r["turn"]))
        total_before = led.committed()
        print(f"finished Flex attempts: {len(rows)}; guard cost {before:.2f} -> {after:.2f} "
              f"(committed total {total_before:.2f} -> {total_before - before + after:.2f}); "
              f"{len(updates)} rows would be lowered")
        if not apply:
            print("(nothing written; pass --apply after setting guard_counts_flex: true)")
            return 0
        if not pricing.flex_confirmed():
            raise SystemExit("Flex billing is not confirmed. Run `ccrl billing confirm-flex --start <date>` "
                             "(compares Cost Explorer with the ledger), or set guard_counts_flex: true in "
                             "config/prices.yaml only with that evidence.")
        with led.tx() as c:
            c.executemany("UPDATE attempts SET cost_guard=? WHERE request_id=? AND attempt=? AND turn=?", updates)
        led.event("repriced_flex", {"rows": len(updates), "guard_before": round(before, 4),
                                    "guard_after": round(after, 4)})
        print(f"applied: committed total now {led.committed():.2f}")
        return 0
    finally:
        led.close()


def _utc_day(iso: str) -> str:
    t = dt.datetime.fromisoformat(iso)
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc)
    return t.date().isoformat()


def flex_verdict(aws_by_day: dict[str, float], ledger_by_day: dict[str, float], today_utc: str,
                 min_ledger: float = 20.0) -> tuple[str, dict]:
    """Compare AWS model cost with the ledger's reported (Flex-discounted)
    cost on complete days at least two days old (Cost Explorer lags ~24 h, so
    a recent, partial day could fake a match). Verdict: confirmed (ratio
    0.85-1.20), standard (>= 1.5: AWS bills about twice the Flex price),
    or inconclusive."""
    cutoff = (dt.date.fromisoformat(today_utc) - dt.timedelta(days=2)).isoformat()
    days = sorted(d for d, v in ledger_by_day.items() if d <= cutoff and v >= 1.0)
    led = sum(ledger_by_day[d] for d in days)
    aws = sum(aws_by_day.get(d, 0.0) for d in days)
    info = {"days": days, "ledger_reported": round(led, 4), "aws_model_cost": round(aws, 4),
            "ratio": round(aws / led, 4) if led > 0 else None, "cutoff": cutoff}
    if led < min_ledger:
        return "inconclusive", info
    r = aws / led
    if 0.85 <= r <= 1.20:
        return "confirmed", info
    if r >= 1.5:
        return "standard", info
    return "inconclusive", info


def confirm_flex(ws: Workspace, start: str, profile: str | None = None) -> int:
    """`ccrl billing confirm-flex --start 2026-11-07`: if AWS bills the Flex calls
    at the Flex rate, record the evidence in runs/flex_confirmed.json and
    re-count past Flex calls at their real price in the budget guard."""
    import json

    import boto3

    from ccrl import paths
    from ccrl.util import now_iso

    today = dt.datetime.now(dt.timezone.utc).date()
    s = boto3.Session(profile_name=profile) if profile else boto3.Session()
    ce = s.client("ce", region_name="us-east-1")
    try:
        res = ce.get_cost_and_usage(TimePeriod={"Start": start, "End": (today + dt.timedelta(days=1)).isoformat()},
                                    Granularity="DAILY", Metrics=["UnblendedCost"],
                                    Filter={"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Usage"]}},
                                    GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}])
    except Exception as e:
        print(f"Cost Explorer query failed ({type(e).__name__}: {e}). Needs ce:GetCostAndUsage.")
        return 1
    keys = ("bedrock", "claude", "anthropic", "openai", "gpt", "grok", "kimi", "moonshot", "deepseek", "glm",
            "qwen", "marketplace")
    aws = defaultdict(float)
    for day in res["ResultsByTime"]:
        for g in day["Groups"]:
            if any(k in g["Keys"][0].lower() for k in keys):
                aws[day["TimePeriod"]["Start"]] += float(g["Metrics"]["UnblendedCost"]["Amount"])
    led = Ledger(ws.ledger)
    try:
        rows = led.q("SELECT finished_at, cost_reported FROM attempts WHERE status IN ('ok','truncated','refused') "
                     "AND finished_at IS NOT NULL")
    finally:
        led.close()
    ledger = defaultdict(float)
    for r in rows:
        ledger[_utc_day(r["finished_at"])] += float(r["cost_reported"] or 0.0)
    verdict, info = flex_verdict(dict(aws), dict(ledger), today.isoformat())
    print(f"days compared (UTC, complete): {info['days']}")
    print(f"AWS model usage ${info['aws_model_cost']:.2f} vs ledger reported ${info['ledger_reported']:.2f} "
          f"-> ratio {info['ratio']}  [{verdict}]")
    if verdict == "confirmed":
        (paths.RUNS / "flex_confirmed.json").write_text(json.dumps({**info, "confirmed_at": now_iso(),
                                                                   "start": start}, indent=2), encoding="utf-8")
        print("Flex billing confirmed; recorded in runs/flex_confirmed.json. Re-counting past Flex calls:")
        return reprice_flex(ws, apply=True)
    if verdict == "standard":
        print("AWS appears to bill these calls at about the Standard price, not the Flex price. Do NOT reprice; "
              "tell the team before running more jobs.")
        return 2
    print("Inconclusive (too little spend on complete days, or Cost Explorer has not caught up). Try again in a "
          "day; nothing was changed.")
    return 1


def confirm_flex_manual(ws: Workspace, aws_total: float, min_age_hours: float = 48.0) -> int:
    """`ccrl billing confirm-flex --aws-total 241.50`: the same check without
    Cost Explorer access (e.g. a machine with only a Bedrock API key).

    Compare the Credits page's "Total estimated amount used" with the
    ledger's total reported cost. Valid only when the newest paid call is at
    least 48 h old, so the AWS figure is complete (no billing lag)."""
    import json

    from ccrl import paths
    from ccrl.util import now_iso

    led = Ledger(ws.ledger)
    try:
        row = led.q("SELECT SUM(cost_reported) AS c, MAX(finished_at) AS last FROM attempts "
                    "WHERE status IN ('ok','truncated','refused')")[0]
        inflight = led.q("SELECT COUNT(*) AS n FROM attempts WHERE status='inflight'")[0]["n"]
    finally:
        led.close()
    total, last = float(row["c"] or 0.0), row["last"]
    if not last or total < 20:
        print("Too little spend in the ledger to compare; nothing changed.")
        return 1
    age_h = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(last).astimezone(dt.timezone.utc)
             ).total_seconds() / 3600
    if inflight or age_h < min_age_hours:
        print(f"The newest paid call is {age_h:.0f} h old (need >= {min_age_hours:.0f} h with nothing running), so "
              "the AWS figure may still be incomplete. Try again later; nothing changed.")
        return 1
    r = aws_total / total
    verdict = "confirmed" if 0.85 <= r <= 1.20 else ("standard" if r >= 1.5 else "inconclusive")
    print(f"AWS Credits page ${aws_total:.2f} vs ledger reported ${total:.2f} (newest call {age_h:.0f} h ago) "
          f"-> ratio {r:.3f}  [{verdict}]")
    if verdict == "confirmed":
        (paths.RUNS / "flex_confirmed.json").write_text(json.dumps({
            "method": "credits_page_total", "aws_total": aws_total, "ledger_reported": round(total, 4),
            "ratio": round(r, 4), "newest_call": last, "confirmed_at": now_iso()}, indent=2), encoding="utf-8")
        print("Flex billing confirmed; recorded in runs/flex_confirmed.json. Re-counting past Flex calls:")
        return reprice_flex(ws, apply=True)
    if verdict == "standard":
        print("AWS appears to bill these calls at about the Standard price. Do NOT reprice; tell the team.")
        return 2
    print("Ratio outside the expected range; nothing changed. Send this output to the team.")
    return 1
