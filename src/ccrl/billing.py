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
