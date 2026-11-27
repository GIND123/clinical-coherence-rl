#!/usr/bin/env python3
"""Lambda Cloud helper for Track B. The Lambda account is SHARED: this script only ever
lists, launches or terminates instances named NAME, and only adds the SSH key NAME.

    export LAMBDA_API_KEY=...            # never commit it
    python scripts/lambda_api.py types                     # GPU types with capacity now, $/h
    python scripts/lambda_api.py list                      # our instances only
    python scripts/lambda_api.py add-key ~/.ssh/govind_llm.pub
    python scripts/lambda_api.py launch gpu_1x_h100_pcie us-west-3 [--allow-existing]
    python scripts/lambda_api.py terminate --yes           # terminates ALL of OUR instances
    python scripts/lambda_api.py terminate <id> --yes      # one of ours, by id
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://cloud.lambdalabs.com/api/v1"
NAME = "govind_llm"


def call(method: str, path: str, body: dict | None = None) -> dict:
    key = os.environ.get("LAMBDA_API_KEY")
    if not key:
        raise SystemExit("set LAMBDA_API_KEY")
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{key}:".encode()).decode())
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "ccrl-lambda/1.0")     # Cloudflare rejects urllib's default agent (1010)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path}: HTTP {e.code} {e.read().decode(errors='replace')[:400]}")


def ours() -> list[dict]:
    return [i for i in call("GET", "/instances")["data"] if i.get("name") == NAME]


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "list"
    if cmd == "types":
        rows = []
        for k, v in call("GET", "/instance-types")["data"].items():
            regions = [r["name"] for r in v.get("regions_with_capacity_available", [])]
            if regions:
                t = v["instance_type"]
                rows.append((t["price_cents_per_hour"] / 100, k, t.get("gpu_description", ""), regions))
        for price, k, gpu, regions in sorted(rows):
            print(f"${price:6.2f}/h  {k:28s} {gpu:30s} {', '.join(regions)}")
    elif cmd == "list":
        for i in ours():
            print(json.dumps({k: i.get(k) for k in ("id", "name", "status", "ip", "jupyter_url")}
                             | {"type": i["instance_type"]["name"], "region": i["region"]["name"]}))
        if not ours():
            print(f"no instances named {NAME}")
    elif cmd == "add-key":
        pub = open(os.path.expanduser(argv[1])).read().strip()
        names = [k["name"] for k in call("GET", "/ssh-keys")["data"]]
        if NAME in names:
            print(f"ssh key {NAME} already exists")
        else:
            call("POST", "/ssh-keys", {"name": NAME, "public_key": pub})
            print(f"added ssh key {NAME}")
    elif cmd == "launch":
        itype, region = argv[1], argv[2]
        if ours() and "--allow-existing" not in argv:
            raise SystemExit(f"an instance named {NAME} already exists; reuse it, or add --allow-existing")
        r = call("POST", "/instance-operations/launch", {"region_name": region, "instance_type_name": itype,
                                                          "ssh_key_names": [NAME], "quantity": 1, "name": NAME})
        print(json.dumps(r))
    elif cmd == "terminate":
        mine = [i["id"] for i in ours()]
        wanted = [a for a in argv[1:] if not a.startswith("--")]
        foreign = [a for a in wanted if a not in mine]
        if foreign:
            raise SystemExit(f"refusing: {foreign} are not instances named {NAME}")
        ids = wanted or mine
        if not ids:
            print(f"no instances named {NAME}")
            return 0
        if "--yes" not in argv:
            raise SystemExit(f"would terminate {ids}; add --yes")
        print(json.dumps(call("POST", "/instance-operations/terminate", {"instance_ids": ids})))
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
