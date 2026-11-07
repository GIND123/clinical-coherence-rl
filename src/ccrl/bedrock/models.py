"""Model resolution, live probing, and the frozen model lock.

    ccrl models list      list every Bedrock model and inference profile visible
                          to these credentials in the region (no cost)
    ccrl models resolve   match each registry entry to a live ID -> config/models.resolved.yaml
    ccrl models probe     tiny real calls per model (~$0.5 total, billed to the ledger
                          under job 'probe') -> config/models.lock.yaml

The probe ladder per model (each step is one small request):
  1. ping with NO optional parameters                 -> the model ID works at all
  2. effort strategy (+ fallbacks)                    -> which effort encoding is accepted
     (required: true and nothing accepted => the model FAILS the probe)
  3. service tier (e.g. flex)                         -> degraded to default if rejected
  4. each optional field (e.g. Claude thinking display) -> dropped if rejected
  5. prompt cache point                               -> disabled if rejected
  6. a toy 5-candidate posterior with the final settings -> must parse
  7. effort-arm models only: low vs high on the same toy case -> warns if the
     output-token count does not rise (effort silently ignored)
Every accepted/degraded decision is written to the lock with the evidence.
"""
from __future__ import annotations

import copy
import json
import platform
import re
from pathlib import Path

import yaml

from ccrl import config, paths
from ccrl.bedrock.client import CallSpec, build_request
from ccrl.bedrock.ledger import Ledger
from ccrl.bedrock.runner import BudgetGuard, BudgetStop, Executor, Task, task_id
from ccrl.prompts.parse import parse_posterior
from ccrl.util import now_iso, stable_hash
from ccrl.workspace import Workspace

PROBE_SYSTEM_PING = "TASK: PROBE_PING\nReply with the single word OK."
TOY_CANDIDATES = [("AA", "Acute appendicitis"), ("AB", "Gastroenteritis"), ("AC", "Ovarian torsion"),
                  ("AD", "Ectopic pregnancy"), ("AE", "Urinary tract infection")]
TOY_VIGNETTE = ("A 24-year-old woman presents for evaluation. The pain began around the umbilicus 18 hours ago "
                "and has moved to the right lower quadrant. She has anorexia and one episode of vomiting. "
                "Temperature is 37.9 C. There is tenderness at McBurney's point with guarding. "
                "A urine pregnancy test is negative.")


# ------------------------------------------------------------------ listing
def list_live(region: str, profile: str | None = None) -> dict:
    import boto3

    s = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)
    br = s.client("bedrock")
    fms = br.list_foundation_models().get("modelSummaries", [])
    profs = []
    for t in ("SYSTEM_DEFINED",):
        token = None
        while True:
            kw = {"typeEquals": t, "maxResults": 1000}
            if token:
                kw["nextToken"] = token
            r = br.list_inference_profiles(**kw)
            profs.extend(r.get("inferenceProfileSummaries", []))
            token = r.get("nextToken")
            if not token:
                break
    return {
        "foundation_models": [{"id": m["modelId"], "name": m.get("modelName"), "provider": m.get("providerName"),
                               "inference": m.get("inferenceTypesSupported", []),
                               "status": (m.get("modelLifecycle") or {}).get("status")} for m in fms],
        "inference_profiles": [{"id": p["inferenceProfileId"], "name": p.get("inferenceProfileName"),
                                "status": p.get("status")} for p in profs],
    }


def resolve(live: dict) -> dict:
    """Pick one live ID per registry entry. Explicit candidates win (in order);
    otherwise the name regex, preferring global. > us. > bare on-demand IDs."""
    ids = {}
    for m in live["foundation_models"]:
        ids[m["id"]] = {"kind": "model", "name": m["name"], "provider": m["provider"],
                        "on_demand": "ON_DEMAND" in (m["inference"] or []), "status": m["status"]}
    for p in live["inference_profiles"]:
        ids[p["id"]] = {"kind": "profile", "name": p["name"], "status": p["status"], "on_demand": True}
    out = {}
    for key, spec in config.models().items():
        chosen, how, alts = None, None, []
        for c in spec.id_candidates:
            if c in ids:
                alts.append(c)
        rx = re.compile(spec.match, re.I) if spec.match else None
        if rx:
            for i, meta in ids.items():
                if (rx.search(i) or rx.search(meta.get("name") or "")) and i not in alts:
                    alts.append(i)

        def rank(i: str) -> tuple:
            explicit = spec.id_candidates.index(i) if i in spec.id_candidates else 99
            pref = 0 if i.startswith("global.") else 1 if i.startswith("us.") else 2
            usable = 0 if ids[i].get("on_demand") or ids[i]["kind"] == "profile" else 1
            return (usable, explicit, pref, i)
        alts.sort(key=rank)
        if alts:
            chosen = alts[0]
            how = "explicit" if chosen in spec.id_candidates else "regex"
        out[key] = {"model_id": chosen, "how": how, "alternatives": alts[:8],
                    "evidence": {a: ids[a] for a in alts[:8]}}
    return out


def write_resolved(res: dict, region: str) -> Path:
    doc = {"generated_at": now_iso(), "region": region, "models": res}
    with open(paths.MODELS_RESOLVED, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    return paths.MODELS_RESOLVED


# ------------------------------------------------------------------ lock
def spec_from_registry(key: str, model_id: str) -> CallSpec:
    m = config.models()[key]
    return CallSpec(model_key=key, model_id=model_id, price_key=key, region=m.region,
                    effort_strategy=m.effort_strategy, service_tier=m.service_tier, prompt_cache=m.prompt_cache,
                    extra_fields=copy.deepcopy(m.optional_fields), max_tokens=dict(m.max_tokens),
                    read_timeout_s=m.read_timeout_s, connect_timeout_s=m.connect_timeout_s,
                    concurrency=m.concurrency, default_effort=m.default_effort)


def spec_to_lock(spec: CallSpec, probe: dict) -> dict:
    return {"model_id": spec.model_id, "price_key": spec.price_key, "region": spec.region,
            "effort_strategy": spec.effort_strategy, "default_effort": spec.default_effort,
            "service_tier": spec.service_tier, "prompt_cache": spec.prompt_cache,
            "extra_fields": spec.extra_fields, "max_tokens": spec.max_tokens,
            "read_timeout_s": spec.read_timeout_s, "connect_timeout_s": spec.connect_timeout_s,
            "concurrency": spec.concurrency, "probe": probe}


def registry_sig(key: str) -> str:
    """Hash of a model's registry entry (config/models.yaml). Stored in the lock
    at probe time, so a lock made under a different registry entry is detected."""
    return stable_hash(json.dumps(config.models()[key].raw, sort_keys=True, default=str))


def stale_lock_models(doc: dict | None = None) -> list[str]:
    """Models that passed a probe under a registry entry that has since changed."""
    doc = doc if doc is not None else lock_doc()
    if not doc:
        return []
    reg = config.models()
    return [k for k, v in doc.get("models", {}).items()
            if v.get("probe", {}).get("ok") and k in reg and v.get("registry_sig") != registry_sig(k)]


def load_lock(dry_run: bool = False) -> dict[str, CallSpec]:
    """Model specs for paid jobs. Dry runs synthesise a lock from the registry
    (mock IDs, real price keys so cost reports are realistic)."""
    if dry_run:
        return {k: spec_from_registry(k, f"mock.{k}") for k in config.models()}
    if not paths.MODELS_LOCK.exists():
        raise SystemExit("config/models.lock.yaml not found: run `ccrl models resolve` then `ccrl models probe`.")
    with open(paths.MODELS_LOCK, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    stale = stale_lock_models(doc)
    if stale:
        raise SystemExit(f"config/models.yaml changed for {stale} since they were probed. Re-probe them "
                         f"(a few cents): ccrl models probe --models {','.join(stale)}")
    out = {}
    for k, v in doc["models"].items():
        if not v.get("probe", {}).get("ok"):
            continue
        out[k] = CallSpec(model_key=k, model_id=v["model_id"], price_key=v["price_key"], region=v["region"],
                          effort_strategy=v["effort_strategy"], service_tier=v["service_tier"],
                          prompt_cache=bool(v["prompt_cache"]), extra_fields=v.get("extra_fields") or {},
                          max_tokens=v["max_tokens"], read_timeout_s=int(v["read_timeout_s"]),
                          connect_timeout_s=int(v["connect_timeout_s"]), concurrency=int(v["concurrency"]),
                          default_effort=v.get("default_effort"))
    return out


def lock_doc() -> dict | None:
    if not paths.MODELS_LOCK.exists():
        return None
    with open(paths.MODELS_LOCK, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ probe
def _effort_arm_models() -> set[str]:
    out = set()
    for name, j in config.jobs().items():
        if any(e != "default" for e in j.get("efforts", ["default"])):
            out.update(config.job_models(name))
    return out


def probe(keys: list[str], ws: Workspace, client_factory, ids: dict[str, str], verbose: bool = True) -> dict:
    """Run the probe ladder. `ids`: key -> resolved model id. Returns lock doc."""
    ledger = Ledger(ws.ledger)
    guard = BudgetGuard(ledger, "probe")
    effort_models = _effort_arm_models()
    doc = lock_doc() or {"models": {}}
    doc.update({"generated_at": now_iso(), "python": platform.python_version()})
    try:
        import boto3
        import botocore
        doc.update({"boto3": boto3.__version__, "botocore": botocore.__version__})
    except Exception:
        pass

    def say(*a):
        if verbose:
            print(*a, flush=True)

    for key in keys:
        mid = ids.get(key)
        if not mid:
            doc["models"][key] = {"model_id": None, "probe": {"ok": False, "error": "unresolved"}}
            say(f"[probe] {key}: NOT RESOLVED — skipped")
            continue
        spec = spec_from_registry(key, mid)
        reg = config.models()[key]
        client = client_factory(spec)
        ex = Executor(ledger, {key: client}, {key: spec}, ws, guard)
        log: list[dict] = []
        warnings: list[str] = []
        n = [0]

        def call(system: str, user: str, sp: CallSpec, label: str, effort: str = "default",
                 max_tokens: int | None = None):
            n[0] += 1
            ex.specs[key] = sp
            rid = task_id("probe", f"{key}:{label}:{now_iso()}:{n[0]}", key, sp.model_id, effort, 0, system, [user])
            t = Task(rid, "probe", "A", key, f"probe:{key}:{label}", "probe", label, "probe", effort, system, [user],
                     kind="json")
            ledger.ensure_requests([{"request_id": rid, "job": "probe", "priority": "A", "model_key": key,
                                     "model_id": sp.model_id, "item_id": t.item_id, "transform": label,
                                     "effort": effort}])
            res = ex.call(t, 1, 1, [{"role": "user", "text": user}], max_tokens=max_tokens)
            ledger.finish_request(rid, "done" if res.billed else "failed", 1)
            entry = {"step": label, "status": res.status, "error": res.error_code, "msg": (res.error_msg or "")[:300],
                     "usage": res.usage, "served_tier": res.served_tier, "stop": res.stop_reason}
            log.append(entry)
            return res

        try:
            bare = copy.deepcopy(spec)
            bare.effort_strategy, bare.service_tier, bare.prompt_cache, bare.extra_fields = "none", "default", False, {}
            r = call(PROBE_SYSTEM_PING, "Ping.", bare, "ping", max_tokens=2048)
            for retry in range(2):          # transient timeouts/errors are not a verdict
                if r.billed or r.fatal:
                    break
                r = call(PROBE_SYSTEM_PING, "Ping.", bare, f"ping_retry{retry + 1}", max_tokens=2048)
            if not r.billed:
                # try alternatives from the resolution
                alts = (load_resolved().get(key) or {}).get("alternatives", [])
                for alt in alts:
                    if alt == mid:
                        continue
                    bare.model_id = alt
                    r = call(PROBE_SYSTEM_PING, "Ping.", bare, f"ping_alt:{alt}", max_tokens=2048)
                    if r.billed:
                        spec.model_id = alt
                        warnings.append(f"model id {mid} failed; using alternative {alt}")
                        break
            if not r.billed:
                doc["models"][key] = spec_to_lock(spec, {"ok": False, "error": f"{r.error_code}: {r.error_msg}",
                                                         "log": log, "at": now_iso()})
                say(f"[probe] {key}: FAILED ping ({r.error_code}: {(r.error_msg or '')[:160]})")
                continue
            # effort
            if spec.effort_strategy != "none":
                accepted = None
                for strat in [spec.effort_strategy] + list(reg.effort.get("fallbacks", []) or []):
                    sp = copy.deepcopy(bare)
                    sp.model_id = spec.model_id
                    sp.effort_strategy, sp.default_effort = strat, spec.default_effort
                    r = call(PROBE_SYSTEM_PING, "Ping.", sp, f"effort:{strat}", max_tokens=4096)
                    if r.billed:
                        accepted = strat
                        break
                if accepted is None:
                    if reg.effort.get("required"):
                        doc["models"][key] = spec_to_lock(spec, {"ok": False, "error": "no effort strategy accepted",
                                                                 "log": log, "at": now_iso()})
                        say(f"[probe] {key}: FAILED — effort is required and no strategy was accepted")
                        continue
                    warnings.append("effort not accepted; running at provider default")
                    spec.effort_strategy, spec.default_effort = "none", None
                else:
                    if accepted != spec.effort_strategy:
                        warnings.append(f"effort strategy {spec.effort_strategy} rejected; using {accepted}")
                    spec.effort_strategy = accepted
            # tier
            if spec.service_tier != "default":
                sp = copy.deepcopy(spec)
                sp.prompt_cache, sp.extra_fields = False, {}
                r = call(PROBE_SYSTEM_PING, "Ping.", sp, f"tier:{spec.service_tier}", max_tokens=4096)
                if not r.billed:
                    warnings.append(f"service tier {spec.service_tier} rejected ({r.error_code}); using default")
                    spec.service_tier = "default"
                elif (r.served_tier or "default") != spec.service_tier:
                    warnings.append(f"requested tier {spec.service_tier}, served {r.served_tier}")
            # optional fields
            kept = {}
            for fk, fv in (spec.extra_fields or {}).items():
                sp = copy.deepcopy(spec)
                sp.prompt_cache, sp.extra_fields = False, {fk: fv}
                r = call(PROBE_SYSTEM_PING, "Ping.", sp, f"field:{fk}", max_tokens=4096)
                if r.billed:
                    kept[fk] = fv
                else:
                    warnings.append(f"optional field {fk} rejected ({r.error_code}); dropped")
            spec.extra_fields = kept
            # cache point
            if spec.prompt_cache:
                sp = copy.deepcopy(spec)
                r = call(PROBE_SYSTEM_PING, "Ping.", sp, "cache", max_tokens=4096)
                if not r.billed:
                    warnings.append(f"cache point rejected ({r.error_code}); caching disabled")
                    spec.prompt_cache = False
            # toy posterior with final settings
            from ccrl.prompts import templates as T
            system = T.system_prompt("posterior", None, fixed=False)
            user = T.user_prompt(TOY_VIGNETTE, TOY_CANDIDATES)
            r = call(system, user, spec, "toy_posterior")
            pr = parse_posterior(r.text, [c for c, _ in TOY_CANDIDATES]) if r.billed else None
            toy_ok = bool(pr and pr.ok)
            if not toy_ok:
                warnings.append(f"toy posterior did not parse ({r.status}; {pr.error if pr else r.error_code})")
            effort_check = None
            if key in effort_models and spec.effort_strategy != "none":
                lo = call(system, user, spec, "toy_low", effort="low")
                hi = call(system, user, spec, "toy_high", effort="high")
                ol = int((lo.usage or {}).get("outputTokens", 0))
                oh = int((hi.usage or {}).get("outputTokens", 0))
                effort_check = {"low_output_tokens": ol, "high_output_tokens": oh}
                if not (lo.billed and hi.billed) or oh <= ol:
                    warnings.append(f"effort may be ignored: output tokens low={ol}, high={oh} (check before effort arm)")
            ok = toy_ok
            doc["models"][key] = spec_to_lock(spec, {"ok": ok, "warnings": warnings, "toy_parse_ok": toy_ok,
                                                     "toy_dx": pr.dx if pr else None, "effort_check": effort_check,
                                                     "log": log, "at": now_iso()})
            doc["models"][key]["registry_sig"] = registry_sig(key)
            say(f"[probe] {key}: {'OK' if ok else 'FAILED'} id={spec.model_id} effort={spec.effort_strategy}/"
                f"{spec.default_effort} tier={spec.service_tier} cache={spec.prompt_cache} "
                f"fields={list(spec.extra_fields)}" + (f" WARN: {warnings}" if warnings else ""))
        except BudgetStop as e:
            say(f"[probe] budget stop: {e}")
            break
    if not ws.dry_run:
        with open(paths.MODELS_LOCK, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    ledger.close()
    return doc


def load_resolved() -> dict:
    if not paths.MODELS_RESOLVED.exists():
        return {}
    with open(paths.MODELS_RESOLVED, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("models", {})


def make_client_factory(dry_run: bool, profile: str | None = None):
    if dry_run:
        from ccrl.bedrock.mock import MockBedrock
        mock = MockBedrock(seed=int(config.study()["seed"]))
        return lambda spec: mock
    from ccrl.bedrock.client import BedrockClient
    cache: dict = {}

    def f(spec: CallSpec):
        k = (spec.region, spec.read_timeout_s, spec.connect_timeout_s)
        if k not in cache:
            cache[k] = BedrockClient(spec.region, profile, spec.read_timeout_s, spec.connect_timeout_s)
        return cache[k]
    return f


def describe_request(spec: CallSpec) -> str:
    """Human-readable example of what a request for this spec looks like."""
    req = build_request(spec, "<system>", [{"role": "user", "text": "<vignette>"}])
    req["system"] = [b if "text" not in b else {"text": "<system>"} for b in req["system"]]
    return json.dumps(req, indent=2)
