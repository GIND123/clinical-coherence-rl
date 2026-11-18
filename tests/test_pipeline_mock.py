"""End to end on the mock: probe -> prep -> items -> freeze -> job -> rerun is a
no-op -> priority order is enforced -> a freeze violation is refused -> budget stop."""
import pytest

from ccrl import config
from ccrl.bedrock import models as M
from ccrl.panel import freeze, items, jobs
from ccrl.prep import medcase, phrasebank
from ccrl.workspace import get as get_ws


@pytest.fixture(scope="module")
def ws_ready(tmp_path_factory):
    from ccrl import paths
    mp = pytest.MonkeyPatch()
    mp.setattr(paths, "RUNS", tmp_path_factory.mktemp("runs"))
    ws = get_ws(True)
    M.probe(list(config.models()), ws, M.make_client_factory(True),
            {k: f"mock.{k}" for k in config.models()}, verbose=False)
    phrasebank.run(ws)
    medcase.run(ws, "all", ["pilot", "panel"])
    phrasebank.finalize(ws)
    medcase.finalize(ws)
    items.build(ws, allow_short_mcr=True, verbose=False)
    freeze.freeze(ws)
    yield ws
    mp.undo()


def test_job_runs_and_is_idempotent(ws_ready):
    s1 = jobs.run_job("pilot", ws_ready, models_filter=["gpt-oss-120b", "kimi-k3"], limit_cases=2)
    assert s1["outcome"].get("done", 0) == s1["ran"] > 0
    s2 = jobs.run_job("pilot", ws_ready, models_filter=["gpt-oss-120b", "kimi-k3"], limit_cases=2)
    assert s2["ran"] == 0


def test_order_gate_and_freeze_violation(ws_ready):
    with pytest.raises(SystemExit):
        jobs.run_job("effort_arm", ws_ready, limit_cases=1)          # panel_main not complete
    p = ws_ready.items_dir / "sets.json"
    orig = p.read_bytes()
    p.write_bytes(orig + b" ")
    try:
        with pytest.raises(SystemExit):
            jobs.run_job("pilot", ws_ready, limit_cases=1)
    finally:
        p.write_bytes(orig)


def test_budget_stop(ws_ready):
    s = jobs.run_job("panel_main", ws_ready, models_filter=["kimi-k3"], limit_cases=4, max_spend=0.02,
                     force_order=True)
    assert s["budget_stop"] and s["outcome"].get("budget_stop", 0) >= 1


def test_partner_job_skips_unavailable_models(ws_ready):
    """partner_panel runs only on partner models that passed their probe;
    with none available it refuses instead of failing half-way."""
    specs = M.load_lock(True)
    panel_only = {k: v for k, v in specs.items() if k not in config.optional_models()}
    with pytest.raises(SystemExit, match="none of its models"):
        jobs.plan_tasks("partner_panel", ws_ready, panel_only, limit_cases=1)
    one = dict(panel_only, **{"sonnet-5.5": specs["sonnet-5.5"]})
    tasks = jobs.plan_tasks("partner_panel", ws_ready, one, limit_cases=1)
    assert tasks and {t.model_key for t in tasks} == {"sonnet-5.5"}


def test_effort_arm_refuses_without_effort_control(ws_ready):
    """If the probe degraded gpt-oss-120b to provider default, the three effort
    levels would be identical requests: the arm must refuse, not pay three times."""
    import dataclasses
    specs = M.load_lock(True)
    tasks = jobs.plan_tasks("effort_arm", ws_ready, specs, limit_cases=1)
    assert {t.effort for t in tasks} == {"low", "medium", "high"}
    specs["gpt-oss-120b"] = dataclasses.replace(specs["gpt-oss-120b"], effort_strategy="none")
    with pytest.raises(SystemExit, match="no working reasoning-effort control"):
        jobs.plan_tasks("effort_arm", ws_ready, specs, limit_cases=1)


def test_resample_and_temp0_plans(ws_ready):
    """resample: DDXPlus only, base + ablate, replicates 1..3 with distinct IDs;
    temp0: temperature 0 on every task; neither collides with panel_main."""
    specs = M.load_lock(True)
    rs = jobs.plan_tasks("resample", ws_ready, specs, limit_cases=2)
    assert {t.dataset for t in rs} == {"ddxplus"} and {t.transform for t in rs} == {"base", "ablate"}
    assert {t.replicate for t in rs} == {1, 2, 3} and len({t.request_id for t in rs}) == len(rs)
    assert all(t.temperature is None for t in rs)
    t0 = jobs.plan_tasks("temp0", ws_ready, specs, limit_cases=2)
    assert t0 and all(t.temperature == 0.0 for t in t0)
    panel = {t.request_id for t in jobs.plan_tasks("panel_main", ws_ready, specs, limit_cases=2)}
    assert panel.isdisjoint({t.request_id for t in rs}) and panel.isdisjoint({t.request_id for t in t0})
