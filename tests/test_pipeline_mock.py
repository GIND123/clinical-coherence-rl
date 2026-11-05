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
    s1 = jobs.run_job("pilot", ws_ready, models_filter=["opus-5.5", "kimi-k3"], limit_cases=2)
    assert s1["outcome"].get("done", 0) == s1["ran"] > 0
    s2 = jobs.run_job("pilot", ws_ready, models_filter=["opus-5.5", "kimi-k3"], limit_cases=2)
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
    s = jobs.run_job("panel_main", ws_ready, models_filter=["opus-5.5"], limit_cases=4, max_spend=0.02,
                     force_order=True)
    assert s["budget_stop"] and s["outcome"].get("budget_stop", 0) >= 1
