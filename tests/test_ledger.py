from ccrl.bedrock.ledger import Ledger
from ccrl.bedrock.runner import BudgetGuard


def _req(led, rid, job="j", model="m"):
    led.ensure_requests([{"request_id": rid, "job": job, "model_key": model, "item_id": rid}])


def test_spend_and_recover(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    _req(led, "a")
    _req(led, "b")
    assert led.ensure_requests([{"request_id": "a", "job": "j", "model_key": "m", "item_id": "a"}]) == 0
    led.open_attempt("a", 1, 1, 0.50, "{}")
    assert abs(led.committed() - 0.50) < 1e-12              # in flight: reservation counted
    led.close_attempt("a", 1, 1, status="ok", cost_reported=0.10, cost_guard=0.12)
    assert abs(led.committed() - 0.12) < 1e-12 and abs(led.reported() - 0.10) < 1e-12
    led.open_attempt("b", 1, 1, 0.30, "{}")                  # simulate a crash here
    assert led.recover() == 1
    assert abs(led.committed() - 0.42) < 1e-12              # uncertain stays counted
    led.close()


def test_guard_caps(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    g = BudgetGuard(led, "pilot", model_caps={"m": 1.0}, job_cap=2.0, hard_cap=3.0)
    assert g.check("m", 0.5)[0]
    _req(led, "x", job="pilot")
    led.open_attempt("x", 1, 1, 0.0, "{}")
    led.close_attempt("x", 1, 1, status="ok", cost_reported=0.9, cost_guard=0.9)
    ok, why = g.check("m", 0.2)
    assert not ok and "model cap" in why
    ok, why = g.check("other", 1.2)
    assert not ok and "job cap" in why
    ok, why = g.check("other", 2.5)
    assert not ok and "hard cap" in why
    led.close()
