from ccrl.panel import freeze, items
from ccrl.workspace import get as get_ws


def test_freeze_detects_change(tmp_runs):
    ws = get_ws(True)
    items.build(ws, allow_missing_para=True, allow_short_mcr=True, verbose=False)
    freeze.freeze(ws)
    assert freeze.check_freeze(ws)[0]
    p = ws.items_dir / "systems.json"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    ok, msg = freeze.check_freeze(ws)
    assert not ok and "systems.json" in msg


def test_spending_controls_not_frozen(tmp_runs):
    """Correcting a price or a cap must not invalidate the design freeze."""
    ws = get_ws(True)
    items.build(ws, allow_missing_para=True, allow_short_mcr=True, verbose=False)
    names = set(freeze.covered_files(ws))
    assert "config/study.yaml" in names and "config/models.yaml" in names
    assert "config/prices.yaml" not in names and "config/budget.yaml" not in names
