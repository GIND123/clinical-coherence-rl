import pytest


@pytest.fixture()
def tmp_runs(tmp_path, monkeypatch):
    """Isolate every ledger/workspace write under a temporary runs/ tree."""
    from ccrl import paths
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    return tmp_path / "runs"
