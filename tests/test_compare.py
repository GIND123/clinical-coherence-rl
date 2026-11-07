"""H6 wiring: a trained model is compared with one panel model from the
Bedrock export, on the same items. The export holds several models and jobs,
so the loader must narrow it, and must refuse an ambiguous file."""
import pandas as pd
import pytest

from ccrl.train.evaluate import load_posteriors


def _rows(model, job, n=3):
    return [{"job": job, "model_key": model, "effort": "default", "item_id": f"i{k}", "case_id": f"c{k}",
             "transform": "base", "dataset": "ddxplus", "turn": 1, "parse_ok": True} for k in range(n)]


def test_panel_export_is_narrowed(tmp_path):
    p = tmp_path / "posteriors.parquet"
    pd.DataFrame(_rows("qwen3-32b", "panel_main") + _rows("qwen3-32b", "panel_ext") + _rows("kimi-k3", "panel_main")
                 ).to_parquet(p, index=False)
    got = load_posteriors(str(p), model="qwen3-32b")
    assert set(got["model_key"]) == {"qwen3-32b"} and set(got["job"]) == {"panel_main"} and len(got) == 3
    with pytest.raises(SystemExit, match="several models"):
        load_posteriors(str(p))
    with pytest.raises(SystemExit, match="no rows"):
        load_posteriors(str(p), model="opus-5.5")


def test_local_eval_file_loads_unchanged(tmp_path):
    p = tmp_path / "ccr.parquet"
    pd.DataFrame(_rows("q8_ccr_s0", "local:panel")).to_parquet(p, index=False)
    assert len(load_posteriors(str(p))) == 3
