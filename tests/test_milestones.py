"""P4 (plan_final.md): milestone adapters survive checkpoint rotation; resume state stays on the last two."""
import pytest

pytest.importorskip("torch")


class _Fake:
    def save_pretrained(self, d):
        (d / "adapter_model.safetensors").write_text("x")

    def state_dict(self):
        return {}


def test_milestones_kept_and_resume_state_rotates(tmp_path):
    from ccrl.train.grpo import _save
    f = _Fake()
    for step in range(25, 201, 25):
        _save(f, f, f, f, tmp_path, step, keep_every=50)
    names = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert names == ["step_0050", "step_0100", "step_0150", "step_0175", "step_0200"]
    with_state = sorted(p.parent.name for p in (tmp_path / "checkpoints").glob("*/trainer_state.pt"))
    assert with_state == ["step_0175", "step_0200"]
    assert all((tmp_path / "checkpoints" / n / "adapter_model.safetensors").exists() for n in names)
