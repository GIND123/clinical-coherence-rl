"""The vLLM backend's plumbing, against a fake `vllm` module (vLLM itself needs a GPU)."""
import math
import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture()
def fake_vllm(monkeypatch):
    calls = {"init": None, "generate": [], "sleep": 0, "wake": 0}

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    class LoRARequest:
        def __init__(self, name, int_id, path):
            self.name, self.int_id, self.path = name, int_id, path

    class LLM:
        def __init__(self, **kw):
            calls["init"] = kw

        def generate(self, prompts, params, lora_request=None, use_tqdm=False):
            calls["generate"].append((prompts, params, lora_request))
            outs = []
            for k, _p in enumerate(prompts):
                ids = [10 + k, 11 + k]                       # stopped on EOS, EOS id not returned
                lps = [{10 + k: SimpleNamespace(logprob=-0.5)}, {11 + k: SimpleNamespace(logprob=-1.5)}]
                outs.append(SimpleNamespace(outputs=[SimpleNamespace(
                    token_ids=ids, text=f"t{k}", logprobs=lps, finish_reason="stop", stop_reason=None)]))
            return outs

        def sleep(self, level=1):
            calls["sleep"] += 1

        def wake_up(self):
            calls["wake"] += 1

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
    lora = types.ModuleType("vllm.lora")
    req = types.ModuleType("vllm.lora.request")
    req.LoRARequest = LoRARequest
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", req)
    return calls


def test_engine_maps_outputs_and_appends_eos(fake_vllm):
    from ccrl.train.vllm_backend import VLLMEngine

    eng = VLLMEngine("m", lora_rank=32, enable_sleep=True, eos_token_ids=[99])
    assert fake_vllm["init"]["enable_lora"] and fake_vllm["init"]["max_lora_rank"] == 32
    assert fake_vllm["init"]["enable_sleep_mode"]
    out = eng.generate([[1, 2], [3]], lora_path="/tmp/a", seeds=[5, 6])
    (ids0, text0, lp0), (ids1, _t1, _lp1) = out
    assert ids0 == [10, 11, 99] and ids1 == [11, 12, 99] and text0 == "t0"
    assert lp0[:2] == [-0.5, -1.5] and math.isnan(lp0[2])
    prompts, params, lr = fake_vllm["generate"][-1]
    assert prompts[0] == {"prompt_token_ids": [1, 2]}
    assert [p.kw["seed"] for p in params] == [5, 6] and params[0].kw["logprobs"] == 1
    assert lr.int_id == 1 and lr.path == "/tmp/a"
    eng.generate([[1]], lora_path="/tmp/a")
    assert fake_vllm["generate"][-1][2].int_id == 2           # fresh adapter id every call
    eng.sleep()
    eng.sleep()
    assert fake_vllm["sleep"] == 1 and eng.asleep
    eng.generate([[1]])
    assert fake_vllm["wake"] == 1 and fake_vllm["generate"][-1][2] is None


def test_rollout_generator_saves_policy_then_sleeps(fake_vllm, tmp_path):
    from ccrl.train.vllm_backend import VLLMEngine, VLLMRolloutGenerator

    saved = []
    model = SimpleNamespace(save_pretrained=lambda d: saved.append(str(d)))

    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            return msgs[-1]["content"]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

    eng = VLLMEngine("m", lora_rank=32, enable_sleep=True, eos_token_ids=[99])
    gen = VLLMRolloutGenerator(eng, model, Tok(), tmp_path / "ad", {"enable_thinking": False})
    out = gen([[{"role": "user", "content": "ab"}], [{"role": "user", "content": "c"}]])
    assert saved == [str(tmp_path / "ad")] and eng.asleep
    assert out[0][0] == [97, 98] and out[0][1][-1] == 99 and len(out[0][3]) == 3


def test_backend_override_keeps_other_generation_keys():
    from ccrl.train.grpo import load_config

    cfg = load_config("config/train/ccr_acc_fid.yaml", {"generation": {"backend": "vllm"}})
    g = cfg["generation"]
    assert g["backend"] == "vllm" and g["max_new_tokens"] == 512 and g["temperature"] == 1.0


def test_tis_weight_ignores_missing_rollout_logprobs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tests.test_train_smoke import _tiny

    from ccrl.train.common import Sample
    from ccrl.train.grpo import pg_kl_loss
    m = _tiny()
    s = [Sample([1, 5, 9], [3, 4, 8], 0.7, [-1.0, float("nan"), -2.0]), Sample([2, 6], [7, 1], -1.2, None)]
    loss, _ = pg_kl_loss(m, s, 0, 5, kl_coef=0.0, chunk=4, tis_cap=2.0)
    loss.backward()
    assert torch.isfinite(loss)


def test_tau_i_from_pilot_eval(tmp_path):
    import numpy as np
    import pandas as pd

    from ccrl.train.evaluate import tau_i_from_eval
    rows = []
    for k in range(20):
        p = np.full(4, 0.25)
        q = p.copy()
        q[0] += 0.01 * k
        q[1] -= 0.01 * k
        for tr, v in (("base", p), ("retest", q)):
            rows.append({"case_id": f"c{k}", "transform": tr, "parse_ok": True, "probs": list(v)})
    rows.append({"case_id": "c99", "transform": "base", "parse_ok": True, "probs": [1, 0, 0, 0]})   # no partner
    f = tmp_path / "e.parquet"
    pd.DataFrame(rows).to_parquet(f)
    r = tau_i_from_eval(str(f))
    assert r["n_cases"] == 20 and 0 < r["median_js"] < r["tau_I"] < 1


def test_collapse_check_ignores_batch_noise_but_catches_collapse():
    import numpy as np

    from ccrl.train.grpo import collapse_check
    # the two real decisive runs (27 Nov 2026) that the single-step rule stopped at steps 10 and 13
    ccr = [0.574, 0.374, 0.515, 0.393, 0.312, 0.556, 0.352, 0.324, 0.316, 0.289]
    rlvr = [0.525, 0.426, 0.509, 0.445, 0.356, 0.589, 0.419, 0.465, 0.458, 0.45, 0.374, 0.441, 0.316]
    for m in (ccr, rlvr):
        assert all(collapse_check(m[:k]) is None for k in range(1, len(m) + 1))
    rng = np.random.default_rng(0)
    noisy = list(0.45 * (1 + 0.25 * rng.standard_normal(300)))           # stationary, CV 0.25
    assert all(collapse_check(noisy[:k]) is None for k in range(1, 301))
    collapsing = noisy[:40] + [0.05] * 40                                  # M -> ~0
    hits = [k for k in range(1, 81) if collapse_check(collapsing[:k])]
    assert hits and hits[0] <= 40 + 20
