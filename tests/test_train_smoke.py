"""CPU smoke tests of the Track B math on a tiny random Qwen3 (no downloads)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from ccrl.train.common import (Sample, completion_logprobs, completion_logprobs_lowmem,  # noqa: E402
                               token_budget_batches)


def _tiny():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                      tie_word_embeddings=False)
    m = Qwen3ForCausalLM(cfg).float()
    return get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"))


def _samples():
    return [Sample([1, 5, 9, 11], [3, 4, 8], 0.7), Sample([2, 6], [7, 7, 1, 30, 2], -1.2)]


def test_lowmem_logprobs_match_naive():
    m = _tiny()
    a, _ = completion_logprobs(m, _samples(), 0)
    b, _ = completion_logprobs_lowmem(m, _samples(), 0, chunk=2)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5)


def test_pg_kl_loss_backprops():
    from ccrl.train.grpo import pg_kl_loss
    m = _tiny()
    for n, p in m.named_parameters():          # make the adapter non-trivial
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)
    s = _samples()
    loss, st = pg_kl_loss(m, s, 0, sum(len(x.completion_ids) for x in s), kl_coef=0.05, chunk=4)
    loss.backward()
    grads = [p.grad for _n, p in m.named_parameters() if p.requires_grad and p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    assert st["kl_sum"] >= 0


def test_token_budget_batches_cover_all():
    s = _samples() * 5
    assert sum(len(b) for b in token_budget_batches(s, max_tokens=20)) == len(s)


def test_posterior_json_sums_to_100_and_parses():
    from ccrl.prompts.parse import parse_posterior
    from ccrl.train.sft import posterior_json
    codes = ["AA", "AB", "AC", "AD"]
    t = posterior_json(codes, np.array([0.333, 0.333, 0.2, 0.134]))
    assert sum(json.loads(t)["p"].values()) == 100 and parse_posterior(t, codes).ok
