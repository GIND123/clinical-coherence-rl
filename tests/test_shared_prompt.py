"""P1 (plan_final.md §3.4): the shared-prompt training pass must give the same log-probs,
entropies and LoRA gradients as the unshared pass, and the same GRPO loss."""
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from ccrl.train.common import (Sample, completion_logprobs_lowmem, shared_prompt_batches,  # noqa: E402
                               shared_prompt_logprobs)


def _tiny(seed=0):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                      tie_word_embeddings=False, attn_implementation="sdpa")
    m = get_peft_model(Qwen3ForCausalLM(cfg).float(),
                       LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"))
    with torch.no_grad():                    # LoRA B starts at 0; make the adapter matter
        for n, p in m.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.2)
    return m


def _samples(seed=1):
    r = random.Random(seed)
    prompts = [[r.randrange(1, 97) for _ in range(n)] for n in (11, 7, 15)]
    out = []
    for p, k in zip(prompts, (4, 2, 3)):
        for _ in range(k):
            out.append(Sample(list(p), [r.randrange(1, 97) for _ in range(r.randint(1, 9))], r.uniform(-1, 1)))
    r.shuffle(out)
    return out


def _loss(lps, samples):
    return sum((-(torch.exp(lp - lp.detach()) * s.advantage)).sum() for lp, s in zip(lps, samples))


def _grads(m):
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}


def test_shared_matches_unshared_logprobs_and_grads():
    m = _tiny()
    m.train()
    s = _samples()
    a, ea = completion_logprobs_lowmem(m, s, 0, chunk=3)
    _loss(a, s).backward()
    ga = _grads(m)
    m.zero_grad(set_to_none=True)
    b, eb = shared_prompt_logprobs(m, s, 0, chunk=3)
    _loss(b, s).backward()
    gb = _grads(m)
    for x, y, u, v in zip(a, b, ea, eb):
        assert x.shape == y.shape
        assert torch.allclose(x, y, atol=1e-5), (x - y).abs().max()
        assert torch.allclose(u, v, atol=1e-5)
    assert ga.keys() == gb.keys() and len(ga) > 0
    for n in ga:
        assert torch.allclose(ga[n], gb[n], atol=1e-5, rtol=1e-4), (n, (ga[n] - gb[n]).abs().max())


def test_shared_matches_under_disabled_adapter():
    m = _tiny()
    s = _samples(3)
    with torch.no_grad(), m.disable_adapter():
        a, _ = completion_logprobs_lowmem(m, s, 0)
        b, _ = shared_prompt_logprobs(m, s, 0)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5)


def test_single_token_completion_and_single_prompt():
    m = _tiny()
    s = [Sample([5, 6, 7], [9]), Sample([5, 6, 7], [9, 10, 11])]
    a, _ = completion_logprobs_lowmem(m, s, 0)
    b, _ = shared_prompt_logprobs(m, s, 0)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5)


def test_refuses_gradient_checkpointing():
    m = _tiny()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    with pytest.raises(ValueError, match="checkpointing"):
        shared_prompt_logprobs(m, _samples(), 0)


def test_batches_keep_prompt_groups_together():
    s = _samples(5) * 3                          # several groups, repeated prompts
    bs = shared_prompt_batches(s, max_tokens=40)
    assert sorted(id(x) for b in bs for x in b) == sorted(id(x) for x in s)
    seen = {}
    for bi, b in enumerate(bs):
        for x in b:
            assert seen.setdefault(tuple(x.prompt_ids), bi) == bi


def test_pg_kl_loss_shared_equals_unshared():
    from ccrl.train.grpo import pg_kl_loss
    m = _tiny()
    m.train()
    s = _samples(7)
    n = sum(len(x.completion_ids) for x in s)
    la, sa = pg_kl_loss(m, s, 0, n, 0.05, 4, None, shared_prompt=False)
    la.backward()
    ga = _grads(m)
    m.zero_grad(set_to_none=True)
    lb, sb = pg_kl_loss(m, s, 0, n, 0.05, 4, None, shared_prompt=True)
    lb.backward()
    gb = _grads(m)
    assert torch.allclose(la, lb, atol=1e-6)
    assert abs(sa["kl_sum"] - sb["kl_sum"]) < 1e-5 and sa["kl_sum"] > 0
    for k in ga:
        assert torch.allclose(ga[k], gb[k], atol=1e-5, rtol=1e-4)
