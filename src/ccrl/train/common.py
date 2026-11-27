"""Shared training utilities (torch / transformers / peft imported lazily).

Adapted from the team's DEFER-Dx trainer (github.com/GIND123/Clinical-Agent-Stanford,
Apache-2.0), where these pieces ran on Qwen3-8B on one GPU:
  * `completion_logprobs_lowmem` applies the frozen output layer only at
    completion positions, in chunks, with a custom backward — memory scales
    with the backbone, not the 152k-token vocabulary;
  * `token_budget_batches` packs micro-batches by padded token count;
  * sequences are rendered with the chat template exactly as at generation.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    prompt_ids: list[int]
    completion_ids: list[int]
    advantage: float = 0.0
    rollout_logprobs: list[float] | None = None   # sampling engine's log-probs (vLLM) for truncated IS


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_chat(tokenizer, messages: list[dict[str, str]], chat_template_kwargs: dict | None = None,
                add_generation_prompt: bool = True) -> list[int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                         **(chat_template_kwargs or {}))
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def load_model_and_tokenizer(cfg: dict[str, Any], trainable: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.get("init_adapter") or cfg["model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
                                                 device_map=cfg.get("device_map", "auto"),
                                                 attn_implementation=cfg.get("attn_implementation"))
    lora = cfg.get("lora") or {}
    if cfg.get("init_adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["init_adapter"], is_trainable=trainable)
    elif trainable:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=int(lora.get("r", 32)), lora_alpha=int(lora.get("alpha", 64)),
                                                 lora_dropout=float(lora.get("dropout", 0.0)),
                                                 target_modules=lora.get("target_modules", "all-linear"),
                                                 task_type="CAUSAL_LM"))
    if trainable and cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tok


def pad_batch(samples: list[Sample], pad_id: int):
    import torch

    seqs = [s.prompt_ids + s.completion_ids for s in samples]
    width = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    spans = []
    for i, (s, seq) in enumerate(zip(samples, seqs)):
        ids[i, : len(seq)] = torch.tensor(seq)
        mask[i, : len(seq)] = 1
        spans.append((len(s.prompt_ids), len(s.completion_ids)))
    return ids, mask, spans


def completion_logprobs(model, samples: list[Sample], pad_id: int, with_entropy: bool = False):
    """Reference implementation (full logits). Used by tests to check the low-memory one."""
    import torch

    device = next(model.parameters()).device
    ids, mask, spans = pad_batch(samples, pad_id)
    ids, mask = ids.to(device), mask.to(device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    lps, ents = [], []
    for i, (start, length) in enumerate(spans):
        lg = logits[i, start - 1: start - 1 + length].float()
        lsm = torch.log_softmax(lg, -1)
        lps.append(lsm.gather(-1, ids[i, start: start + length].unsqueeze(-1)).squeeze(-1))
        if with_entropy:
            ents.append(-(lsm.exp() * lsm).sum(-1))
    return lps, (ents if with_entropy else None)


def _frozen_head_fn():
    import torch

    class FrozenHeadLogprob(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h, weight, target, chunk):
            lps, ents = [], []
            for c in range(0, h.shape[0], chunk):
                z = (h[c: c + chunk] @ weight.T).float()
                lse = torch.logsumexp(z, -1)
                lps.append(z.gather(-1, target[c: c + chunk, None])[:, 0] - lse)
                ents.append(lse - (torch.softmax(z, -1) * z).sum(-1))
            ctx.save_for_backward(h, weight, target)
            ctx.chunk = chunk
            ent = torch.cat(ents)
            ctx.mark_non_differentiable(ent)
            return torch.cat(lps), ent

        @staticmethod
        def backward(ctx, g_lp, g_ent):
            h, weight, target = ctx.saved_tensors
            gh = torch.empty_like(h)
            for c in range(0, h.shape[0], ctx.chunk):
                g = g_lp[c: c + ctx.chunk].float()
                gz = torch.softmax((h[c: c + ctx.chunk] @ weight.T).float(), -1) * (-g[:, None])
                gz.scatter_add_(-1, target[c: c + ctx.chunk, None], g[:, None])
                gh[c: c + ctx.chunk] = (gz.to(weight.dtype) @ weight).to(h.dtype)
            return gh, None, None, None

    return FrozenHeadLogprob


_FHL = None


def completion_logprobs_lowmem(model, samples: list[Sample], pad_id: int, chunk: int = 2048):
    """(log-probs, entropies) per sample; requires a frozen lm_head (LoRA on all-linear)."""
    global _FHL
    if _FHL is None:
        _FHL = _frozen_head_fn()
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone, head = m.model, m.lm_head
    if head.weight.requires_grad:
        raise ValueError("completion_logprobs_lowmem needs a frozen lm_head")
    device = head.weight.device
    ids, mask, spans = pad_batch(samples, pad_id)
    ids, mask = ids.to(device), mask.to(device)
    hidden = backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    lps, ents = [], []
    for i, (start, length) in enumerate(spans):
        lp, ent = _FHL.apply(hidden[i, start - 1: start - 1 + length], head.weight, ids[i, start: start + length], chunk)
        lps.append(lp)
        ents.append(ent)
    return lps, ents


def _cache_layers(cache) -> list[tuple]:
    """Per-layer (keys, values) of a DynamicCache, across transformers 4.5x and 5.x."""
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [tuple(kv[:2]) for kv in cache]          # legacy tuple of tuples


def _expanded_cache(cache, index):
    """A new DynamicCache whose batch row r is row index[r] of `cache` (differentiable)."""
    from transformers import DynamicCache

    out = DynamicCache()
    for li, (k, v) in enumerate(_cache_layers(cache)):
        out.update(k.index_select(0, index), v.index_select(0, index), li)
    return out


def shared_prompt_logprobs(model, samples: list[Sample], pad_id: int, chunk: int = 2048):
    """Same result as `completion_logprobs_lowmem`, but each distinct prompt is run once.

    GRPO trains several rollouts per prompt. Here the prompts are run once (right-padded), their
    key/value cache is expanded to the rollouts, and the completions are run on top of it with the
    positions they have in the unshared sequence. Gradients from every rollout reach the shared
    prompt through the expansion, so the loss and its gradient equal the unshared ones up to
    floating-point order (tests/test_shared_prompt.py). Needs gradient checkpointing off, because
    transformers disables the cache under checkpointing."""
    import torch

    global _FHL
    if _FHL is None:
        _FHL = _frozen_head_fn()
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone, head = m.model, m.lm_head
    if head.weight.requires_grad:
        raise ValueError("shared_prompt_logprobs needs a frozen lm_head")
    if getattr(backbone, "gradient_checkpointing", False) and backbone.training:
        raise ValueError("shared_prompt_logprobs needs gradient checkpointing off")
    device = head.weight.device
    groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(samples):
        if not s.completion_ids:
            raise ValueError("empty completion")
        groups.setdefault(tuple(s.prompt_ids), []).append(i)
    prompts = list(groups)
    plen = [len(p) for p in prompts]
    P = max(plen)
    pid = torch.full((len(prompts), P), pad_id, dtype=torch.long)
    pmask = torch.zeros((len(prompts), P), dtype=torch.long)
    for j, p in enumerate(prompts):
        pid[j, : len(p)] = torch.tensor(p)
        pmask[j, : len(p)] = 1
    pid, pmask = pid.to(device), pmask.to(device)
    ppos = torch.arange(P, device=device).expand(len(prompts), P)
    out = backbone(input_ids=pid, attention_mask=pmask, position_ids=ppos, use_cache=True)
    h_last = out.last_hidden_state[torch.arange(len(prompts), device=device),
                                   torch.tensor(plen, device=device) - 1]
    owner, rows = [], []
    for j, p in enumerate(prompts):
        for i in groups[p]:
            owner.append(j)
            rows.append(i)
    own = torch.tensor(owner, device=device)
    clen = [len(samples[i].completion_ids) for i in rows]
    C = max(clen)
    cid = torch.full((len(rows), C), pad_id, dtype=torch.long)
    cmask = torch.zeros((len(rows), C), dtype=torch.long)
    for r, i in enumerate(rows):
        cid[r, : clen[r]] = torch.tensor(samples[i].completion_ids)
        cmask[r, : clen[r]] = 1
    cid, cmask = cid.to(device), cmask.to(device)
    cpos = torch.tensor(plen, device=device)[own][:, None] + torch.arange(C, device=device)[None]
    out2 = backbone(input_ids=cid, attention_mask=torch.cat([pmask[own], cmask], 1), position_ids=cpos,
                    past_key_values=_expanded_cache(out.past_key_values, own), use_cache=True)
    hc = out2.last_hidden_state
    lps: list = [None] * len(samples)
    ents: list = [None] * len(samples)
    for r, i in enumerate(rows):
        h = torch.cat([h_last[owner[r]][None], hc[r, : clen[r] - 1]], 0)
        lps[i], ents[i] = _FHL.apply(h, head.weight, cid[r, : clen[r]], chunk)
    return lps, ents


def shared_prompt_batches(samples: list[Sample], max_tokens: int) -> list[list[Sample]]:
    """Micro-batches that keep each prompt's rollouts together. A batch costs its distinct prompts
    plus its completions (padded widths); a prompt group larger than the budget gets a batch alone."""
    groups: dict[tuple, list[Sample]] = {}
    for s in samples:
        groups.setdefault(tuple(s.prompt_ids), []).append(s)
    order = sorted(groups.values(), key=lambda g: len(g[0].prompt_ids) + max(len(s.completion_ids) for s in g),
                   reverse=True)
    batches: list[list[Sample]] = []
    cur: list[list[Sample]] = []
    for g in order:
        trial = cur + [g]
        P = max(len(x[0].prompt_ids) for x in trial)
        C = max(len(s.completion_ids) for x in trial for s in x)
        n_r = sum(len(x) for x in trial)
        if cur and len(trial) * P + n_r * C > max_tokens:
            batches.append([s for x in cur for s in x])
            cur = [g]
        else:
            cur = trial
    if cur:
        batches.append([s for x in cur for s in x])
    return batches


def token_budget_batches(samples: list[Sample], max_tokens: int, max_samples: int = 64) -> list[list[Sample]]:
    order = sorted(samples, key=lambda s: len(s.prompt_ids) + len(s.completion_ids), reverse=True)
    batches: list[list[Sample]] = []
    cur: list[Sample] = []
    width = 0
    for s in order:
        n = len(s.prompt_ids) + len(s.completion_ids)
        w = max(width, n)
        if cur and (w * (len(cur) + 1) > max_tokens or len(cur) >= max_samples):
            batches.append(cur)
            cur, w = [], n
        cur.append(s)
        width = w
    if cur:
        batches.append(cur)
    return batches


def lr_lambda(warmup: int, total: int, schedule: str = "constant"):
    def f(s: int) -> float:
        if s < warmup:
            return (s + 1) / max(1, warmup)
        if schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - warmup))))
        return 1.0
    return f


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
