"""GPU check of the shared-prompt training pass (plan_final.md P1 / run 0), before any paid run uses it.

    python -m ccrl.train.check_shared --config config/train/ccr_acc_fid.yaml [--out runs/env/shared_check.json]

On the real model (Qwen3-4B, bf16, LoRA with a non-zero adapter) and one step's worth of real
training prompts (16 patients, 4 + 4 + 2 rollouts, random completions of the measured length),
it compares the unshared and the shared pass:
  * per-token log-probs: max and mean absolute difference;
  * LoRA gradient: cosine similarity and relative norm difference;
  * seconds for the whole step's forward + backward, and peak memory (the unshared pass runs here
    without checkpointing, in smaller micro-batches; its production time is the logged 66 s).
Pass: gradient cosine >= 0.999 and relative norm difference <= max(1%, 1.5 x the bf16 noise floor,
i.e. the difference between two groupings of the unshared pass itself). Exit code 0 = pass, 4 = fail
(the queue then trains with `--no-shared`, the unshared pass with gradient checkpointing).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def main(argv=None) -> int:
    import torch

    from ccrl.train.common import (Sample, completion_logprobs_lowmem, encode_chat, load_model_and_tokenizer,
                                   shared_prompt_batches, shared_prompt_logprobs, token_budget_batches)
    from ccrl.train.data import PairedSampler
    from ccrl.train.grpo import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/train/ccr_acc_fid.yaml")
    ap.add_argument("--out", default="runs/env/shared_check.json")
    ap.add_argument("--completion-len", type=int, default=340)
    ap.add_argument("--mem-fraction", type=float, default=0.62,
                    help="cap on this process's GPU memory: leaves vLLM's 30%% share and a margin, as in training")
    a = ap.parse_args(argv)
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    cfg = load_config(a.config)
    cfg["gradient_checkpointing"] = False
    model, tok = load_model_and_tokenizer(cfg, trainable=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 1e-3)
    smp = PairedSampler(cfg.get("dataset", "ddxplus"), 0, cfg.get("transforms"), cfg.get("contrast", "triage"), False,
                        cfg.get("pool_path"), cfg.get("phrasebank_path"), cfg.get("cue_bank", "panel"))
    kw = cfg.get("chat_template_kwargs", {"enable_thinking": False})
    r = random.Random(0)
    samples = []
    for gp in smp.sample(int(cfg.get("patients_per_step", 16)), int(cfg.get("group_size", 4)),
                         int(cfg.get("contrast_size", 2))):
        for u in gp.x + gp.xp + gp.xm:
            p = encode_chat(tok, [{"role": "system", "content": gp.system}, {"role": "user", "content": u}], kw)
            n = max(8, int(r.gauss(a.completion_len, 60)))
            samples.append(Sample(p, [r.randrange(1000, 30000) for _ in range(n)], r.uniform(-1, 1)))
    params = [p for p in model.parameters() if p.requires_grad]
    n_tok = sum(len(s.completion_ids) for s in samples)
    model.train()

    def step(shared: bool, tokens: int = 4096, per_batch: int = 2):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        lps = {}
        mbs = shared_prompt_batches(samples, tokens) if shared else \
            token_budget_batches(samples, 4096, per_batch)      # checkpointing is off here: small batches
        for mb in mbs:
            fn = shared_prompt_logprobs if shared else completion_logprobs_lowmem
            out, _ = fn(model, mb, tok.pad_token_id or tok.eos_token_id, int(cfg.get("logprob_chunk", 2048)))
            loss = sum((-(torch.exp(lp - lp.detach()) * s.advantage)).sum() for lp, s in zip(out, mb)) / n_tok
            loss.backward()
            for s, lp in zip(mb, out):
                lps[id(s)] = lp.detach().float().cpu()
        torch.cuda.synchronize()
        sec = time.time() - t0
        g = torch.cat([p.grad.detach().float().flatten() for p in params if p.grad is not None])
        return sec, torch.cuda.max_memory_allocated() / 2**30, g, lps

    s_un, m_un, g_un, lp_un = step(False)
    # noise floor: the same unshared pass with a different micro-batch grouping (1 sequence per
    # batch). bf16 kernels give slightly different sums for different batch shapes, so this is the
    # difference that already exists between two runs of the registered pass
    _s1, _m1, g_un1, lp_un1 = step(False, per_batch=1)
    floor_rel = float((g_un - g_un1).norm() / g_un.norm())
    floor_cos = float(torch.nn.functional.cosine_similarity(g_un, g_un1, dim=0))
    floor_lp = float(torch.cat([(lp_un[k] - lp_un1[k]).abs() for k in lp_un]).mean())
    tokens = None
    for tk in (int(cfg.get("shared_micro_batch_tokens", 4096)), 3072, 2048, 1536):
        try:
            s_sh, m_sh, g_sh, lp_sh = step(True, tk)
            tokens = tk
            break
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"shared pass: {tk} tokens per micro-batch ran out of memory; trying smaller", flush=True)
    if tokens is None:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"pass": False, "reason": "out of memory at 1536 tokens"}), encoding="utf-8")
        return 4
    d = torch.cat([(lp_un[k] - lp_sh[k]).abs() for k in lp_un])
    cos = float(torch.nn.functional.cosine_similarity(g_un, g_sh, dim=0))
    rel = float((g_un - g_sh).norm() / g_un.norm())
    res = {"n_sequences": len(samples), "n_distinct_prompts": len({tuple(s.prompt_ids) for s in samples}),
           "completion_tokens": n_tok, "prompt_tokens_mean": sum(len(s.prompt_ids) for s in samples) / len(samples),
           "sec_unshared": s_un, "sec_shared": s_sh, "speedup": s_un / s_sh, "mem_gb_unshared": m_un,
           "mem_gb_shared": m_sh, "logprob_absdiff_max": float(d.max()), "logprob_absdiff_mean": float(d.mean()),
           "grad_cosine": cos, "grad_rel_diff": rel, "shared_micro_batch_tokens": tokens,
           "mem_fraction": a.mem_fraction, "floor_grad_rel_diff": floor_rel, "floor_grad_cosine": floor_cos,
           "floor_logprob_absdiff_mean": floor_lp,
           # equivalent if the shared pass differs from the unshared one by no more than regrouping the
           # unshared pass does (x1.5), or by at most 1%; the fp32 CPU test checks exact equality
           "pass": bool(cos >= 0.999 and rel <= max(0.01, 1.5 * floor_rel))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
