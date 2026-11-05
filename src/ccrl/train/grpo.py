"""Paired-group GRPO with the coherence-gated reward (Track B).

    ccrl train grpo --config config/train/ccr_acc_fid.yaml --seed 0

One step:
  1. sample B patients; per patient a paired group of prompts (x: G, x': G, x-: H)
  2. generate completions with the current LoRA policy (transformers `generate`;
     thinking off, <= 512 new tokens — a posterior JSON). This backend is
     CPU-testable and fits the plan's 10-14 GPU-h per 4B run. For more speed,
     port the colocated vLLM loop (sleep/wake, frozen-weight parking, LoRA
     hot-swap, truncated IS) from DEFER-Dx `training/grpo_vllm.py`; `Sample`
     already carries `rollout_logprobs` and `pg_kl_loss` the TIS weight for it.
  3. parse each completion's stated posterior (same parser as the panel)
  4. rewards for the whole batch (ccrl.reward.coherence.batch_rewards):
     M and the sensitivity gate are batch-level; I, fidelity and accuracy per rollout
  5. advantages standardised within each PAIRED group (2G + H rollouts); groups
     with no reward spread are skipped (DAPO dynamic sampling)
  6. loss = token-level policy gradient (on-policy, one optimiser step per batch)
          + kl_coef * KL(pi || pi_ref) (k3 estimator; reference = LoRA disabled)
Safeguards (plan §3.6): per-step logging of M, between-patient JS, entropy,
invalid fraction; early stop when M falls > 30% below its first-5-step mean.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ccrl.prompts.parse import parse_posterior
from ccrl.reward.coherence import PatientGroup, RewardConfig, advantages, batch_rewards
from ccrl.train.common import (JsonlLogger, Sample, completion_logprobs_lowmem, encode_chat, load_model_and_tokenizer,
                               lr_lambda, set_seed, token_budget_batches)
from ccrl.train.data import PairedSampler


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, overrides: dict | None = None) -> dict:
    """YAML with `base:` inheritance; nested dicts (reward, generation, ...) are deep-merged."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    own_name = cfg.get("name")
    base = cfg.pop("base", None)
    if base:
        cfg = _deep_merge(load_config(str(Path(path).parent / base)), cfg)
    cfg = _deep_merge(cfg, overrides or {})
    cfg["name"] = own_name or Path(path).stem
    return cfg


def reward_config(cfg: dict) -> RewardConfig:
    r = cfg.get("reward", {})
    return RewardConfig(**{k: v for k, v in r.items() if k in RewardConfig.__dataclass_fields__})


# --------------------------------------------------------------------- generation
class HFGenerator:
    def __init__(self, model, tok, max_new_tokens: int, temperature: float, top_p: float, batch_size: int,
                 chat_kwargs: dict):
        self.model, self.tok = model, tok
        self.max_new, self.temp, self.top_p, self.bs = max_new_tokens, temperature, top_p, batch_size
        self.kw = chat_kwargs
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.stop = {int(tok.eos_token_id)} if tok.eos_token_id is not None else set()

    def __call__(self, convs: list[list[dict]]) -> list[tuple[list[int], list[int], str, list[float] | None]]:
        import torch

        prompts = [encode_chat(self.tok, c, self.kw) for c in convs]
        out = []
        dev = next(self.model.parameters()).device
        self.model.eval()
        for s in range(0, len(prompts), self.bs):
            chunk = prompts[s:s + self.bs]
            w = max(len(p) for p in chunk)
            ids = torch.full((len(chunk), w), self.pad, dtype=torch.long)
            att = torch.zeros((len(chunk), w), dtype=torch.long)
            for i, p in enumerate(chunk):
                ids[i, w - len(p):] = torch.tensor(p)
                att[i, w - len(p):] = 1
            with torch.no_grad():
                seq = self.model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), do_sample=True,
                                          temperature=self.temp, top_p=self.top_p, max_new_tokens=self.max_new,
                                          pad_token_id=self.pad, eos_token_id=sorted(self.stop) or None)
            for i, p in enumerate(chunk):
                comp = []
                for t in seq[i, w:].tolist():
                    comp.append(t)
                    if t in self.stop:
                        break
                out.append((list(p), comp, self.tok.decode(comp, skip_special_tokens=True), None))
        return out


# --------------------------------------------------------------------- loss
def pg_kl_loss(model, samples: list[Sample], pad_id: int, n_tok_total: int, kl_coef: float, chunk: int,
               tis_cap: float | None = None):
    """On-policy PG (ratio = 1 with gradient) + k3 KL to the adapter-disabled reference."""
    import torch

    lps, ents = completion_logprobs_lowmem(model, samples, pad_id, chunk)
    ref = None
    if kl_coef > 0:
        with torch.no_grad(), model.disable_adapter():
            ref, _ = completion_logprobs_lowmem(model, samples, pad_id, chunk)
    total = 0.0
    st = {"entropy_sum": 0.0, "kl_sum": 0.0}
    for k, (s, lp, ent) in enumerate(zip(samples, lps, ents)):
        adv = torch.as_tensor(float(s.advantage), dtype=lp.dtype, device=lp.device)
        ratio = torch.exp(lp - lp.detach())
        w = 1.0
        if tis_cap and s.rollout_logprobs is not None and len(s.rollout_logprobs) == len(s.completion_ids):
            rl = torch.tensor(s.rollout_logprobs, dtype=lp.dtype, device=lp.device)
            ok = torch.isfinite(rl)
            w = torch.exp(torch.where(ok, lp.detach() - rl, torch.zeros_like(rl))).clamp(max=tis_cap)
        term = -(ratio * adv) * w
        if ref is not None:
            d = ref[k] - lp
            kl = torch.exp(d) - d - 1.0          # k3 estimator, >= 0
            term = term + kl_coef * kl
            st["kl_sum"] += float(kl.detach().sum())
        total = total + term.sum()
        st["entropy_sum"] += float(ent.sum())
    return total / max(1, n_tok_total), st


# --------------------------------------------------------------------- trainer
def train(cfg: dict[str, Any]) -> Path:
    import torch

    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log = JsonlLogger(out_dir / "train_log.jsonl")
    rcfg = reward_config(cfg)
    G = int(cfg.get("group_size", 4))
    H = int(cfg.get("contrast_size", max(1, G // 2)))
    B = int(cfg.get("patients_per_step", 16))
    steps = int(cfg.get("steps", 300))
    sampler = PairedSampler(cfg.get("dataset", "ddxplus"), seed, cfg.get("transforms"),
                            cfg.get("contrast", "triage"), bool(cfg.get("label_free", False)),
                            cfg.get("pool_path"), cfg.get("phrasebank_path"))
    model, tok = load_model_and_tokenizer(cfg, trainable=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_cfg = cfg.get("generation", {})
    gen = HFGenerator(model, tok, int(gen_cfg.get("max_new_tokens", 512)), float(gen_cfg.get("temperature", 1.0)),
                      float(gen_cfg.get("top_p", 1.0)), int(gen_cfg.get("batch_size", 32)),
                      cfg.get("chat_template_kwargs", {"enable_thinking": False}))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-5)), betas=(0.9, 0.99),
                            weight_decay=float(cfg.get("weight_decay", 0.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(int(cfg.get("warmup_steps", 10)), steps,
                                                             cfg.get("lr_schedule", "constant")))
    start = 1
    ck = sorted((out_dir / "checkpoints").glob("step_*/trainer_state.pt"))
    if ck and cfg.get("resume", True):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        d = ck[-1].parent
        set_peft_model_state_dict(model, load_file(str(d / "adapter_model.safetensors")))
        state = torch.load(d / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start = int(state["step"]) + 1
        sampler.r.seed(seed * 100003 + start)
        print(f"[grpo] resumed from {d} at step {start}", flush=True)
    m_hist: list[float] = []
    K_of = None
    for step in range(start, steps + 1):
        t0 = time.time()
        groups = sampler.sample(B, G, H)
        convs, index = [], []
        for gi, gp in enumerate(groups):
            for part, texts in (("x", gp.x), ("xp", gp.xp), ("xm", gp.xm)):
                for u in texts:
                    convs.append([{"role": "system", "content": gp.system}, {"role": "user", "content": u}])
                    index.append((gi, part))
        outs = gen(convs)
        parsed: dict[tuple[int, str], list] = {}
        invalid = 0
        for (gi, part), (_p, _c, text, _lp) in zip(index, outs):
            res = parse_posterior(text, groups[gi].codes)
            if not res.ok:
                invalid += 1
            parsed.setdefault((gi, part), []).append(res.vector if res.ok else None)
        pgs = []
        for gi, gp in enumerate(groups):
            K_of = len(gp.codes)
            pgs.append(PatientGroup(parsed.get((gi, "x"), []), parsed.get((gi, "xp"), []), parsed.get((gi, "xm"), []),
                                    gold=gp.gold, added_loglik=gp.added_loglik))
        res, M = batch_rewards(pgs, rcfg, K_of)
        samples: list[Sample] = []
        kept = skipped = 0
        k = 0
        for gi, gr in enumerate(res):
            n = len(gr.total)
            adv = advantages(gr.total, bool(cfg.get("normalize_std", True)), float(cfg.get("min_group_std", 1e-4)))
            rows = list(range(k, k + n))
            k += n
            if adv is None:
                skipped += 1
                continue
            kept += 1
            for a, row in zip(adv, rows):
                p_ids, c_ids, _t, lp = outs[row]
                if c_ids:
                    samples.append(Sample(p_ids, c_ids, float(a), lp))
        stats = {"entropy_sum": 0.0, "kl_sum": 0.0}
        n_tok = sum(len(s.completion_ids) for s in samples)
        if samples:
            model.train()
            for mb in token_budget_batches(samples, int(cfg.get("micro_batch_tokens", 16384)),
                                           int(cfg.get("micro_batch_max_samples", 8))):
                loss, st = pg_kl_loss(model, mb, pad, n_tok, float(cfg.get("kl_coef", 0.01)),
                                      int(cfg.get("logprob_chunk", 2048)), cfg.get("tis_cap"))
                loss.backward()
                for kk, v in st.items():
                    stats[kk] += v
            gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0))))
            opt.step()
            opt.zero_grad(set_to_none=True)
        else:
            gn = 0.0
        sched.step()
        allr = np.concatenate([g.total for g in res])
        coh = np.concatenate([g.coh for g in res])
        acc = np.concatenate([g.acc for g in res])
        fid = np.concatenate([g.fid for g in res])
        gate_s = res[0].terms.get("gate_S") if res else None
        row = {"step": step, "sec": round(time.time() - t0, 1), "reward": float(allr.mean()), "r_coh": float(coh.mean()),
               "r_acc": float(acc.mean()), "r_fid": float(fid.mean()), "M": float(M), "gate_S": gate_s,
               "invalid_frac": invalid / max(1, len(outs)), "kept_groups": kept, "skipped_groups": skipped,
               "tokens": n_tok, "entropy": stats["entropy_sum"] / max(1, n_tok), "kl": stats["kl_sum"] / max(1, n_tok),
               "grad_norm": gn, "lr": sched.get_last_lr()[0]}
        log.log(row)
        m_hist.append(float(M))
        if len(m_hist) > 5:
            base_m = float(np.mean(m_hist[:5]))
            if base_m > 0 and M < 0.7 * base_m and cfg.get("collapse_stop", True):
                log.log({"step": step, "event": "early_stop_M_collapse", "M": float(M), "baseline_M": base_m})
                _save(model, tok, opt, sched, out_dir, step, final=True)
                return out_dir / "final"
        if cfg.get("save_every") and step % int(cfg["save_every"]) == 0:
            _save(model, tok, opt, sched, out_dir, step)
    return _save(model, tok, opt, sched, out_dir, steps, final=True)


def _save(model, tok, opt, sched, out_dir: Path, step: int, final: bool = False) -> Path:
    import shutil
    import torch

    d = out_dir / ("final" if final else f"checkpoints/step_{step:04d}")
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)
    tok.save_pretrained(d)
    if not final:
        torch.save({"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, d / "trainer_state.pt")
        cks = sorted((out_dir / "checkpoints").glob("step_*"))
        for old in cks[:-2]:
            shutil.rmtree(old, ignore_errors=True)
    return d
