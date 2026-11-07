"""Supervised baselines (Track B comparators), one masked-loss SFT trainer.

    ccrl train sft --config config/train/sft_kb_bayes.yaml --seed 0

Targets (the assistant turn is the posterior JSON in the panel's schema):
  kb_bayes     the corpus reference posterior (Qiu-style Bayesian teaching)
  distill      the teacher's (Kimi K3) archived answers on the 2,000 distillation items
               (read from the panel ledger export; SFT-from-frontier)
  bct          bias-augmented consistency training, posterior version: the
               model's OWN answer on the clean rendering x becomes the target
               for a transformed rendering T(x) (Chua et al.; Irpan et al.)
Loss is on the assistant tokens only (prompt masked).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from ccrl.train.common import JsonlLogger, encode_chat, load_model_and_tokenizer, set_seed
from ccrl.train.data import PairedSampler


def posterior_json(codes: list[str], p: np.ndarray) -> str:
    """Integer percentages summing to 100 (largest-remainder rounding)."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    raw = p * 100
    flo = np.floor(raw).astype(int)
    rem = 100 - int(flo.sum())
    for i in np.argsort(-(raw - flo))[:rem]:
        flo[i] += 1
    return json.dumps({"dx": codes[int(np.argmax(p))], "p": {c: int(v) for c, v in zip(codes, flo)}},
                      separators=(", ", ": "))


def kb_bayes_examples(sampler: PairedSampler, n: int) -> list[list[dict]]:
    from ccrl.data import ddxplus as dx
    lik = dx.load_likelihood()
    out = []
    for gp in sampler.sample(n, 1, 1):
        toks = gp.case.meta["tokens"]
        for u in (gp.x[0], gp.xp[0]):
            out.append([{"role": "system", "content": gp.system}, {"role": "user", "content": u},
                        {"role": "assistant", "content": posterior_json(gp.codes, lik.posterior(toks))}])
    return out


def distill_examples(export_dir: str) -> list[list[dict]]:
    import pandas as pd
    from ccrl.panel import items as I
    from ccrl.workspace import Workspace
    post = pd.read_parquet(Path(export_dir) / "posteriors.parquet")
    post = post[(post["job"] == "distill") & post["parse_ok"]]
    ws = Workspace(False)
    items, systems = I.load_items(ws), I.load_systems(ws)
    out = []
    for r in post.itertuples():
        it = items[r.item_id]
        codes = [c for c, _ in I.candidates_of(it)]
        out.append([{"role": "system", "content": systems[it["system"]]}, {"role": "user", "content": it["turns"][0]},
                    {"role": "assistant", "content": posterior_json(codes, np.asarray(r.probs))}])
    return out


def train_sft(cfg: dict[str, Any], examples: list[list[dict]]) -> Path:
    import torch

    set_seed(int(cfg.get("seed", 0)))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = JsonlLogger(out_dir / "train_log.jsonl")
    model, tok = load_model_and_tokenizer(cfg, trainable=True)
    kw = cfg.get("chat_template_kwargs", {"enable_thinking": False})
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    data = []
    for msgs in examples:
        p = encode_chat(tok, msgs[:-1], kw)
        c = list(tok(msgs[-1]["content"], add_special_tokens=False)["input_ids"]) + [tok.eos_token_id]
        data.append((p, c))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-4)))
    bs = int(cfg.get("batch_size", 8))
    epochs = float(cfg.get("epochs", 1.0))
    total = int(len(data) * epochs / bs)
    rng = random.Random(int(cfg.get("seed", 0)))
    dev = next(model.parameters()).device
    step = 0
    model.train()
    while step < total:
        rng.shuffle(data)
        for b in range(0, len(data) - bs + 1, bs):
            batch = data[b:b + bs]
            w = max(len(p) + len(c) for p, c in batch)
            ids = torch.full((bs, w), pad, dtype=torch.long)
            lab = torch.full((bs, w), -100, dtype=torch.long)
            att = torch.zeros((bs, w), dtype=torch.long)
            for i, (p, c) in enumerate(batch):
                s = p + c
                ids[i, :len(s)] = torch.tensor(s)
                lab[i, len(p):len(s)] = torch.tensor(c)
                att[i, :len(s)] = 1
            loss = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % int(cfg.get("log_every", 10)) == 0:
                log.log({"step": step, "loss": float(loss.detach())})
            if step >= total:
                break
    d = out_dir / "final"
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return d


def bct_examples(cfg: dict, sampler: PairedSampler, n: int) -> list[list[dict]]:
    """Targets = the BASE model's own (greedy-free, temperature 0.7) answer on the
    clean rendering x, attached to the transformed rendering x' (BCT)."""
    from ccrl.prompts.parse import parse_posterior
    from ccrl.train.grpo import HFGenerator

    model, tok = load_model_and_tokenizer({**cfg, "init_adapter": None}, trainable=False)
    gen = HFGenerator(model, tok, 512, 0.7, 0.95, 16, cfg.get("chat_template_kwargs", {"enable_thinking": False}))
    groups = sampler.sample(n, 1, 1)
    outs = gen([[{"role": "system", "content": g.system}, {"role": "user", "content": g.x[0]}] for g in groups])
    ex = []
    for g, (_p, _c, text, _l) in zip(groups, outs):
        r = parse_posterior(text, g.codes)
        if r.ok and r.informative:
            target = posterior_json(g.codes, r.vector)
            for u in (g.x[0], g.xp[0]):
                ex.append([{"role": "system", "content": g.system}, {"role": "user", "content": u},
                           {"role": "assistant", "content": target}])
    del model
    return ex
