"""Differentiable baseline: logit-readout posteriors with JS consistency losses.

The posterior is read out in ONE forward pass: the prompt is followed by the
assistant prefix '{"dx": "' and each candidate code is scored by the exact
log-likelihood of its tokens; a softmax over the K scores is the posterior.
Training minimises, per patient,

    L = JS(q(x), q(x')) - lambda_s * min(JS(q(x), q(x-)), delta) + lambda_ce * CE(q(x), y)

(invariance, a hinge for sensitivity to the contrast, optional label term),
i.e. the R-Drop / BCT-style answer to "why not a differentiable loss?".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ccrl.train.common import JsonlLogger, encode_chat, load_model_and_tokenizer, set_seed
from ccrl.train.data import PairedSampler

PREFIX = '{"dx": "'


def code_scores(model, tok, convs: list[list[dict]], codes: list[str], chat_kwargs: dict):
    """(n, K) log-likelihood of each code continuation, differentiable."""
    import torch

    dev = next(model.parameters()).device
    code_ids = [list(tok(c, add_special_tokens=False)["input_ids"]) for c in codes]
    out = []
    for conv in convs:
        p = encode_chat(tok, conv, chat_kwargs) + list(tok(PREFIX, add_special_tokens=False)["input_ids"])
        seqs = [p + c for c in code_ids]
        w = max(len(s) for s in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((len(seqs), w), pad, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False).logits
        lsm = torch.log_softmax(logits.float(), -1)
        sc = []
        for i, c in enumerate(code_ids):
            pos = torch.arange(len(p) - 1, len(p) - 1 + len(c), device=dev)
            sc.append(lsm[i, pos, torch.tensor(c, device=dev)].sum())
        out.append(torch.stack(sc))
    return torch.stack(out)


def js_t(p, q):
    import torch
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + 1e-12) - torch.log(b + 1e-12))).sum(-1)  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def train_logit_readout(cfg: dict[str, Any]) -> Path:
    import torch

    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = JsonlLogger(out_dir / "train_log.jsonl")
    model, tok = load_model_and_tokenizer(cfg, trainable=True)
    kw = cfg.get("chat_template_kwargs", {"enable_thinking": False})
    sampler = PairedSampler(cfg.get("dataset", "ddxplus"), seed, cfg.get("transforms"), cfg.get("contrast", "triage"),
                            bool(cfg.get("label_free", False)), cfg.get("pool_path"), cfg.get("phrasebank_path"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-5)))
    lam_s, lam_ce, delta = float(cfg.get("lambda_s", 1.0)), float(cfg.get("lambda_ce", 0.0)), float(cfg.get("delta", 0.2))
    model.train()
    for step in range(1, int(cfg.get("steps", 300)) + 1):
        loss_acc = 0.0
        for gp in sampler.sample(int(cfg.get("patients_per_step", 8)), 1, 1):
            msgs = lambda u: [{"role": "system", "content": gp.system}, {"role": "user", "content": u}]  # noqa: E731
            s = code_scores(model, tok, [msgs(gp.x[0]), msgs(gp.xp[0]), msgs(gp.xm[0])], gp.codes, kw)
            q = torch.softmax(s, -1)
            loss = js_t(q[0], q[1]) - lam_s * torch.clamp(js_t(q[0], q[2]), max=delta)
            if lam_ce and gp.gold is not None:
                loss = loss - lam_ce * torch.log(q[0, gp.gold] + 1e-12)
            loss.backward()
            loss_acc += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        log.log({"step": step, "loss": loss_acc})
    d = out_dir / "final"
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return d
