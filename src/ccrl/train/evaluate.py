"""Evaluate a local model (base or LoRA adapter) on the frozen panel items.

    ccrl train eval --model Qwen/Qwen3-4B [--adapter runs/train/ccr/final] --name ccr_s0 [--set panel]
    ccrl train compare --ref runs/local_eval/rlvr_s0.parquet --new runs/local_eval/ccr_s0.parquet
    ccrl train compare --ref runs/export/posteriors.parquet --ref-model qwen3-32b                        --new runs/local_eval/q8_ccr_s0.parquet          # H6: trained vs panel model

Same items, same system/user prompts, same parser and the same estimands as
the hosted panel, so trained models and frontier models sit in one table.
Decoding is SAMPLED (temperature 0.7, top-p 0.95, per-item seed): greedy
decoding would make `retest` identical to `base` and the noise floor zero.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ccrl import paths
from ccrl.analysis.estimands import analyze
from ccrl.analysis.stats import paired_bootstrap_diff, paired_noninferiority
from ccrl.panel import items as I
from ccrl.prompts.parse import parse_posterior
from ccrl.util import read_jsonl
from ccrl.workspace import Workspace


def _row(name: str, item_set: str, iid: str, it: dict, text: str) -> dict:
    cands = I.candidates_of(it)
    kind = "s" if it["kind"] == "scores" else "p"
    r = parse_posterior(text, [c for c, _ in cands], key=kind, names=dict(cands))
    return {"job": f"local:{item_set}", "model_key": name, "effort": "default", "item_id": iid,
            "case_id": it["case_id"], "transform": it["transform"], "dataset": it["dataset"], "turn": 1,
            "parse_ok": r.ok, "informative": r.informative, "probs": r.probs, "dx": r.dx,
            "gold": it["gold"], "request_status": "done", "text": text}


def _item_seed(name: str, iid: str) -> int:
    return int(hashlib.sha256((name + iid).encode()).hexdigest()[:8], 16)


def evaluate_local(model_name: str, name: str, adapter: str | None = None, item_set: str = "panel",
                   transforms: list[str] | None = None, max_new_tokens: int = 512, temperature: float = 0.7,
                   top_p: float = 0.95, batch_size: int = 16, limit_cases: int | None = None,
                   chat_kwargs: dict | None = None, ws: Workspace | None = None, backend: str = "hf",
                   gpu_memory_utilization: float = 0.85) -> Path:
    from transformers import AutoTokenizer

    from ccrl.train.common import encode_chat

    ws = ws or Workspace(False)
    items, sets, systems = I.load_items(ws), I.load_sets(ws), I.load_systems(ws)
    ids = [i for i in sets[item_set] if not transforms or items[i]["transform"] in transforms]
    ids = [i for i in ids if items[i]["transform"] not in ("seqA", "seqB")]
    if limit_cases:
        keep: list[str] = []
        for i in ids:
            if items[i]["case_id"] not in keep:
                keep.append(items[i]["case_id"])
        keep = keep[:limit_cases]
        ids = [i for i in ids if items[i]["case_id"] in keep]
    tok = AutoTokenizer.from_pretrained(adapter or model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = chat_kwargs if chat_kwargs is not None else {"enable_thinking": False}
    dest = paths.RUNS / "local_eval" / f"{name}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if backend == "vllm":
        from ccrl.train.vllm_backend import VLLMEngine, adapter_rank
        eng = VLLMEngine(model_name, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                         gpu_memory_utilization=gpu_memory_utilization, lora_rank=adapter_rank(adapter))
        prompts = [encode_chat(tok, [{"role": "system", "content": systems[items[i]["system"]]},
                                     {"role": "user", "content": items[i]["turns"][0]}], kw) for i in ids]
        outs = eng.generate(prompts, lora_path=adapter, seeds=[_item_seed(name, i) for i in ids], logprobs=False)
        rows = [_row(name, item_set, i, items[i], text) for i, (_ids, text, _lp) in zip(ids, outs)]
        print(f"[eval {name}] {len(rows)} items (vLLM)", flush=True)
        pd.DataFrame(rows).to_parquet(dest, index=False)
        return dest
    if backend != "hf":
        raise SystemExit(f"unknown backend {backend!r} (hf | vllm)")
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    pad = tok.pad_token_id
    dev = next(model.parameters()).device
    rows = []
    for s in range(0, len(ids), batch_size):
        chunk = ids[s:s + batch_size]
        prompts = [encode_chat(tok, [{"role": "system", "content": systems[items[i]["system"]]},
                                     {"role": "user", "content": items[i]["turns"][0]}], kw) for i in chunk]
        w = max(len(p) for p in prompts)
        x = torch.full((len(chunk), w), pad, dtype=torch.long)
        a = torch.zeros((len(chunk), w), dtype=torch.long)
        for k, p in enumerate(prompts):
            x[k, w - len(p):] = torch.tensor(p)
            a[k, w - len(p):] = 1
        seed = int(hashlib.sha256((name + chunk[0]).encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        with torch.no_grad():
            samp = dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else dict(do_sample=False)
            out = model.generate(input_ids=x.to(dev), attention_mask=a.to(dev), max_new_tokens=max_new_tokens,
                                 pad_token_id=pad, **samp)
        for k, iid in enumerate(chunk):
            rows.append(_row(name, item_set, iid, items[iid], tok.decode(out[k, w:], skip_special_tokens=True)))
        print(f"[eval {name}] {min(s + batch_size, len(ids))}/{len(ids)}", flush=True)
    pd.DataFrame(rows).to_parquet(dest, index=False)
    return dest


def load_posteriors(parquet: str, model: str | None = None, job: str | None = None) -> pd.DataFrame:
    """A local-eval parquet, or the panel export (runs/export/posteriors.parquet)
    narrowed to one model and job, so trained and panel models compare on the
    same items. The panel export holds several models: `model` is then required."""
    post = pd.read_parquet(parquet)
    if model is not None:
        post = post[post["model_key"] == model]
    if job is None and "job" in post and post["job"].nunique() > 1:
        job = "panel_main" if (post["job"] == "panel_main").any() else None
    if job is not None:
        post = post[post["job"] == job]
    if post.empty:
        raise SystemExit(f"{parquet}: no rows for model={model!r} job={job!r}")
    if post["model_key"].nunique() > 1:
        raise SystemExit(f"{parquet} holds several models {sorted(post['model_key'].unique())}: pass --ref-model/--new-model")
    return post


def estimands_for(parquet: str, ws: Workspace | None = None, model: str | None = None,
                  job: str | None = None) -> dict[str, pd.DataFrame]:
    ws = ws or Workspace(False)
    post = load_posteriors(parquet, model, job)
    ci = {r["case_id"]: r for r in read_jsonl(ws.items_dir / "case_index.jsonl")}
    return analyze(post, I.load_items(ws), ci)


def compare(ref: str, new: str, margin_top1: float = 0.02, ws: Workspace | None = None,
            ref_model: str | None = None, new_model: str | None = None) -> pd.DataFrame:
    """Paired, case-level comparison of two evaluated models (H2/H3; H6 with a
    panel model as `ref`)."""
    from ccrl.analysis.stability import reliability_per_case
    a, b = estimands_for(ref, ws, ref_model), estimands_for(new, ws, new_model)
    pa, pb = a["per_case"], b["per_case"]

    def with_stability(pc: pd.DataFrame, parquet: str, model: str | None) -> pd.DataFrame:
        # Amendment 1 endpoints: retest flip and the single-vs-ensemble gap (H6, H7)
        post = load_posteriors(parquet, model)
        post = post[post["parse_ok"].astype(bool) & post["probs"].notna()] if "probs" in post else post.iloc[0:0]
        st = [reliability_per_case(g).assign(dataset=ds) for ds, g in post.groupby("dataset")] if len(post) else []
        if not st:
            return pc
        st = pd.concat(st, ignore_index=True)
        st["ens_gap_top1"] = st["ens_top1"] - st["single_top1"] if "ens_top1" in st else np.nan
        keep = [c for c in ("case_id", "dataset", "retest_flip", "ens_gap_top1") if c in st]
        return pc.merge(st[keep], on=["case_id", "dataset"], how="left")

    pa, pb = with_stability(pa, ref, ref_model), with_stability(pb, new, new_model)
    out = []
    for ds in sorted(set(pa["dataset"]) & set(pb["dataset"])):
        A = pa[pa["dataset"] == ds].set_index("case_id")
        Bv = pb[pb["dataset"] == ds].set_index("case_id")
        for col, kind in (("js_order", "lower_better"), ("js_dup", "lower_better"), ("js_para", "lower_better"),
                          ("js_cue", "lower_better"), ("floor", "lower_better"), ("retest_flip", "lower_better"),
                          ("ens_gap_top1", "lower_better"), ("sensitive", "higher_better"),
                          ("top1", "noninferiority"), ("brier", "lower_better")):
            if col not in A or col not in Bv:
                continue
            if kind == "noninferiority":
                r = paired_noninferiority(A[col], Bv[col], margin_top1)
            else:
                r = paired_bootstrap_diff(A[col], Bv[col])
            out.append({"dataset": ds, "metric": col, "test": kind, **r})
    return pd.DataFrame(out)


def tau_i_from_eval(parquet: str, q: float = 0.95) -> dict:
    """Pre-specified reward constant (OSF osf.io/4xmq7): tau_I = the 95th percentile of the
    untrained base model's retest JS (base vs verbatim retest, per case) on the pilot items."""
    from ccrl.analysis import metrics as Mx

    df = pd.read_parquet(parquet)
    df = df[df["parse_ok"] & df["transform"].isin(["base", "retest"])]
    js = []
    for _cid, g in df.groupby("case_id"):
        b, r = g[g["transform"] == "base"], g[g["transform"] == "retest"]
        if len(b) and len(r):
            js.append(Mx.js(np.asarray(b.iloc[0]["probs"]), np.asarray(r.iloc[0]["probs"])))
    if not js:
        raise SystemExit("no base/retest pairs in the evaluation")
    return {"tau_I": float(np.percentile(js, 100 * q)), "n_cases": len(js), "median_js": float(np.median(js))}
