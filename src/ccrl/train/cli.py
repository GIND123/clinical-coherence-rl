"""`ccrl train ...` — Track B entry points (GPU workstation)."""
from __future__ import annotations

import argparse
from pathlib import Path


def add_parser(sub) -> None:
    p = sub.add_parser("train", help="Track B: local training and evaluation")
    t = p.add_subparsers(dest="taction", required=True)
    s = t.add_parser("build-pool", help="DDXPlus training pool (needs the raw corpus)")
    s.add_argument("--n", type=int, default=60000)
    s = t.add_parser("reward-suite", help="synthetic-policy reward tests (CPU, seconds)")
    s.add_argument("--sensitivity", default="batch", choices=["batch", "rollout"])
    s = t.add_parser("run", help="train one config (grpo / sft / logit_readout per `method`)")
    s.add_argument("--config", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--backend", choices=["hf", "vllm"], default=None, help="rollout engine (default: config)")
    s.add_argument("--tau-i", type=float, default=None, help="reward tau_I measured by `ccrl train tau-i`")
    s = t.add_parser("eval", help="evaluate a local model on the frozen panel items")
    s.add_argument("--model", required=True)
    s.add_argument("--adapter", default=None)
    s.add_argument("--name", required=True)
    s.add_argument("--set", default="panel")
    s.add_argument("--limit-cases", type=int, default=None)
    s.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    s.add_argument("--temperature", type=float, default=0.7, help="0.7 = registered decoding; 0 = baseline")
    s = t.add_parser("tau-i", help="reward constant tau_I from the base model's pilot evaluation")
    s.add_argument("--eval", required=True, help="runs/local_eval/<name>.parquet of the untrained model, --set pilot")
    s = t.add_parser("compare", help="paired comparison of two evaluated models")
    s.add_argument("--ref", required=True)
    s.add_argument("--new", required=True)
    s.add_argument("--margin", type=float, default=0.02)
    s.add_argument("--ref-model", default=None, help="model_key inside --ref (needed for the panel export)")
    s.add_argument("--new-model", default=None, help="model_key inside --new")
    p.set_defaults(fn=run)


def run(a: argparse.Namespace) -> int:
    if a.taction == "build-pool":
        from ccrl.train.data import build_pool
        build_pool(a.n)
        return 0
    if a.taction == "reward-suite":
        from ccrl.reward.synthetic import run_suite
        ok, _ = run_suite(sensitivity=a.sensitivity)
        return 0 if ok else 1
    if a.taction == "run":
        from ccrl.train.grpo import load_config
        over = {"seed": a.seed}
        if a.steps:
            over["steps"] = a.steps
        if a.model:
            over["model"] = a.model
        if a.backend:
            over["generation"] = {"backend": a.backend}
        if a.tau_i is not None:
            over["reward"] = {"tau_I": float(a.tau_i)}
        cfg = load_config(a.config, over)
        cfg["output_dir"] = cfg["output_dir"].format(name=cfg["name"], seed=a.seed)
        m = cfg.get("method", "grpo")
        if m == "grpo":
            from ccrl.train.grpo import train
            print(train(cfg))
        elif m == "logit_readout":
            from ccrl.train.logit_readout import train_logit_readout
            print(train_logit_readout(cfg))
        elif m == "sft":
            from ccrl.train import sft
            from ccrl.train.data import PairedSampler
            src = cfg["sft_source"]
            if src == "distill":
                ex = sft.distill_examples(cfg["export_dir"])
            else:
                smp = PairedSampler(cfg.get("dataset", "ddxplus"), a.seed, cfg.get("transforms"), "triage", False,
                                    cfg.get("pool_path"), cfg.get("phrasebank_path"))
                n = int(cfg.get("sft_examples", 8000)) // 2
                ex = sft.kb_bayes_examples(smp, n) if src == "kb_bayes" else sft.bct_examples(cfg, smp, n)
            print(sft.train_sft(cfg, ex))
        else:
            raise SystemExit(f"unknown method {m}")
        return 0
    if a.taction == "eval":
        from ccrl.train.evaluate import evaluate_local
        print(evaluate_local(a.model, a.name, a.adapter, a.set, limit_cases=a.limit_cases, backend=a.backend,
                             temperature=a.temperature, top_p=0.95 if a.temperature > 0 else 1.0))
        return 0
    if a.taction == "tau-i":
        import json as _json

        from ccrl.train.evaluate import tau_i_from_eval
        print(_json.dumps(tau_i_from_eval(a.eval)))
        return 0
    if a.taction == "compare":
        from ccrl.train.evaluate import compare
        df = compare(a.ref, a.new, a.margin, ref_model=a.ref_model, new_model=a.new_model)
        print(df.to_string(index=False))
        out = Path(a.new).with_suffix(".compare.csv")
        df.to_csv(out, index=False)
        return 0
    return 1
