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
    s.add_argument("--gpu-mem", type=float, default=None, help="vLLM share of GPU memory (A100-40GB: 0.45)")
    s.add_argument("--vllm-sleep", action="store_true", help="free vLLM memory during the backward pass")
    s.add_argument("--no-shared", action="store_true",
                   help="unshared training pass with gradient checkpointing (fallback if check_shared fails)")
    s = t.add_parser("eval", help="evaluate a local model on the frozen panel items")
    s.add_argument("--model", required=True)
    s.add_argument("--adapter", default=None)
    s.add_argument("--name", required=True)
    s.add_argument("--set", default="panel")
    s.add_argument("--limit-cases", type=int, default=None)
    s.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    s.add_argument("--temperature", type=float, default=0.7, help="0.7 = registered decoding; 0 = baseline")
    s = t.add_parser("general", help="MMLU-Pro / GSM8K subsets for the H3 general-capability test (vLLM)")
    s.add_argument("--model", required=True)
    s.add_argument("--adapter", default=None)
    s.add_argument("--name", required=True, help="output runs/local_eval/general_<name>.parquet")
    s.add_argument("--gpu-mem", type=float, default=0.85)
    s.add_argument("--limit", type=int, default=None, help="first N questions per benchmark (smoke test)")
    s = t.add_parser("gonogo", help="go/no-go check at step 50 of the first coherence run (plan_final.md)")
    s.add_argument("--run", required=True)
    s.add_argument("--pilot", required=True, help="pilot eval of the step-50 adapter")
    s.add_argument("--base", required=True, help="pilot eval of the untrained model")
    s.add_argument("--at-step", type=int, default=50)
    s = t.add_parser("tau-i", help="reward constant tau_I from the base model's pilot evaluation")
    s.add_argument("--eval", required=True, help="runs/local_eval/<name>.parquet of the untrained model, --set pilot")
    s = t.add_parser("hypotheses", help="the registered confirmatory tests H2-H7 over all evaluated arms")
    s.add_argument("--dir", default="runs/local_eval")
    s.add_argument("--panel", default=None, help="panel export posteriors.parquet (for H6)")
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
        gen_over = {}
        if a.backend:
            gen_over["backend"] = a.backend
        if a.gpu_mem is not None:
            gen_over["gpu_memory_utilization"] = float(a.gpu_mem)
        if a.vllm_sleep:
            gen_over["sleep"] = True
        if gen_over:
            over["generation"] = gen_over
        if a.tau_i is not None:
            over["reward"] = {"tau_I": float(a.tau_i)}
        if a.no_shared:
            over.update(shared_prompt=False, gradient_checkpointing=True)
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
                                    cfg.get("pool_path"), cfg.get("phrasebank_path"), cfg.get("cue_bank", "panel"))
                n = int(cfg.get("sft_examples", 8000)) // 2
                if src == "kb_bayes":
                    ex = sft.kb_bayes_examples(smp, n)
                elif src == "bct":
                    ex = sft.bct_examples(cfg, smp, n)
                elif src == "selfcons":
                    ex = sft.selfcons_examples(cfg, smp, int(cfg.get("sft_examples", 8000)) // 4)
                else:
                    raise SystemExit(f"unknown sft_source {src!r}")
            print(sft.train_sft(cfg, ex))
        else:
            raise SystemExit(f"unknown method {m}")
        return 0
    if a.taction == "gonogo":
        import json as _json

        from ccrl.train.gonogo import check
        res = check(a.run, a.pilot, a.base, a.at_step)
        print(_json.dumps(res, indent=1))
        return 0 if res["go"] else 3
    if a.taction == "general":
        from ccrl.train.general import run as run_general
        print(run_general(a.model, a.name, a.adapter, gpu_mem=a.gpu_mem, limit=a.limit))
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
    if a.taction == "hypotheses":
        from ccrl.train.hypotheses import run as run_h
        df = run_h(a.dir, a.panel)
        print(df.to_string(index=False) if len(df) else "no evaluated arms found")
        if len(df):
            df.to_csv(Path(a.dir) / "hypotheses.csv", index=False)
        return 0
    if a.taction == "compare":
        from ccrl.train.evaluate import compare
        df = compare(a.ref, a.new, a.margin, ref_model=a.ref_model, new_model=a.new_model)
        print(df.to_string(index=False))
        out = Path(a.new).with_suffix(".compare.csv")
        df.to_csv(out, index=False)
        return 0
    return 1
