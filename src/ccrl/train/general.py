"""General-capability check (registered H3, second part; plan_final.md §3.3, 27 Nov 2026).

    ccrl train general --model Qwen/Qwen3-4B [--adapter runs/train/<arm>_s0/final] --name <arm>_s0

Fixed subsets of two public benchmarks, chosen by seed before any trained model exists:
  MMLU-Pro (TIGER-Lab/MMLU-Pro, test split)  2,000 questions, letter answer, max 16 new tokens
  GSM8K    (openai/gsm8k, main, test split)   all 1,319 questions, final answer after '####'
Thinking off, temperature 0, the same chat template as training. Output:
runs/local_eval/general_<name>.parquet (one row per question: task, qid, correct).
Test: each trained arm non-inferior to the untrained model, paired by question, margin 1.5
points (one-sided 95% bootstrap bound), per benchmark (`noninferior`).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

N_MMLU = 2000
SEED = 20261127
LETTERS = "ABCDEFGHIJ"
MARGIN = 0.015


# ------------------------------------------------------------------ data
# dataset revisions pinned on 27 Nov 2026, before any trained model was evaluated
REVISIONS = {"TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
             "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866"}


def _hub_parquet(repo: str, pattern: str) -> tuple[pd.DataFrame, str]:
    from huggingface_hub import HfApi, hf_hub_download
    rev = REVISIONS[repo]
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
             if re.search(pattern, f) and f.endswith(".parquet")]
    if not files:
        raise SystemExit(f"{repo}: no file matching {pattern}")
    frames = [pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", revision=rev)) for f in sorted(files)]
    return pd.concat(frames, ignore_index=True), rev


def select_mmlu(df: pd.DataFrame, n: int = N_MMLU, seed: int = SEED) -> pd.DataFrame:
    """A seeded sample, stratified by category in proportion to its size."""
    rng = np.random.default_rng(seed)
    df = df.sort_values("question_id").reset_index(drop=True)
    parts = []
    for _cat, g in df.groupby("category", sort=True):
        k = max(1, round(n * len(g) / len(df)))
        parts.append(g.iloc[np.sort(rng.choice(len(g), size=min(k, len(g)), replace=False))])
    return pd.concat(parts).sort_values("question_id").head(n).reset_index(drop=True)


def items() -> tuple[list[dict], dict]:
    mm, sha_m = _hub_parquet("TIGER-Lab/MMLU-Pro", r"(^|/)test")
    gs, sha_g = _hub_parquet("openai/gsm8k", r"^main/test")
    out = []
    for r in select_mmlu(mm).itertuples(index=False):
        opts = list(r.options)
        out.append({"task": "mmlu_pro", "qid": f"mmlu-{r.question_id}", "prompt": mmlu_prompt(r.question, opts),
                    "gold": LETTERS[int(r.answer_index)], "n_opts": len(opts), "max_new": 16})
    for i, r in enumerate(gs.itertuples(index=False)):
        out.append({"task": "gsm8k", "qid": f"gsm8k-{i:04d}", "prompt": gsm_prompt(r.question),
                    "gold": gsm_gold(r.answer), "max_new": 512})
    return out, {"mmlu_pro_sha": sha_m, "gsm8k_sha": sha_g}


# ------------------------------------------------------------------ prompts and parsing
def mmlu_prompt(q: str, opts: list[str]) -> str:
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
    return (f"Answer the following multiple-choice question. Reply with the letter of the correct option only.\n\n"
            f"Question: {q}\n\nOptions:\n{lines}\n\nAnswer:")


def gsm_prompt(q: str) -> str:
    return (f"Solve the following problem. Show brief working, then give the final answer as a number on the "
            f"last line in the form '#### <number>'.\n\n{q}")


def gsm_gold(ans: str) -> str:
    return _num(ans.split("####")[-1])


def _num(s: str) -> str:
    m = re.findall(r"-?\d[\d,]*\.?\d*", s or "")
    if not m:
        return ""
    v = m[-1].replace(",", "").rstrip(".")
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


def parse_mmlu(text: str, n_opts: int = 10) -> str:
    t = (text or "").strip()
    m = re.search(rf"\b([A-{LETTERS[n_opts - 1]}])\b", t)
    return m.group(1) if m else ""


def parse_gsm(text: str) -> str:
    t = text or ""
    return _num(t.split("####")[-1]) if "####" in t else _num(t)


# ------------------------------------------------------------------ run
def run(model: str, name: str, adapter: str | None = None, out_dir: str = "runs/local_eval",
        gpu_mem: float = 0.85, limit: int | None = None) -> Path:
    import json

    from transformers import AutoTokenizer

    from ccrl.train.common import encode_chat
    from ccrl.train.vllm_backend import VLLMEngine, adapter_rank

    its, meta = items()
    if limit:
        its = [x for t in ("mmlu_pro", "gsm8k") for x in [y for y in its if y["task"] == t][:limit]]
    tok = AutoTokenizer.from_pretrained(model)
    kw = {"enable_thinking": False}
    eng = VLLMEngine(model, max_new_tokens=512, temperature=0.0, top_p=1.0, gpu_memory_utilization=gpu_mem,
                     max_model_len=4096, lora_rank=adapter_rank(adapter), seed=0)
    rows = []
    for task, parse in (("mmlu_pro", None), ("gsm8k", None)):
        sub = [x for x in its if x["task"] == task]
        prompts = [encode_chat(tok, [{"role": "user", "content": x["prompt"]}], kw) for x in sub]
        outs = eng.generate(prompts, lora_path=adapter, temperature=0.0, top_p=1.0,
                            max_new_tokens=sub[0]["max_new"] if sub else 16, logprobs=False)
        for x, (_ids, text, _lp) in zip(sub, outs):
            pred = parse_mmlu(text, x["n_opts"]) if task == "mmlu_pro" else parse_gsm(text)
            rows.append({"task": task, "qid": x["qid"], "gold": x["gold"], "pred": pred,
                         "correct": float(pred == x["gold"] and pred != ""), "text": text})
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["model"], df["adapter"] = model, adapter or ""
    out = d / f"general_{name}.parquet"
    df.to_parquet(out)
    summ = {"name": name, **meta, **{t: float(g["correct"].mean()) for t, g in df.groupby("task")},
            "n": {t: int(len(g)) for t, g in df.groupby("task")}}
    (d / f"general_{name}.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(summ)
    return out


# ------------------------------------------------------------------ test
def noninferior(new: pd.DataFrame, base: pd.DataFrame, label: str, margin: float = MARGIN,
                seed: int = 0, n_boot: int = 2000) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for task in ("mmlu_pro", "gsm8k"):
        m = new[new["task"] == task][["qid", "correct"]].merge(base[base["task"] == task][["qid", "correct"]],
                                                                on="qid", suffixes=("_n", "_b"))
        if m.empty:
            continue
        d = (m["correct_n"] - m["correct_b"]).to_numpy(float)
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
        lo = float(np.percentile(bs, 5))
        rows.append({"hypothesis": "H3 general", "arm": label, "metric": task, "n": len(d),
                     "new": float(m["correct_n"].mean()), "ref": float(m["correct_b"].mean()),
                     "diff": float(d.mean()), "lo_one_sided_95": lo, "margin": -margin, "supported": lo > -margin})
    return rows
