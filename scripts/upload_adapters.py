#!/usr/bin/env python3
"""Upload trained LoRA adapters (weights only) to a public Hugging Face model repo.

    HF_TOKEN=... python scripts/upload_adapters.py <copied-back runs dir> [--repo GOVINDFROM/coherence-reward-qwen3-4b-lora]

Runs on the analysis machine after `stage_a_local.sh` has copied runs/ back, so the token never
goes to the shared Lambda instance. Only these files are uploaded, per adapter:
    adapter_model.safetensors, adapter_config.json
for runs/train/<run>/final and runs/train/<run>/checkpoints/step_XXXX. No training data, rollouts,
logs, evaluations or prompts are uploaded. The repo README is (re)written from this script.
Re-running skips adapters already present with the same content.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = "GOVINDFROM/coherence-reward-qwen3-4b-lora"
ALLOWED = ("adapter_model.safetensors", "adapter_config.json")
README = """---
license: apache-2.0
base_model: Qwen/Qwen3-4B
library_name: peft
tags: [lora, grpo, clinical, consistency, research]
---

# Coherence as a reward: LoRA adapters for Qwen3-4B

Research adapters from the training study of *Coherence as a Reward*
(code, design and preregistration: https://github.com/GIND123/clinical-coherence-rl,
OSF osf.io/4xmq7). Each folder is one run: `<arm>_s<seed>/final` and milestone checkpoints
`<arm>_s<seed>/step_XXXX`. Arms are defined in `config/train/<arm>.yaml` of the code repository.

- Base model: Qwen/Qwen3-4B, thinking off; LoRA r = 32 on all linear layers.
- Training data: synthetic DDXPlus patients (public corpus); no real patient data.
- **This repository holds adapter weights only.** No training data, prompts, rollouts or
  evaluation outputs are stored here.
- Not a medical device. For research on training methods only; not for clinical use.

Load:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(base, "%s", subfolder="ccr_acc_fid_s0/final")
```
"""


def adapters(runs: Path):
    for run in sorted((runs / "train").glob("*_s*")):
        for d in [run / "final", *sorted((run / "checkpoints").glob("step_*"))]:
            if all((d / f).is_file() for f in ALLOWED):
                yield f"{run.name}/{'final' if d.name == 'final' else d.name}", d


def main(argv=None) -> int:
    from huggingface_hub import HfApi
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", help="a copied-back runs/ directory")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("set HF_TOKEN")
    api = HfApi()
    found = list(adapters(Path(a.runs)))
    print(f"{len(found)} adapters under {a.runs}")
    if a.dry_run:
        for dest, d in found:
            print(" ", dest, "<-", d)
        return 0
    api.create_repo(a.repo, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=(README % a.repo).encode(), path_in_repo="README.md", repo_id=a.repo,
                    commit_message="model card")
    for dest, d in found:
        api.upload_folder(folder_path=str(d), path_in_repo=dest, repo_id=a.repo, allow_patterns=list(ALLOWED),
                          commit_message=f"adapter {dest}")
        print("uploaded", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
