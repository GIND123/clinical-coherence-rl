# Track B — local training and evaluation

Runs on a GPU workstation after the panel is archived (from 19 Dec 2026).
Nothing here calls Bedrock.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements/train.txt && pip install -e . --no-deps
ccrl data download ddxplus           # raw corpus, md5-checked (~180 MB)
ccrl train build-pool                # 60k DDXPlus training patients (held-out pathologies excluded)
ccrl train reward-suite              # synthetic-policy suite; must print 10 PASS (seconds, CPU)
python -m pytest -q tests/test_train_smoke.py
```

The label-free MedCaseReasoning arm needs the training-split preprocessing
(`ccrl prep mcr --stage extract --roles train`, run on the panel machine
before 18 Dec) and `ccrl prep finalize`; the resulting
`artifacts/frozen/medcase/prepared.jsonl` must be copied/committed.

## One run

```bash
ccrl train run --config config/train/ccr_acc_fid.yaml --seed 0
ccrl train eval --model Qwen/Qwen3-4B --adapter runs/train/ccr_acc_fid_s0/final --name ccr_acc_fid_s0
ccrl train compare --ref runs/local_eval/rlvr_s0.parquet --new runs/local_eval/ccr_acc_fid_s0.parquet
```

Training resumes from the newest checkpoint if interrupted (re-run the same
command). Logs: `runs/train/<name>_s<seed>/train_log.jsonl` with reward
terms, M, the sensitivity gate, invalid fraction, entropy, KL; a run stops
early if M falls > 30 % below its first-5-step mean (collapse safeguard).

## Grid (plan §6)

| | configs | seeds |
|---|---|---|
| 4B main | `ccr_only`, `ccr_acc`, `ccr_acc_fid`, `rlvr`, `rlcr`, `sft_kb_bayes`, `sft_distill`, `bct`, `logit_readout` | 3 |
| 4B ablations | `abl_no_invariance`, `abl_no_sensitivity`, `abl_no_mgate`, `abl_rollout_sensitivity`, `abl_no_para`, `abl_no_cue`, `abl_g8`, `abl_kl05` | 1 |
| label-free real cases | `labelfree_mcr` | 2 |
| 8B | `qwen3_8b_ccr_acc_fid`, `qwen3_8b_rlvr` (+ best baseline) | 3 |

Inference-time baselines (no training) are evaluated with `ccrl train eval`
on the base model: CoT and "reason like a Bayesian" prompts are prompt
variants; canonicalisation sorts the findings before rendering;
neuro-symbolic Bayes (elicited likelihood ratios + code) is the ELR-Fusion
method of the previous study.

## Reward constants

`delta_S` and `m0` are half the reference posterior's batch-mean contrast
movement and cross-patient information on DDXPlus training patients
(printed by `ccrl train reward-suite`: 0.234 and 0.459 at τ = 0.6). `tau_I`
should be set from the pilot to the 95th percentile of the base model's
retest JS (default 0.10).

## Implementation notes

The loss utilities (`src/ccrl/train/common.py`) are adapted from the team's
DEFER-Dx trainer (Qwen3-8B, single GPU): the exact frozen-head log-prob with
chunked backward (memory independent of the 152k vocabulary) and
token-budget micro-batching. Rollouts use transformers `generate` (thinking
off, ≤ 512 new tokens — a posterior JSON is ~300 tokens). For speed, port
DEFER-Dx's colocated vLLM loop (`training/grpo_vllm.py`: sleep/wake,
frozen-weight parking, LoRA hot-swap, truncated importance sampling);
`Sample.rollout_logprobs` and the TIS weight in `pg_kl_loss` are already in
place for it.

## Transfer and retention

* Held-out pathologies: the 8 pathologies in `artifacts/frozen/ddxplus/heldout.json`
  appear in the panel items but never in training.
* Real cases: MedCaseReasoning panel items.
* Non-medical: BayesBench coin/recommender tasks (public code, arXiv:2606.30850).
* Capability retention: MMLU-Pro and GSM8K subsets with lm-eval-harness on the
  base model and every adapter (H3, Δ = 1.5 points).
