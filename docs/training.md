# Training and local evaluation

Runs on a GPU workstation and **can start now**, in parallel with the
Bedrock panel. Nothing here calls Bedrock. Only two inputs come from the
panel machine:

* the `distill` job's Kimi K3 answers, for the `sft_distill` baseline;
* the MedCaseReasoning training extraction, for the `labelfree_mcr` arm.

Everything else needs only the public DDXPlus corpus.

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

## Endpoints after amendment 1

**Registered on OSF on 26 Nov 2026, before any training run: https://osf.io/4xmq7.** Any change to the reward,
configs or endpoints from here on needs a dated OSF update *before* the affected runs.

The main panel showed that run-to-run noise dominates (docs/results_panel_main.md).
The training endpoints were amended before any training run
(docs/osf_amendment_1.md):

- **primary:** *total* incoherence, the raw JS between meaning-preserving
  answers (`js_order`, `js_dup`, `js_para`, `js_cue`). The retest JS and the
  retest flip rate are reported with it;
- **H7:** the ensemble gap, single-answer top-1 minus 8-answer-mean top-1
  (`ens_gap_top1`);
- **new baselines:** base model at temperature 0; the base model's 8-answer
  ensemble; SFT on the base model's own 8-answer mean.

`ccrl train compare` reports all of these per dataset.

**Inputs: all in the repository (25 Nov 2026).**

- the items, because evaluation runs on exactly the panel's items;
- the phrase bank (`banks/ddx_phrasebank.json`), which the training
  sampler's paraphrase transform uses;
- `medcase/prepared_train.jsonl`: 2,000 MedCaseReasoning training cases for
  the label-free arm (plus 134 reserve and 16 rejected; SHA-256 `1883e4f7…`).
  It is separate from the panel's frozen `prepared.jsonl`, which is unchanged;
- the panel answers for H6 (`posteriors.parquet`), from the PI's verified
  archive (`ccrl-archive-20261125-144504.tar.gz`; not committed, ~775 MB).

Panel reference values the trained models must beat are in
docs/results_panel_main.md and docs/results_secondary.md. Temperature 0 is a
measured baseline: it halves the flip rate for the noisiest panel models.

## Running on Lambda (GPU)

Rollouts and evaluation use **vLLM** (`--backend vllm`; `src/ccrl/train/vllm_backend.py`):

- one engine holds the frozen base model;
- the current LoRA policy is saved after every optimiser step and loaded as a fresh adapter;
- prompts go in as the training tokenizer's token ids;
- the engine's per-token log-probs feed the truncated importance-sampling correction
  (`tis_cap: 2.0`). The mean weight is logged as `tis_w` and should stay close to 1.

Training samples at temperature 1.0 and top-p 1.0, so the sampling distribution is the policy
being optimised. Evaluation keeps the registered decoding (temperature 0.7, top-p 0.95, fixed
seed per item), plus temperature 0 for the baseline. This engine choice was made before any
training run. The method, reward and endpoints are unchanged.

The account is shared, so everything we create is named `govind_llm`, and
`scripts/lambda_api.py` refuses to touch any other instance.

```bash
# on this machine (key in the environment, never in a file)
python scripts/lambda_api.py types                      # what has capacity now
python scripts/lambda_api.py add-key ~/.ssh/govind_llm.pub
python scripts/lambda_api.py launch gpu_1x_h100_pcie us-west-3
python scripts/lambda_api.py list                       # IP once it is "active"

# on the instance
curl -fsSL https://raw.githubusercontent.com/GIND123/clinical-coherence-rl/main/scripts/lambda_setup.sh | bash
bash scripts/lambda_smoke.sh 2>&1 | tee runs/env/smoke.log          # ~20 min: parse rate, sec/step, tis_w
nohup bash scripts/lambda_decisive.sh > runs/env/decisive.log 2>&1 &

# afterwards, here: copy runs/ back (rsync), then
python scripts/lambda_api.py terminate --yes            # billing stops only when terminated
```

`tau_I` is measured, not chosen. It is the 95th percentile of the untrained model's retest JS on
the pilot items (`ccrl train tau-i`). It is passed to every run with `--tau-i` and recorded in
the run's `config.yaml`.

**Run log, safeguard change and untrained baselines:** [training_runs.md](training_runs.md).

## Order of runs (decisive comparison first)

1. `ccrl train reward-suite`: 10 PASS (CPU, seconds).
2. Evaluate the untrained base model on the pilot items. Its retest JS sets
   `tau_I` (95th percentile):
   `ccrl train eval --model Qwen/Qwen3-4B --name base4b --set pilot`.
3. **The decisive pair**, 4B, seed 0: `rlvr` and `ccr_acc_fid`, then
   `ccrl train compare`. If CCR does not beat RLVR on `js_order` and
   `js_para` here, stop and diagnose before spending GPU-days. Check the
   per-term reward logs, the M gate, collapse stops and the ablations.
4. The rest of the 4B grid (3 seeds), then the ablations (1 seed).
5. 8B: `qwen3_8b_ccr_acc_fid` and `qwen3_8b_rlvr`, 3 seeds each (needed for
   H6).
6. Label-free MedCaseReasoning arm (needs the training extraction).

## Grid

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

## Scale comparison (H6): trained model vs the Bedrock panel

The panel export and local evaluations share item IDs, parser and estimands.
Each model is normalised by its own retest floor, so different decoding
noise does not favour either side.

```bash
ccrl train eval --model Qwen/Qwen3-8B --adapter runs/train/qwen3_8b_ccr_acc_fid_s0/final --name q8_ccr_s0
ccrl train compare --ref runs/export/posteriors.parquet --ref-model qwen3-32b --new runs/local_eval/q8_ccr_s0.parquet
ccrl train compare --ref runs/export/posteriors.parquet --ref-model qwen3-next-80b --new runs/local_eval/q8_ccr_s0.parquet
```

`runs/export/posteriors.parquet` comes from the panel archive (`ccrl export`
on the panel machine). `--ref-model` picks one panel model, and its
`panel_main` rows are used.

## Reward constants

`delta_S` and `m0` are half the reference posterior's batch-mean contrast
movement and cross-patient information on DDXPlus training patients
(printed by `ccrl train reward-suite`: 0.234 and 0.459 at τ = 0.6). `tau_I`
is the 95th percentile of the untrained base model's retest JS on the pilot
items (step 2 above; default 0.10).

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
