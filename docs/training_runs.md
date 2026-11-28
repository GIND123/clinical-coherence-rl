# Training run log (Paper 2)

Every completed run is listed here, including runs stopped by a safeguard, as the OSF
registration requires (osf.io/4xmq7). The run directories are archived under
`E:\ccrl-data\lambda_runs\` on the analysis machine. They will be deposited with the Paper 2
artefacts.

## Environment (Lambda, 27 Nov 2026)

| | |
|---|---|
| software | vLLM 0.11.0, torch 2.8.0+cu128, transformers 4.57.6, peft 0.21.3 (`runs/env/pip-freeze-final.txt`) |
| hardware | 1× H100 80 GB PCIe (coherence-reward arm), 1× A100 40 GB SXM4 (RLVR arm, `GPU_MEM=0.45 VLLM_SLEEP=1`) |
| speed | 62 s per GRPO step on the H100 with vLLM rollouts (10 s of it generation), against 173 s with transformers `generate` |
| engine agreement | vLLM and training log-probs agree: mean importance weight `tis_w` = 1.000 |
| `tau_I` | 0.788, the 95th percentile of the untrained Qwen3-4B's pilot retest JS (30 cases: median 0.026; 5 flips, two of them confident at JS 0.97). Kept as registered |

## Untrained baselines (Qwen3-4B, thinking off, confirmatory items)

| | DDXPlus top-1 | DDXPlus retest flip | DDXPlus JS order / retest | MedCaseReasoning top-1 | MedCaseReasoning retest flip |
|---|---|---|---|---|---|
| temperature 0.7 (registered decoding) | 0.330 | 0.250 | 0.314 / 0.228 | 0.107 | 0.120 |
| temperature 0 | 0.320 | 0.040 | 0.223 / 0.030 | 0.107 | 0.010 |

- **At temperature 0 the order effect stays large** (0.22 against a retest JS of 0.03). Coherence
  training has a real systematic effect to remove, not only noise.
- **On real case reports the untrained 4B model is at chance** (10 candidates). Its answers are
  not degenerate: no positional bias, and a mean top probability of 0.52. This matters for H5
  (top-3 non-inferiority on real cases).

## Runs

| run | GPU | steps | outcome |
|---|---|---|---|
| `ccr_acc_fid_s0` (attempt 1) | H100 | 10 / 300 | stopped by the collapse safeguard at step 10 (M 0.289 against a 5-step baseline of 0.434) |
| `rlvr_s0` (attempt 1) | A100 | 13 / 300 | stopped by the collapse safeguard at step 13 (M 0.316 against 0.452) |

**Neither run collapsed:**

- invalid answers 0%;
- entropy steady at about 0.015;
- KL to the reference at most 0.003, so the policies had barely moved;
- rewards steady.

**The safeguard fired on batch noise.** M is computed on 16 patients per step. Its step-to-step
coefficient of variation was 0.26 (CCR) and 0.16 (RLVR). A *single* step 30% below a *5-step*
mean is therefore a routine event. The RLVR arm, which does not optimise M at all, tripped it
after 13 steps.

## Change to the collapse safeguard (dated 27 Nov 2026, before any further run)

| | definition |
|---|---|
| **was** | stop when M at the current step is more than 30% below the mean of the first 5 steps |
| **now** | stop when the **mean M over the last 20 steps** is more than 30% below the **mean of the first 10 steps** (checked from step 21; `collapse_check` in `src/ccrl/train/grpo.py`) |

- **The threshold is unchanged.** Only the noise in the comparison is reduced.
- **It still catches real collapse.** At the measured noise, a stationary run never triggers it
  (tested on 300 simulated steps with a CV of 0.25). A collapse to M ≈ 0 triggers within 20 steps
  (tests in `tests/test_vllm_backend.py`).
- **The M history is read back from the run log on resume,** so the baseline survives a restart.
- **Both arms of the decisive pair are rerun from scratch** under the new rule.

## Other changes before the rerun (27 Nov 2026; [audit_2026-11-27.md](audit_2026-11-27.md))

These were all made before any further training run:

- Each patient group uses its own candidate count. 29 real training cases have 9 candidates, so
  without this the label-free arm crashes.
- The x- invariance term uses a leave-one-out mean.
- Advantages use a per-rendering baseline (`advantage_baseline: rendering`).
- `tau_I = 0.788` is set in the base config for every arm. Qwen3-8B runs measure their own
  value from the untrained 8B's pilot evaluation, by the same rule.
- The registered tests H2–H7 are implemented in `ccrl train hypotheses`.
- Every trained model is also evaluated at temperature 0, and on the post-freeze real case
  reports once they exist.

## Rerun after the audit (27 Nov 2026)

| run | GPU | steps | outcome |
|---|---|---|---|
| `ccr_acc_fid_s0` (attempt 2) | H100 | 21 / 300 | stopped by the collapse safeguard: 20-step mean M 0.266 against a 10-step baseline of 0.418 |
| `rlvr_s0` (attempt 2) | A100 | 36 / 300 | stopped by the collapse safeguard: 20-step mean M 0.315 against 0.452 (−30.4%) |

**This time the decline is real, not noise.** M by 5-step block was 0.45 → 0.39 → 0.18 → 0.14 →
0.12 (CCR) and 0.44 → … → 0.31 (RLVR).

Evaluation of the stopped adapters (DDXPlus, temperature 0.7):

| | top-1 | mean top probability | posterior entropy (nats; uniform = 3.89) | distinct top-1 answers | JS order |
|---|---|---|---|---|---|
| untrained | 0.330 | 0.666 | 1.55 | 39 | 0.314 |
| RLVR, 36 steps | 0.350 | 0.423 | 2.62 | 39 | 0.187 |
| CCR+acc+fid, 21 steps | 0.330 | 0.235 | 3.37 | 37 | 0.065 |

**Diagnosis.** The coherence reward reduced incoherence mainly by **hedging**: it flattened every
posterior towards uniform.

- **Accuracy and the argmax** did not change. The cross-patient information M fell to 0.12,
  i.e. the answers stopped being patient-specific.
- **Why the gate failed.** The M and sensitivity gates are batch-level multipliers. Under
  group-relative advantages they give no individual rollout a reason to avoid flatness.
- **What rewarded flatness.** The invariance term (flatter answers diverge less) and the Brier
  accuracy term (an overconfident model gains by hedging) both do.
- **The counterweight was too weak.** Fidelity fell from 0.28 to 0.02, but at γ = 0.5 it was
  outweighed.

Proposition 4 (gates exclude collapse) holds for the optimum of the expected reward. It does not
constrain the per-rollout gradient that GRPO follows. RLVR also drifted towards flatter
posteriors, but much less.

### Revised diagnosis (27 Nov 2026, same day): two mechanisms, not one

Reading the stopped adapters' raw answers changed the picture
([amendment_4.md](amendment_4.md)).

- **The flattening was filler, not stated hedging.** The prompt asks for percentages that sum to
  100. The trained models kept their top diagnosis and wrote 2–5 for all 48 other codes, and the
  parser renormalises any sum.
  - Answers summing to 100 ± 2: untrained 30%, RLVR-36 16%, CCR-21 0%.
  - Median raw sum: 147, 199 and 178.
  - This also explains why RLVR, whose binary reward ignores the stated probabilities, flattened.
- **The gates never reached the update.** M and G_S are batch statistics. Per-rendering advantages
  hold them fixed (theory.md, Proposition 6). The coherence run's batch reward fell from 1.22 to
  0.75 while each rollout's own terms improved.
- **The estimator audit reproduces this with no language model** (theory.md §6). Under the
  registered credit, M falls from 0.58 to 0.005 and the answer becomes uniform. Under team credit,
  M holds at 0.50 and the invariance gain comes from lower presentation sensitivity.
- **The raw H2 endpoint can be gamed.** RLVR-36's total incoherence fell 34% through flattening
  alone, while normalised incoherence did not move (0.363 → 0.365).

Changes before the next run: strict readout for the reward, team credit, and the safeguard on the
registered readout's M (Amendment 4).

## Stage A (started 27 Nov 2026, 23:00 UTC; [plan_final.md](plan_final.md), [amendment_5.md](amendment_5.md))

**Hardware.** No H100 PCIe was available, so stage A started on one H100 SXM5 ($4.89/h) and one
A100 40 GB ($1.99/h).

**The H100 was lost before any result.**

- **What it did.** It finished the untrained-model evaluations: confirmatory items at
  temperature 0.7 and 0, pilot, and MMLU-Pro/GSM8K. These were copied back and reused.
- **What failed.** Its first coherence run ran out of memory at step 3: the shared-prompt pass at
  4,096 tokens, with vLLM holding 26 GB.
- **How it was lost.** Minutes later the instance became unreachable, and Lambda then reported it
  as terminated. Our scripts did not terminate it: the orchestrator log has no terminate call. The
  cause is unknown.
- **What it showed.** Before failing, the shared pass ran at about 22 s per step, against 78 s
  before.

**Replacement.** No single H100 was available afterwards, so the coherence queue moved to a second
A100 40 GB.

- **The shared pass does not fit a 40 GB card** without checkpointing: its smallest unit is one
  prompt with its four rollouts. The GPU check found no size that fits.
- **Both A100s therefore train with the unshared pass and gradient checkpointing,** which is the
  pass every earlier run used. The two passes are the same computation: equal in fp32
  (`tests/test_shared_prompt.py`), and on the H100 within bf16 noise.
- **Result:** no stage A run uses the shared pass.

**Orchestration fixes made during the launch,** before any training result:

- the process check matched its own SSH command line;
- the GPU check retried inside its exception handler, so the failed pass was never freed;
- a machine unreachable for 30 minutes is now terminated, so it stops billing.
