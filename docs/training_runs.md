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
