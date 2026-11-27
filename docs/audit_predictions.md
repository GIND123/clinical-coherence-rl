# Estimator audit: predictions for stage A (27 Nov 2026)

These predictions were committed **before any stage A run** (claim C7, [plan_final.md](plan_final.md);
[amendment_5.md](amendment_5.md) B.7). They come from `python -m ccrl.reward.audit_arms`, a CPU-only
simulation with no language model. The full output is in `audit_predictions.json`. After stage A,
each prediction is compared with what the GPU run did, and the comparison is reported whether it
agrees or not.

## The simulated policy

Each rollout draws four knobs around learnable means:

- temperature T;
- presentation sensitivity k;
- sampling noise s;
- filler f, added to every code's percentage.

The patients are DDXPlus training patients. Their belief is imperfect and overconfident, like the
untrained Qwen3-4B: top-1 about 0.33, and about 30% of answers sum to 100.

The readouts:

- **lenient** (registered): any sum is renormalised, so filler flattens the answer for free;
- **strict** (Amendment 4): the answer counts only if its percentages sum to 100 ± 2.

Each arm uses its own reward, credit and readout. It runs 150 steps with batches of 16 patients
(4 + 4 + 2 rollouts), and the collapse safeguard is applied to M as on the GPU.

## Predictions

Start → end are 15-step means. Here M is the cross-patient information of the answers the reward
counts. x/x' divergence is the JS between a patient's clean and transformed answers.

| arm | safeguard fires | M | entropy (nats) | x/x' divergence | sum-to-100 | top-1 |
|---|---|---|---|---|---|---|
| attempt 2 (registered credit, lenient), **already run** | yes, step 22 | 0.38 → 0.01 | 2.39 → 3.85 | 0.127 → 0.004 | 0.21 → 0.07 | 0.33 → 0.36 |
| cell A (team credit, lenient) | no | 0.50 → 0.52 | 1.89 → 1.89 | 0.129 → 0.038 | 0.23 → 0.00 | 0.33 → 0.36 |
| cell B (registered credit, strict) | yes, step 24 | 0.45 → 0.01 | 2.10 → 3.85 | 0.214 → 0.005 | 0.41 → 1.00 | 0.32 → 0.36 |
| CCR (team credit, strict) | no | 0.54 → 0.51 | 1.76 → 1.88 | 0.219 → 0.060 | 0.41 → 1.00 | 0.33 → 0.36 |
| RLVR (strict) | no | 0.66 → 0.68 | 1.25 → 1.19 | 0.222 → 0.061 | 0.41 → 1.00 | 0.32 → 0.36 |
| Brier-only (strict) | yes, step 23 | 0.43 → 0.01 | 2.16 → 3.86 | 0.227 → 0.005 | 0.41 → 1.00 | 0.32 → 0.36 |

## What the audit predicts, in words

1. **The completed GPU run is reproduced.** The registered credit with the lenient readout
   collapses at step 22 (the GPU run stopped at step 21), with M near 0 and the answer near
   uniform.
2. **The credit is the decisive mechanism.**
   - **Cell B** (registered credit, strict readout) still collapses. Without filler, it flattens
     by stated hedging: its temperature rises.
   - **Cell A** (team credit, lenient readout) does not collapse. It writes filler (compliance
     falls to 0), but M holds.
   - **Revision of C1.** If the GPU agrees, C1 is revised from "two mechanisms, each sufficient"
     to: the estimator is necessary and sufficient, and the readout is a channel the collapse
     uses when it is open.
3. **CCR trains without collapse.** Its M holds and its entropy rises by about 0.1 nats (it
   calibrates). Its x/x' divergence falls about 3.7-fold.
4. **RLVR reduces presentation divergence about as much as CCR** (0.061 vs 0.060). In this
   simulation, presentation sensitivity costs accuracy, so accuracy training removes it too.
   - **Where the audit predicts a difference: calibration and information.** RLVR's answers stay
     sharper (entropy 1.19 vs 1.88) and its M is higher (0.68 vs 0.51).
   - **What this means for C2.** The audit does not predict that CCR beats RLVR on the nuisance
     endpoints. A GPU difference there would come from presentation effects that do not cost
     accuracy, which the simulation leaves out.
5. **Brier-only collapses to hedging** (step 23). For an overconfident, often wrong belief, the
   Brier score improves by flattening. CCR contains the same Brier term but does not collapse.
   - **What this means for C3.** The predicted difference is that the M-gated coherence term,
     under team credit, holds the information that Brier alone gives up.
   - **Brier-only on the GPU is expected to stop early** (about 25 steps, about $2).
6. **Under the strict readout, every arm learns to sum to 100** (compliance 0.41 → 1.00).

## How the comparison will be scored

| prediction | GPU statistic |
|---|---|
| safeguard fires | the same safeguard in the training log |
| direction of M | M_lenient |
| direction of entropy | posterior entropy of the evaluated adapter vs the untrained model |
| direction of x/x' divergence | JS to the nuisance renderings (`js_nuis`) |
| direction of compliance | `sum100_frac` |

A direction counts as matched if it agrees with the audit's direction (down / flat / up, with the
thresholds in `audit_arms.predict`).
