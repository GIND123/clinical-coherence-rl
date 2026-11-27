# Amendment 4: readout and credit assignment (27 Nov 2026)

Made after the second decisive attempt stopped and **before any further training run**
(osf.io/4xmq7: "any change to the reward or training configuration after the first runs is made by
a dated amendment before further runs"). The registered objective, data, items, prompts, models,
hypotheses and evaluation readout are unchanged.

## What the stopped runs showed

Both attempts are in [training_runs.md](training_runs.md). Analysis of the stopped adapters found
two separate mechanisms, each sufficient to produce the flattening, and one weakness in the
endpoints.

### 1. The readout let the policy flatten for free

The prompt asks for integer percentages over the 49 codes that sum to 100. The parser, shared with
the Bedrock panel, renormalises whatever sum it reads.

| model (DDXPlus confirmatory items) | answers summing to 100 ± 2 | median raw sum | mean top probability after renormalising |
|---|---|---|---|
| untrained Qwen3-4B | 30% (93% of them one-hot) | 147 | 0.666 |
| RLVR, 36 steps | 16% | 199 | 0.423 |
| coherence arm, 21 steps | 0% | 178 | 0.235 |

- **How the flattening happened.** The trained models kept their top diagnosis (often at 50) and
  wrote filler values of 2–5 for all 48 other codes. After renormalising, a filler of 5 turns a
  top value of 50 into a probability of 0.17.
- **Why it cost nothing.** This is a change of a few digits per code, so it costs almost nothing
  in token-level KL. It makes every answer flatter and every pair of answers closer in JS.
- **RLVR flattened for the same reason.** Its binary reward ignores the stated probabilities, so
  nothing held the filler down.

### 2. The anti-collapse gates never reached the update

- **What the gates are.** The cross-patient information M and the sensitivity gate G_S are batch
  statistics that multiply every rollout's coherence term. They are the terms that make a
  constant or flat answer worthless (Proposition 4).
- **Why the update could not see them.** Per-rendering-centred, SD-normalised advantages credit
  rollout i only with the dependence of its *own* reward on its *own* answer. The update therefore
  follows the gradient of Σ R_j with the gates frozen at their current value. Flattening raises a
  rollout's own invariance and, for an overconfident model, its Brier score. Its cost, a lower M
  for the whole batch, is charged to no one.
- **What the log showed.** The coherence run's total batch reward fell from 1.22 to 0.75 while
  each rollout's own terms improved: the policy moved against its own objective.
- **Prior work.** This is the batch-level, cross-prompt case of the result of Gao et al.
  (arXiv:2611.02404) for group-dependent rewards within one prompt.

The **estimator audit** (`python -m ccrl.reward.audit`, `src/ccrl/reward/audit.py`) reproduces
the failure without a language model:

- **Setup.** A policy chooses, per rollout, a temperature, a presentation sensitivity and a noise
  level. It is trained on the registered reward of DDXPlus training patients, with an
  overconfident belief like the untrained 4B's (top-1 0.31, entropy 1.6).
- **Registered credit:** the temperature rose about sevenfold, entropy reached 3.87 nats
  (uniform = 3.89), M fell from 0.58 to 0.005, and the registered objective fell from 1.24 to
  0.52.
- **Team credit (below):**
  - M stayed at 0.50, above m0 = 0.459;
  - entropy stayed near 1.9;
  - presentation sensitivity fell eightfold and the divergence between renderings from 0.22 to
    0.06;
  - the objective rose to 1.48.

### 3. The registered endpoints could be met by hedging

- **The weakness.** Total incoherence (H2), top-1 (H3) and the ensemble gap (H7) are all
  satisfied by a model that flattens every answer while keeping its argmax.
- **The evidence.** RLVR's 36-step model shows it. Its total incoherence fell 34% (0.277 →
  0.182) through flattening alone, while incoherence normalised by the between-patient divergence
  did not move (0.363 → 0.365).

## Changes

| # | change | applies to | file |
|---|---|---|---|
| A4.1 | **Strict readout for the reward.** A rollout's answer is valid only if it parses *and* its percentages sum to 100 ± 2, as the prompt instructs. Invalid answers get reward 0 (as before for parse failures) and are excluded from every coherence statistic. | every GRPO arm | `readout: {strict_sum: true, sum_tol: 2}` in `config/train/base.yaml`; `readout_valid` in `src/ccrl/train/grpo.py` |
| A4.2 | **Team credit.** Each rollout's advantage is computed from its marginal contribution to the batch objective, D_i = J(y) − J(y without i), where J = Σ_j R_j is the registered reward summed over the batch and J(y without i) recomputes every term without rollout i. D_i replaces R_i in the same per-rendering centred, SD-normalised advantages. J(y without i) does not depend on rollout i, so the update is an unbiased estimate of ∇E[J] (Proposition 7). For arms without coupled terms (RLVR, RLCR) D_i = R_i exactly, so they are unchanged. | every GRPO arm | `credit: team`; `team_credit` in `src/ccrl/reward/coherence.py` |
| A4.3 | **Safeguard statistic kept as registered.** The collapse safeguard (20-step mean of M against the first 10 steps, OSF update of 27 Nov) is computed with the registered readout, over every parsed answer, renormalised. Strict readout therefore does not change what it monitors. Logged as `M_lenient`; the reward's M is logged as `M`. | every GRPO arm | `src/ccrl/train/grpo.py` |
| A4.4 | **Informativeness guard for H2** (an addition, not a change of the test). H2 counts as supported only if, on the same DDXPlus items, the CCR arm keeps ≥ 90% of RLVR's cross-patient information M of the base answers (a registered safeguard estimand) and a mean posterior entropy at most 0.5 nats above RLVR's. Reported as secondary: the sum-to-100 compliance rate, and normalised incoherence (total incoherence / mean JS to other cases' base answers) with the same 25% threshold. | analysis | `guard` in `src/ccrl/train/hypotheses.py` |

Applied to the stopped adapters, A4.4 rejects the coherence model: its M is 0.077 against
RLVR's 0.206.

## Unchanged

- **The reward:** every term and every constant (α, β, γ, τ_I = 0.788, δ_S, m0, c_fid).
- **The training setup:** data, items, prompts, models, seeds and the number of steps.
- **The evaluation readout.** Every confirmatory estimand is still computed with the panel
  parser, so trained and panel models stay comparable.
- **The hypotheses and their tests.**

## Tests

- **`tests/test_team_credit.py`:**
  - fast team credit equals a full recomputation (to 1e-10, both sensitivity modes, mixed
    candidate counts, invalid rollouts);
  - D_i = R_i when nothing couples;
  - team credit charges a hedging rollout for its effect on the gate;
  - the estimator audit's update direction;
  - the strict readout rule.
- **`python -m ccrl.reward.audit`:** the six audit checks.
- **The full suite:** 124 tests pass.

## OSF update text (to file on osf.io/4xmq7)

> **Update 2 (27 Nov 2026), before any further training run.** The second decisive attempt
> (reported, docs/training_runs.md) stopped under the collapse safeguard. Analysis found:
> (1) the parser's renormalisation let the policy flatten its answers by writing filler
> percentages, at no token cost; (2) the batch-level anti-collapse gates did not reach the policy
> update under group-relative advantages.
>
> Changes, all before further runs (docs/amendment_4.md):
> (a) for the training reward, an answer is valid only if its percentages sum to 100 ± 2, as the
> prompt already instructs;
> (b) advantages use each rollout's marginal contribution to the registered batch objective
> (team credit), an unbiased estimator of its gradient, which is identical to the old rule for
> RLVR/RLCR;
> (c) the safeguard statistic is computed with the registered readout;
> (d) H2 is reported as supported only if the CCR arm keeps ≥ 90% of RLVR's cross-patient
> information and its entropy is within 0.5 nats of RLVR's.
>
> The reward's terms and constants, data, prompts, models, hypotheses, tests and evaluation
> readout are unchanged. All runs, including the stopped ones, are reported.
