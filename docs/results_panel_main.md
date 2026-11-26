# Results: pilot and main panel (run 7–18 Nov 2026)

**Source:**

- the PI's `runs/` archive: ledger, raw mirrors, export, reports;
- the export files, all matching their SHA-256 manifest;
- ledger `PRAGMA integrity_check`: ok.

**What ran:** 36,000 main-panel requests (600 cases × 10 calls × 6 models,
all `done`) plus 1,980 pilot requests. Analysis by `ccrl analyze` (estimands)
and `ccrl.analysis.stability` (run-to-run noise). Intervals are
case-clustered bootstrap 95% CIs.

JS is Jensen–Shannon divergence in bits:

- 0 means identical answers;
- 1 means completely disjoint answers;
- the "ceiling" is the average JS between *different* patients (0.81–0.89 on
  DDXPlus).

## 1. Headline

At the decoding settings a deployment would use (provider defaults), **the
dominant failure is run-to-run instability, not sensitivity to how the case
is presented.**

- **Asked the identical question twice,** the model names a different top
  diagnosis in 9–41% of synthetic cases and 19–43% of real case reports.
- **Beyond that noise,** reordering, duplicating or rewording the findings,
  or adding an irrelevant sentence, adds little. Normalised effects are
  ≤ 0.09 of the distance to a different patient, and most CIs include 0.
- **The noise costs accuracy.** For DeepSeek V3.2, averaging the eight
  equivalent answers raises DDXPlus top-1 by 12.8 points, and up to 24% of a
  model's Brier error is pure answer-to-answer variance.
- **Disagreement is a better error alarm than stated confidence for the noisy
  models.** For example, AUROC 0.87 vs 0.67 for DeepSeek V3.2. For the
  stable Kimi K3 the opposite holds.

## 2. Accuracy, stability and invariance

### DDXPlus (300 cases, 49 candidates)

| model | top-1 [95% CI] | top-3 | retest flip [CI] | retest JS | NOE [CI] | ensemble gain, top-1 [CI] | variance share of Brier | AUROC: confidence / disagreement | ECE |
|---|---|---|---|---|---|---|---|---|---|
| Kimi K3 | **0.84** [0.79, 0.88] | 0.96 | **0.10** [0.06, 0.13] | 0.05 | 0.003 [−0.001, 0.008] | −0.005 [−0.021, 0.009] | 0.04 | **0.86** / 0.75 | 0.20 |
| gpt-oss-120b | 0.74 [0.69, 0.79] | 0.89 | 0.13 [0.09, 0.17] | 0.13 | 0.019 [0.004, 0.034] | +0.015 [−0.003, 0.035] | 0.10 | 0.77 / 0.78 | 0.08 |
| GLM-5 | 0.69 [0.64, 0.74] | 0.83 | 0.17 [0.13, 0.22] | 0.13 | 0.005 [−0.010, 0.020] | +0.027 [0.008, 0.049] | 0.08 | 0.82 / 0.83 | 0.11 |
| Qwen3-Next-80B | 0.67 [0.61, 0.72] | 0.81 | **0.09** [0.06, 0.12] | 0.09 | **0.056** [0.031, 0.082] | +0.017 [0.002, 0.033] | 0.14 | 0.72 / 0.71 | 0.14 |
| DeepSeek V3.2 | 0.65 [0.59, 0.70] | 0.85 | **0.41** [0.36, 0.47] | 0.30 | 0.002 [−0.043, 0.044] | **+0.128** [0.104, 0.155] | **0.24** | 0.67 / **0.87** | 0.13 |
| Qwen3-32B | 0.51 [0.45, 0.57] | 0.66 | 0.34 [0.29, 0.40] | 0.35 | 0.048 [−0.009, 0.102] | +0.049 [0.025, 0.071] | **0.24** | 0.59 / **0.80** | 0.23 |

### MedCaseReasoning (300 real case reports, 10 candidates)

| model | top-1 [95% CI] | top-3 | retest flip [CI] | retest JS | NOE [CI] | ensemble gain, top-1 [CI] | variance share | AUROC: confidence / disagreement | ECE |
|---|---|---|---|---|---|---|---|---|---|
| Kimi K3 | **0.61** [0.55, 0.66] | 0.80 | 0.19 [0.15, 0.23] | 0.08 | 0.019 [−0.002, 0.040] | +0.030 [0.012, 0.050] | 0.08 | 0.72 / 0.76 | 0.07 |
| gpt-oss-120b | 0.41 [0.35, 0.46] | 0.58 | 0.29 [0.23, 0.34] | 0.13 | 0.012 [−0.016, 0.039] | +0.013 [−0.007, 0.032] | 0.07 | 0.65 / 0.72 | 0.21 |
| GLM-5 | 0.40 [0.35, 0.46] | 0.56 | 0.32 [0.27, 0.37] | 0.13 | 0.031 [−0.019, 0.077] | +0.025 [0.005, 0.046] | 0.08 | 0.65 / 0.71 | 0.17 |
| Qwen3-Next-80B | 0.32 [0.27, 0.38] | 0.46 | 0.19 [0.15, 0.24] | 0.15 | **0.088** [0.036, 0.134] | +0.018 [0.000, 0.037] | 0.13 | 0.57 / 0.60 | 0.53 |
| DeepSeek V3.2 | 0.30 [0.25, 0.36] | 0.53 | **0.40** [0.34, 0.45] | 0.23 | −0.013 [−0.082, 0.045] | +0.036 [0.015, 0.058] | 0.11 | 0.72 / 0.78 | 0.27 |
| Qwen3-32B | 0.24 [0.19, 0.29] | 0.40 | **0.43** [0.38, 0.49] | 0.28 | 0.022 [−0.086, 0.120] | +0.013 [−0.010, 0.036] | 0.15 | 0.55 / 0.63 | 0.43 |

**Column definitions:**

- **retest flip:** the share of cases whose top diagnosis differs between two
  byte-identical requests.
- **NOE:** the order effect beyond the retest noise, as a share of the
  distance to a different patient.
- **Ensemble gain:** top-1 of the mean of the 8 meaning-preserving answers
  (base, 3 re-orderings, duplication, paraphrase, irrelevant cue, retest)
  minus the mean single-answer top-1.
- **Variance share:** (single-answer Brier − ensemble Brier) / single-answer
  Brier. For a squared-error score this difference is exactly the answers'
  spread around their mean.
- **AUROC:** how well 1 − stated confidence, or the disagreement of the 7
  other answers with the base top-1, detects that the base answer is wrong.
  The difference (disagreement − confidence) has a CI excluding 0 for:

  | model | dataset | difference [95% CI] |
  |---|---|---|
  | DeepSeek V3.2 | DDXPlus | +0.20 [0.13, 0.28] |
  | Qwen3-32B | DDXPlus | +0.21 [0.13, 0.30] |
  | gpt-oss-120b | MedCaseReasoning | +0.07 [0.01, 0.14] |
  | Kimi K3 | DDXPlus | **−0.11** [−0.18, −0.04] (confidence is better) |

**Other presentation effects (normalised, beyond retest noise).** Paraphrase,
duplication and irrelevant-cue effects have CIs that include 0 for almost
every model. The exceptions:

- **Kimi K3:** small cue effects, 0.006 [0.001, 0.011] on DDXPlus and
  0.029 [0.003, 0.057] on real cases;
- **Qwen3-Next-80B:** paraphrase effects, 0.033 [0.003, 0.063] on DDXPlus and
  0.107 [0.044, 0.170] on real cases.

## 3. Evidence and authority

- **Removing one diagnostically informative finding** moves a single answer
  beyond the model's own 95th-percentile retest noise in 0–13% of cases. The
  direction of the gold diagnosis's change agrees with the corpus likelihood
  ratio only at 0.48–0.60 (chance band 0.45–0.55).

  With single samples this cannot separate "ignores the evidence" from
  "noise swamps a correct update". In addition, the tempered-Bayes reference
  itself moves little under single-finding ablation on DDXPlus
  (docs/theory.md, Finding 1). The `resample` arm (Amendment 1) answers this
  on mean posteriors.
- **An attending's suggestion of a wrong but plausible diagnosis,** stated to
  be right 50% of the time, shifts the suggested diagnosis's log-odds by:
  - 0.60–0.79 of the Bayes-normative amount (log 48 on DDXPlus) for five
    models, i.e. they under-weight a reliable signal;
  - 1.12 for Qwen3-32B, which over-weights it.

  On real cases the rates are 0.83–0.91 for four models, against 1.43 for
  Qwen3-32B and 1.11 for Qwen3-Next-80B. Top-1 under the wrong suggestion
  falls from 51% to 22% for Qwen3-32B (DDXPlus), and from 84% to 64% for
  Kimi K3. Every model's exchange rate differs from 1 (all CIs exclude the
  normative value).
- **Stated vs named diagnosis:** the `dx` field matches the stated
  posterior's argmax in ≥ 87% of cases (100% for Kimi K3 and gpt-oss-120b).
  The "knows vs states" gap of the previous study is essentially absent when
  both come from one JSON answer.

## 4. Pre-registered hypothesis H1

H1 predicted that every model would have NOE > 0 (CI lower bound) on both
datasets, and an authority exchange rate ≠ 1.

- **Exchange rate ≠ 1:** supported for all six models.
- **NOE > 0:** supported only for Qwen3-Next-80B (both datasets) and
  gpt-oss-120b (DDXPlus).

For the other four models, re-ordering adds nothing measurable beyond retest
noise. This is reported as a failed prediction, not reinterpreted. It is the
reason the noise itself became an estimand (Amendment 1).

## 5. Reasoning and stability

- **Two reasoning models.** Kimi K3 (≈ 3,700 output tokens per answer, mostly
  reasoning) and gpt-oss-120b (≈ 1,200) reason before answering. The other
  four models answer directly (≈ 180–220 tokens).
- **The most stable and most accurate model is the heaviest reasoner.**
- **Within a reasoning model,** cases with longer reasoning are *less* stable
  (Spearman 0.31–0.39 between output length and retest JS), most likely
  because harder cases elicit both.

The pre-registered `effort_arm` (gpt-oss-120b at low / medium / high) is the
causal test.

## 6. Spend and operations

**Spend:** ledger reported cost $240.36 in total. The guard showed about $481
committed, because Flex calls are counted at full price until Flex billing is
confirmed.

| job | reported cost |
|---|---|
| preprocessing | $29.73 |
| pilot | $10.41 |
| main panel | $200.19 |
| probes | $0.02 |

Kimi K3 is 95% of the panel cost: $0.026 per DDXPlus call and $0.038 per
real case, against ≤ $0.001 for every other model.

| incident | effect | fix (this repo, after the run) |
|---|---|---|
| job caps hit at about half the real spend (pilot $15, panel $250) | PI raised caps by hand | `ccrl billing confirm-flex`: checks Cost Explorer, then re-counts Flex calls at their real price |
| disk full during the panel (`OSError: No space left on device`), mainly from ~0.5 GB ledger backups | run stopped; resumed cleanly, nothing re-sent | disk check before each job (≥ 5 GB), keep only the newest 3 backups, raw-mirror write failures no longer stop a run |
| 135 of 36,504 raw-mirror lines merged by concurrent writes (Kimi K3, gpt-oss-120b) | none: the ledger holds every response; the export is built from the ledger | per-file lock on the mirror |
| preprocessing interrupted twice (Mac sleep / shell exit) | 8 verify attempts counted as uncertain (≤ $0.44 reserved) | runbook guidance; resumes are idempotent |

Pilot gates (RUNBOOK §6) all passed:

- parse rate ≥ 98.2%;
- informative answers ≥ 99.5%;
- no refusals or truncations;
- retry-free in 98.3% of attempts.

## 7. Follow-up

Done since (25 Nov 2026), in [results_secondary.md](results_secondary.md):

- the secondary arms, replication and Amendment 1's `temp0` and `resample`;
- the training extraction;
- the frozen inputs, committed to the repository.

One result there changes how §1 should be read. At temperature 0, the noise falls and
systematic order and paraphrase effects appear. NOE ≈ 0 at provider defaults therefore means
"hidden by noise", not "invariant".

Still open: clinician review of the cue bank and paraphrases, and a rating of whether retest
flips are clinically meaningful (docs/papers.md).
