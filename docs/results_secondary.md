# Results: secondary arms (run 24–25 Nov 2026, analysed 25 Nov 2026)

**Source.** The PI's archive `ccrl-archive-20261125-144504.tar.gz`, with SHA-256
`18782203…4624e`:

- all nine export files match their manifest;
- ledger `PRAGMA integrity_check`: ok;
- no attempt left in flight;
- frozen inputs byte-identical to the repository (freeze `ea6f2a24…`).

**What ran.** Every planned request was completed, except `temp0` on Kimi K3. Bedrock rejects the
`temperature` field for that model ("This model doesn't support the temperature field"). The
guard stopped after 4 rejected calls, at no cost. Amendment 1 had pre-specified this case: models
that reject temperature 0 are reported as such.

| job | design | requests | reported cost |
|---|---|---|---|
| `panel_ext` | 300 new DDXPlus cases × 10 calls × 6 models (replication) | 18,000 | $84.64 |
| `resample` | 300 DDXPlus panel cases × (base, ablate) × 3 replicates × 6 models | 10,800 | $47.44 |
| `distill` | 2,000 DDXPlus training renderings × Kimi K3 (teacher) | 2,000 | $46.82 |
| `sequential` | 150 cases × 2 orders × 2 turns × 6 models | 3,600 | $12.68 |
| `effort_arm` | core 150 × 6 calls × {low, medium, high} × gpt-oss-120b | 2,700 | $1.86 |
| `temp0` | core 150 × 6 calls × 5 models at temperature 0 | 4,500 | $1.59 |
| training extraction | 2,150 MedCaseReasoning training cases (Kimi K3) | 2,150 | ≈ $124 |

The study total is **$560.02** reported, of the $1,100 credit; all calls were billed at the Flex
tier. **Parse rates** are ≥ 96% in every arm, with one exception: DeepSeek V3.2 in `sequential`
parsed at 91%, so its sequential estimate is flagged.

**Analysis.** All results come from `ccrl analyze`, which writes:

- `estimands.csv`, `stability.csv` and `updating.csv`;
- `paired_vs_panel.csv` and `sequential_gap.csv` (`ccrl.analysis.secondary`).

**Arm comparisons are paired.** Each arm is compared with the main panel's answers to the *same
cases*. Intervals are case-bootstrap 95% CIs. Top-1 is the argmax of the stated posterior.

**Registration status.** These arms were specified in Amendment 1, which was committed publicly on
18 Nov 2026 (`3e17eb5`). They were run, and then analysed, before any OSF filing.

## 1. Headline

1. **Everything replicates.** On 300 new DDXPlus cases, every main-panel finding reproduces:
   - accuracy, within 1.3 points;
   - the retest flip rate;
   - the ensemble gain (DeepSeek V3.2 +14.0 points);
   - "disagreement beats confidence" for the noisy models;
   - the authority exchange rates.
2. **Temperature 0 reduces the instability but does not remove it.** Hosted "temperature 0"
   is not deterministic:
   - only 0–41% of byte-identical requests return byte-identical text;
   - the leading diagnosis still flips in 5–20% of cases.
3. **Once the noise is reduced, systematic presentation effects appear.** This is the new
   result. At temperature 0, re-ordering and rewording the same findings change the answer
   *beyond* the model's own noise:
   - Qwen3-32B on both datasets;
   - DeepSeek V3.2 on DDXPlus;
   - GLM-5 for paraphrase.

   At default decoding the noise hid these effects. NOE ≈ 0 in the main panel therefore did
   not mean "invariant"; it meant "too noisy to tell".
4. **DeepSeek V3.2's sampling settings cost it a lot of accuracy.** At temperature 0 its top-1
   rises by 18.9 points on DDXPlus [6.8, 31.1] and by 10.7 points on real cases [4.0, 18.7].
   This is the size of its ensemble gain. For the other models, temperature 0 does not change
   accuracy.
5. **The models barely track the evidence, even with the noise averaged out.** On mean
   posteriors (5 base and 4 ablated answers per case):
   - removing a finding moves the gold diagnosis in the corpus-implied direction in only
     49–60% of cases;
   - the size of the move is uncorrelated with the Bayes reference (r ≤ 0.10).
6. **The order of evidence matters across turns.** Giving the same two halves of the evidence
   in the opposite order changes the final top diagnosis in 8–44% of cases. For four models,
   the posteriors differ by more than one-shot retest noise.
7. **More reasoning effort is not a fix.** For gpt-oss-120b, low, medium and high effort do not
   differ in stability. Low effort costs 9 points on real cases.

## 2. Replication (`panel_ext`, 300 new DDXPlus cases; pre-registered secondary)

| model | top-1 main → replication | retest flip main → replication | ensemble gain, top-1 (replication) [CI] | AUROC disagreement − confidence (replication) [CI] | authority exchange rate (replication) [CI] |
|---|---|---|---|---|---|
| Kimi K3 | 0.84 → 0.83 | 0.10 → 0.09 | +0.023 [0.009, 0.037] | **−0.10** [−0.16, −0.03] | 0.63 [0.59, 0.67] |
| gpt-oss-120b | 0.74 → 0.74 | 0.13 → 0.17 | −0.003 [−0.021, 0.016] | +0.05 [−0.01, 0.11] | 0.64 [0.60, 0.68] |
| GLM-5 | 0.70 → 0.71 | 0.17 → 0.18 | +0.027 [0.006, 0.046] | +0.03 [−0.03, 0.08] | 0.75 [0.71, 0.79] |
| Qwen3-Next-80B | 0.67 → 0.68 | 0.09 → 0.12 | +0.022 [0.008, 0.037] | +0.00 [−0.09, 0.10] | 0.65 [0.59, 0.71] |
| DeepSeek V3.2 | 0.65 → 0.64 | 0.41 → 0.48 | **+0.140** [0.110, 0.168] | **+0.25** [0.17, 0.32] | 0.65 [0.60, 0.71] |
| Qwen3-32B | 0.51 → 0.52 | 0.34 → 0.35 | +0.028 [0.003, 0.052] | **+0.27** [0.19, 0.35] | **1.13** [1.06, 1.20] |

**H1 on the replication sample.** All six exchange rates differ from 1 again.

NOE has a CI lower bound > 0 for Qwen3-Next-80B again: 0.028 [0.002, 0.051]. For GLM-5 the lower
bound is exactly 0: 0.016 [0.000, 0.032]. gpt-oss-120b's main-panel order effect does **not**
replicate: 0.009 [−0.006, 0.024]. Paraphrase, duplication and cue effects include 0 for every
model except GLM-5's paraphrase effect, 0.022 [0.003, 0.042].

## 3. Temperature 0 (`temp0`, core 150 cases; Amendment 1)

All figures below compare the same cases (74–75 per dataset) at provider default and at
temperature 0.

| model | dataset | retest flip: default → temp 0 (Δ [CI]) | byte-identical responses at temp 0 | top-1 Δ [CI] | order effect beyond own noise at temp 0 (bits) [CI] |
|---|---|---|---|---|---|
| DeepSeek V3.2 | DDXPlus | 0.49 → **0.05** (−0.43 [−0.55, −0.31]) | 13% (pooled over both datasets) | **+0.189** [0.068, 0.311] | **+0.055** [0.013, 0.097] |
| DeepSeek V3.2 | real cases | 0.37 → 0.17 (−0.20 [−0.31, −0.08]) | | **+0.107** [0.040, 0.187] | +0.006 [−0.051, 0.062] |
| Qwen3-32B | DDXPlus | 0.24 → 0.14 (−0.11 [−0.22, 0.00]) | 41% (pooled) | 0.000 | **+0.128** [0.038, 0.216] |
| Qwen3-32B | real cases | 0.37 → **0.12** (−0.25 [−0.37, −0.13]) | | +0.013 | **+0.122** [0.058, 0.192] |
| GLM-5 | DDXPlus | 0.23 → 0.11 (−0.12 [−0.23, −0.03]) | 3% (pooled) | +0.013 | +0.018 [−0.003, 0.041] |
| GLM-5 | real cases | 0.33 → 0.19 (−0.15 [−0.25, −0.04]) | | −0.027 | +0.027 [−0.008, 0.064] |
| gpt-oss-120b | DDXPlus | 0.16 → 0.11 (n.s.) | 0% (pooled) | 0.000 | +0.024 [−0.002, 0.053] |
| gpt-oss-120b | real cases | 0.27 → 0.20 (n.s.) | | −0.067 [−0.147, 0.000] | **+0.037** [0.006, 0.075] |
| Qwen3-Next-80B | DDXPlus | 0.08 → 0.09 (n.s.) | 18% (pooled) | −0.013 | +0.041 [−0.007, 0.096] |
| Qwen3-Next-80B | real cases | 0.13 → 0.15 (n.s.) | | −0.013 | **+0.079** [0.015, 0.147] |
| Kimi K3 | both | not available: Bedrock rejects `temperature` | | | |

**Byte-identical responses.** At provider default, 0–4% of identical requests return identical
text. At temperature 0, 0–41% do.

**Order effect beyond own noise.** This is JS(base, re-ordered) − JS(base, verbatim retest), per
case, averaged. At default decoding the same quantity is at most 0.07 and includes 0 for every
model except Qwen3-Next-80B on real cases (+0.069 [0.014, 0.127]). Paraphrase shows the same pattern at temperature 0:

- Qwen3-32B: +0.165 [0.070, 0.259] on DDXPlus and +0.129 [0.069, 0.193] on real cases;
- DeepSeek V3.2: +0.060 [0.018, 0.101] on DDXPlus;
- GLM-5: +0.027 [0.001, 0.054] on DDXPlus and +0.039 [0.007, 0.073] on real cases.

**Interpretation.** Provider-default sampling adds noise that is large for DeepSeek V3.2,
Qwen3-32B and GLM-5. Removing most of it reveals two things:

- whatever accuracy the sampling had been costing (DeepSeek V3.2);
- a genuine dependence on how the case is written (Qwen3-32B, and DeepSeek V3.2 on DDXPlus).

Serving non-determinism (batching and floating-point order) keeps a residual floor even at
temperature 0. This matches reports that hosted inference is not batch-invariant.

## 4. Evidence updating on mean posteriors (`resample`; Amendment 1)

There are 300 DDXPlus panel cases, each with 5 base answers (panel base, retest and 3
replicates) and 4 answers with one diagnostically informative finding removed.

| model | change beyond 2 SE (detectable) [CI] | direction agrees with corpus LR [CI] (chance 0.5) | correlation with tempered-Bayes reference |
|---|---|---|---|
| Kimi K3 | 0.24 [0.20, 0.29] | **0.60** [0.55, 0.66] | 0.10 |
| Qwen3-32B | 0.12 [0.08, 0.16] | **0.60** [0.54, 0.65] | 0.05 |
| gpt-oss-120b | 0.19 [0.15, 0.23] | 0.57 [0.51, 0.62] | 0.00 |
| GLM-5 | 0.18 [0.14, 0.22] | 0.56 [0.50, 0.62] | −0.01 |
| Qwen3-Next-80B | 0.21 [0.16, 0.26] | 0.55 [0.49, 0.60] | 0.02 |
| DeepSeek V3.2 | 0.11 [0.08, 0.15] | 0.49 [0.43, 0.54] | −0.02 |

Averaging removes most of the sampling noise, yet a single informative finding still changes the
mean answer detectably in only 11–24% of cases. Where it does, the direction is right only modestly
more often than chance. The size of the change does not track the Bayes reference.

**Caveat.** On DDXPlus the reference posterior itself moves little under single-finding ablation
(docs/theory.md, Finding 1), so the correlation has little variance to explain. Direction
agreement is the more robust measure. This answers claim F6 of Paper 1: the weak single-sample
updating of the main panel is not only noise, because the models are also weakly evidence-driven.

## 5. Sequential evidence (`sequential`; pre-registered secondary)

There are 150 DDXPlus cases. The evidence is split into two halves and given over two turns, in
order A→B or B→A.

| model | final top diagnosis differs between orders | JS(A→B, B→A) − one-shot retest JS [CI] |
|---|---|---|
| Qwen3-32B | 0.44 | **+0.063** [0.006, 0.118] |
| DeepSeek V3.2 (parse 91%, flagged) | 0.42 | −0.031 [−0.087, 0.029] |
| GLM-5 | 0.24 | **+0.033** [0.008, 0.059] |
| gpt-oss-120b | 0.21 | +0.016 [−0.012, 0.044] |
| Qwen3-Next-80B | 0.19 | **+0.092** [0.050, 0.136] |
| Kimi K3 | 0.08 | **+0.012** [0.003, 0.022] |

The pre-registered prediction was that the commutativity gap exceeds the retest floor for each
model. It holds for 4 of 6 models.

**Caveat.** The floor is a one-shot retest, while the sequential answer comes from two sampled
turns. Part of the gap may therefore be extra two-turn noise, so these gaps are upper bounds on
the systematic order effect.

## 6. Reasoning effort (`effort_arm`, gpt-oss-120b; pre-registered secondary)

These figures compare the core 150 cases with the main panel (which ran at medium effort).

| effort | retest flip DDXPlus / real | retest JS DDXPlus | top-1 DDXPlus / real | Brier DDXPlus |
|---|---|---|---|---|
| low | 0.20 / 0.25 | 0.21 | 0.73 / **0.41** (−0.093 [−0.187, −0.013] vs main panel) | 0.44 |
| medium (re-run) | 0.21 / 0.20 | 0.16 | 0.73 / 0.48 | 0.45 |
| high | 0.13 / 0.27 | 0.10 | 0.77 / 0.52 | 0.33 |

The medium re-run agrees with the main panel's medium-effort answers on the same cases, so the arm
is internally consistent. There is no significant dose-response in the flip rate or in the order
effect.

Higher effort lowers retest JS on DDXPlus (0.21 → 0.10) and improves the Brier score. However, it
does not change the leading-diagnosis flip rate significantly. Low effort costs 9 points of top-1
on real cases. More reasoning makes the probabilities somewhat steadier, but the top answer is
no more stable.

## 7. Distillation teacher and training data

- **`distill`.** Kimi K3 answered all 2,000 DDXPlus training renderings, with 100% parsed and
  top-1 0.868. These answers are the teacher targets for the `sft_distill` baseline.
- **Training extraction.** All 2,150 MedCaseReasoning training cases were extracted.
  `ccrl prep finalize` yields:
  - 2,000 usable training cases, the first 2,000 that pass in the frozen order;
  - 16 rejected by the deterministic rules (findings outside 6–24, or no stage-0 finding);
  - 134 reserve cases.

  They are in `artifacts/frozen/medcase/prepared_train.jsonl` (SHA-256 `1883e4f7…fb11`). This is
  a separate file, so the panel's frozen `prepared.jsonl` is unchanged.

## 8. What this changes

- **Paper 1.** The story is now sharper than "noise dominates":

  > At provider defaults, sampling noise dominates and hides everything else. At temperature 0
  > the noise falls but does not vanish, and systematic sensitivity to how the case is written
  > appears. Averaged over the noise, the models still barely follow the evidence.

  Claims F3, F6 and F7 are now answered (docs/papers.md).
- **Paper 2.** Two baselines now have measured panel analogues: temperature 0 and the ensemble.
  - The coherence reward must beat temperature-0 decoding on total incoherence. This matters
    because temperature 0 alone halves the flip rate for the noisy models.
  - The invariance target is not only noise. Temperature-0 Qwen3-32B, the same family as the
    trained models, shows real order and paraphrase effects that decoding does not fix.
