# OSF amendment 1 (draft, 18 Nov 2026)

*File this on OSF **before** running `temp0` or `resample` and before any
training run. Attach the new freeze hash after re-freezing:
`shasum -a 256 artifacts/frozen/FREEZE.json` →* `__________`
*The previous freeze file is kept in `artifacts/frozen/freeze_history/`.*

## 1. Status at the time of the amendment

**Completed:** `pilot` (1,980 requests) and `panel_main` (36,000 requests:
600 cases × 10 calls × 6 models), on 17–18 Nov 2026, under the original
freeze.

**Not yet run:**

- the pre-registered secondary arms: `effort_arm`, `sequential`,
  `panel_ext`, `distill`;
- all training (Paper 2).

**Results seen before this amendment.** The main-panel estimands
(`docs/results_panel_main.md`) and a reliability analysis of the same data
(retest flip rate, ensemble gain, error detection). No secondary arm and no
training result has been seen.

## 2. What we observed that motivates the amendment

At provider-default decoding, answers to byte-identical requests differ
substantially (retest JS 0.05–0.35 bits; the leading diagnosis changes in
9–43% of cases).

The pre-registered order-effect estimand (NOE) is normalised by this retest
noise, so it measures *systematic* effects only. It is ≈ 0 for four of the
six models: H1 is not supported for those models, and that is reported as
such. The noise itself, which a clinician would experience, is not a
pre-registered estimand.

## 3. Additions (Paper 1, exploratory but specified in advance)

**Two new Bedrock arms** (`config/jobs.yaml`, priority B, after
`panel_main`):

- **`temp0`.** The core 150 cases (75 DDXPlus + 75 MedCaseReasoning), calls
  base, perm1, para, cue, ablate and retest, all 6 models, at temperature 0.
  - *Question:* does deterministic decoding remove the instability, and does
    it reveal systematic presentation effects hidden by noise?
  - *Analysis:* estimands and reliability as for the main panel, reported
    separately. Models that reject or ignore temperature 0 are reported as
    such.
- **`resample`.** The 300 DDXPlus panel cases, calls base and ablate,
  3 independent replicates each, all 6 models. Combined with the main
  panel's base, retest and ablate answers, this gives 5 base and 4 ablated
  answers per case.
  - *Question:* once the noise is averaged out, does removing an informative
    finding move the gold diagnosis in the direction the corpus likelihood
    ratio implies, and by how much relative to the tempered-Bayes reference?
  - *Estimands* (`ccrl.analysis.stability.updating`):
    - `dir_agree_mean`, direction agreement on mean log-odds;
    - `detectable`, the share of cases whose mean change exceeds 2 SE;
    - `slope_vs_reference`.

**New estimands on existing and future panel data**
(`ccrl.analysis.stability.reliability`):

- the retest flip rate;
- the mean and median retest JS;
- single-answer vs 8-answer-mean top-1 and Brier, and the variance share of
  the Brier score;
- AUROC for detecting a wrong base answer by stated confidence vs by
  disagreement of the other answers.

All have case-clustered bootstrap CIs.

## 4. Changes to the training hypotheses (Paper 2)

These are made before any training run.

- **H2.** The primary endpoint becomes *total incoherence*: mean JS between
  the base answer and the meaning-preserving answers, retest included, not
  noise-normalised. The ≥ 25% relative reduction vs RLVR, Holm-corrected
  over order, duplication, paraphrase and cue, is unchanged. NOE becomes
  secondary.
- **H5.** The same endpoint as H2.
- **H6.** The trained Qwen3-8B (single answer, mean over 3 seeds) has lower
  total incoherence **and** a lower retest flip rate than untrained Qwen3-32B
  and than the panel median, on the confirmatory DDXPlus items. One-sided
  case-clustered bootstrap, α = 0.05, Holm over the four comparisons.

  The original H6 used NOE. NOE is ≈ 0 for most panel models, so no model
  could be shown to beat them on it.
- **H7 (new).** CCR+acc+fid closes ≥ 50% of the base model's ensemble gap:
  (8-answer-mean top-1 − single-answer top-1), and the same for Brier.
- **New baselines:**
  - base model at temperature 0;
  - base model 8-answer ensemble;
  - SFT on the base model's own 8-answer mean (self-consistency
    distillation).

## 5. Unchanged

- Data, items, prompts and models.
- The pre-registered analysis of every existing estimand.
- Exclusion rules.
- H3 and H4.
- The secondary arms already registered.

## 6. Budget

At measured per-call costs, the two new arms cost about $80. Total expected
real spend is about $617 of the $1,100 credit, under the harness hard cap of
$1,000.
