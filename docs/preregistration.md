# Pre-registration (OSF draft)

*Register on OSF (osf.io → "OSF Preregistration" template) **after**
`ccrl freeze` and **before** `ccrl run panel_main`. Paste the freeze hash:
`shasum -a 256 artifacts/frozen/FREEZE.json` →* `__________`

## 1. Title
Coherence as a Reward: Label-Free Reinforcement Learning for Rational Belief
Revision in Open-Weight Clinical Reasoning Models.

## 2. Research questions
See `docs/problem_statement.md` §4.
* **RQ1** Are open-weight models' diagnostic posteriors invariant to
  meaning-preserving re-presentation and sensitive to evidence?
* **RQ2** Does the coherence-gated reward (CCR) improve invariance beyond
  accuracy-only RLVR (and RLCR, BCT, SFT baselines) without losing accuracy?
* **RQ3** Do the gains transfer to held-out pathologies and real cases?
* **RQ4** Does label-free CCR training on unlabeled real case reports improve
  invariance on held-out real cases?
* **RQ5** Is a CCR-trained small model more coherent than much larger
  untrained open-weight models?

## 3. Data and sampling (frozen; seeds in `config/study.yaml`, master seed 20261105)
* **Confirmatory panel.** DDXPlus test split, 300 cases (6 per pathology × 49
  + 6). MedCaseReasoning test split, 300 cases: the first 300 that pass
  verification, in a frozen seeded order of 420.
* **Replication sample.** 300 further DDXPlus test cases, disjoint from the
  panel.
* **Pilot.** 20 DDXPlus and 10 MedCaseReasoning cases, disjoint from both;
  never used in confirmatory analyses.
* **Training.** DDXPlus train split with 8 held-out pathologies excluded
  (seeded draw per body system); 2,000 MedCaseReasoning train-split cases,
  whose gold diagnoses are never used by the label-free reward.

Selection rules and filters are in `docs/design.md` §1.

## 4. Models
* **Panel (Amazon Bedrock, us-east-1):** Kimi K3, GLM-5, DeepSeek V3.2,
  Qwen3-Next-80B-A3B, gpt-oss-120b (reasoning effort medium), Qwen3-32B.
  Exact model IDs and accepted parameters are in `config/models.lock.yaml`,
  which is covered by the freeze hash. No temperature or top-p is sent.
* **Trained (local):** Qwen3-4B and Qwen3-8B with LoRA.
* **Optional, exploratory:** closed partner models, only if AWS enables them;
  reported separately and never pooled with the confirmatory panel.

## 5. Estimands (docs/design.md §4)
* **Invariance:** NOE and the normalised duplication, paraphrase and cue
  effects (0 = retest noise, 1 = different patient).
* **Evidence:** evidence sensitivity; direction agreement with corpus
  log-odds (DDXPlus).
* **Authority:** authority exchange rate.
* **Accuracy and calibration:** top-1, top-3, Brier, ECE; knows-vs-states
  ratio.
* **Safeguards:** entropy; cross-patient information.
* **Sequential:** commutativity gap.

## 6. Hypotheses (confirmatory)
* **H1** Every panel model has an order effect above its retest floor (NOE
  95% CI lower bound > 0) on DDXPlus and on MedCaseReasoning, and an
  authority exchange rate different from 1.
* **H2 (superiority)** CCR+acc+fid reduces free-text NOE relative to
  accuracy-only RLVR by ≥ 25% (relative), Holm-corrected across the four
  invariance estimands (order, duplication, paraphrase, cue).
* **H3 (non-inferiority)** Top-1 on held-out DDXPlus is within Δ = 2.0 points
  of RLVR, and MMLU-Pro and GSM8K are within Δ = 1.5 points of the base model
  (one-sided 95% paired-bootstrap lower bound > −Δ). If the seed-to-seed
  spread in the first training seeds exceeds 2Δ, Δ is fixed by an OSF
  amendment before the remaining seeds run.
* **H4 (transfer)** On held-out pathologies and on MedCaseReasoning, the CCR
  improvement in NOE retains ≥ 50% of the in-distribution improvement.
* **H5 (label-free arm)** Training on unlabeled MedCaseReasoning cases reduces
  NOE on held-out real cases vs the base model (superiority), and is
  non-inferior on top-3 (Δ = 3 points).
* **H6 (scale)** The CCR-trained Qwen3-8B (CCR+acc+fid, mean over 3 seeds) has
  lower free-text NOE than untrained Qwen3-32B and than the median panel model
  on the confirmatory DDXPlus items (one-sided case-clustered bootstrap,
  α = 0.05, Holm over the two comparisons).

## 7. Secondary (pre-registered, not confirmatory)
* The fidelity arm raises DDXPlus direction agreement above the chance band
  (0.45–0.55). This is *not* predicted for the label-free arm
  (Proposition 2).
* The H1 pattern replicates on the 300-case replication sample.
* Effort dose-response in gpt-oss-120b (low / medium / high) for NOE, cue
  effect and accuracy; no directional prediction.
* The sequential commutativity gap exceeds the retest floor for each model.

## 8. Analysis
* **Models.** Mixed-effects models with case and pathology random intercepts;
  fixed effects for model, transformation and dataset.
* **Intervals and corrections.** Case-clustered bootstrap (2,000 resamples);
  Holm within each hypothesis family. Floors and ceilings are measured per
  model, per dataset.
* **Comparing trained and panel models.** Each model is normalised by its own
  retest floor, so sampling noise does not favour either side. Trained models
  are decoded at temperature 0.7, top-p 0.95; panel models at provider
  default.

## 9. Exclusions and missing data
* **Parse failures.** A response is a parse failure only if no answer object
  can be read. All-zero answers count as uniform, not as failures.
* **Retries.** Each request gets one fresh retry on parse failure, one on
  truncation and one on timeout; the final attempt is analysed.
* **Uneven parse rates.** Parse rates are reported per model and
  transformation. If they differ across transformations by > 5 points for a
  model, its conditional estimands are flagged and only unconditional ones are
  interpreted.
* **Refusals** are reported, not imputed.
* **Go/no-go.** A model that fails the pilot gates (`RUNBOOK.md` §6) is
  dropped before the panel by an OSF amendment, with the reason.

## 10. Deviations already known
* Planning document vs this version: `docs/problem_statement.md` §7.
* Method deviations: `docs/design.md` §6 and `docs/theory.md` §3 (triage
  contrast, batch-level sensitivity gate, tempered reference posterior,
  label-permutation ceiling for MedCaseReasoning).

All were decided before any panel call.

## 11. Reporting
* TRIPOD-LLM-aligned checklist, plus data and model cards.
* A DECIDE-AI statement that this is not a clinical deployment study.
* All raw requests and responses archived (Zenodo DOI) with SHA-256
  manifests; adapters and evaluation outputs released.
