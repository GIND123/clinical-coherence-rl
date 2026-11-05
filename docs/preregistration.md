# Pre-registration (OSF draft)

*Register on OSF before `ccrl run panel_main` (target 14 Nov 2026). Paste the
freeze hash: `shasum -a 256 artifacts/frozen/FREEZE.json` →* `__________`

## 1. Title
Coherence as a Reward: Label-Free Reinforcement Learning for Rational Belief
Revision in Clinical Reasoning Models.

## 2. Research questions
* **RQ1** Do hosted frontier and open-weight models give diagnostic posteriors
  that are invariant to meaning-preserving re-presentation yet sensitive to
  evidence, and how does this change with reasoning effort and scale?
* **RQ2** Does the coherence-gated reward (CCR) improve invariance on free-text
  inputs beyond canonicalisation, BCT, a differentiable consistency loss,
  accuracy-only RLVR and RLCR, without losing accuracy?
* **RQ3** Does label-free CCR training on unlabeled real case reports improve
  invariance on held-out real cases?
* **RQ4** Do gains transfer to held-out pathologies, real cases and
  non-medical Bayesian tasks?

## 3. Data and sampling (frozen; seeds in `config/study.yaml`, master seed 20261105)
DDXPlus test split: 300 panel cases (6 per pathology × 49 + 6), 20 pilot,
300 extension; MedCaseReasoning test split: 150 panel, 10 pilot (first passing
cases in a frozen order). Selection rules, filters and the 8 held-out training
pathologies are in `docs/design.md` §1. Pilot cases are disjoint from the
panel and are not used in confirmatory analyses.

## 4. Estimands (docs/design.md §4)
NOE and normalised duplication, paraphrase and cue effects (0 = retest
noise, 1 = different patient); evidence sensitivity; direction agreement with
corpus log-odds (DDXPlus); authority exchange rate; top-1, top-3, Brier, ECE;
knows-vs-states ratio; entropy; cross-patient information; sequential
commutativity gap.

## 5. Hypotheses
* **H1** Every hosted model has an order effect above its retest floor (NOE
  95 % CI lower bound > 0 on DDXPlus) and an authority exchange rate different
  from 1.
* **H2 (superiority)** CCR+acc+fid reduces free-text NOE relative to
  accuracy-only RLVR by ≥ 25 % (relative), Holm-corrected across the four
  invariance estimands (order, duplication, paraphrase, cue).
* **H3 (non-inferiority)** Top-1 on held-out DDXPlus within Δ = 2.0 points of
  RLVR; MMLU-Pro and GSM8K within Δ = 1.5 points of the base model. One-sided
  95 % paired-bootstrap lower bound > −Δ. If the pilot seed-to-seed spread is
  larger than 2Δ, Δ is fixed by an OSF amendment before the main runs.
* **H4 (transfer)** On held-out pathologies and MedCaseReasoning, the CCR
  improvement in NOE retains ≥ 50 % of the in-distribution improvement.
* **H5 (label-free arm)** Training on unlabeled MedCaseReasoning cases reduces
  NOE on held-out real cases vs the base model (superiority) and is
  non-inferior on top-3 (Δ = 3 points).
* **H6** The fidelity arm raises DDXPlus direction agreement above the chance
  band (0.45–0.55). *Not* predicted for the label-free arm (Proposition 2).

## 6. Analysis
Mixed-effects models with case and pathology random intercepts; fixed
effects for model, transformation, dataset. Case-clustered bootstrap (2,000).
Holm within each hypothesis family. Floors and ceilings measured per model.
Power: simulation from pilot JS distributions; if 300 DDXPlus cases are
under-powered for a 25 % NOE reduction at 80 % power, the priority-C2
extension is triggered.

## 7. Exclusions and missing data
A response is a parse failure only if no answer object can be read (all-zero
answers are uniform, not failures). Each request gets one fresh retry on
parse failure, one on truncation, one on timeout; the final attempt is
analysed. Parse rates are reported per model and transformation; if they
differ across transformations by > 5 points for a model, its conditional
estimands are flagged and only unconditional ones are interpreted.
Refusals are reported, not imputed.

## 8. Deviations already known
Listed in `docs/design.md` §6 and `docs/theory.md` §3 (triage contrast,
batch-level sensitivity gate, tempered reference posterior,
label-permutation ceiling for MedCaseReasoning), all decided before any panel
call.

## 9. Reporting
TRIPOD-LLM-aligned checklist; data and model cards; DECIDE-AI note that this
is not a clinical deployment study. All raw requests and responses archived
(Zenodo DOI) with SHA-256 manifests.
