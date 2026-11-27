# Amendment 3 (27 Nov 2026): a contamination-free real-case replication

*Written and committed publicly **before** any call below was made. The design runs under the
freeze `51c85394fcb175cb5ca257609222eb99e33f627fae694a0356f2509aab59f089`
(`artifacts/frozen/FREEZE.json`). That freeze adds only this amendment's job and two prompts to
Amendment 2's freeze, so every Amendment 2 input is unchanged. The selected case reports are in
`artifacts/frozen/fresh/cases_selected.jsonl`
(SHA-256 `352dddfaa0b9fef2c2893f031a94a292e99b3bae28bdb0bb3f24d1526a22a95e`).*

This amendment concerns Paper 1 only. It does not change the training study registered on OSF
(osf.io/4xmq7).

## 1. Why

- **No real-case replication.** The panel's real cases (MedCaseReasoning) are published case
  reports from 2005–2025. Its replication sample is synthetic (DDXPlus).
- **Contamination can't be ruled out.** The contamination probe (Amendment 2) can show
  memorisation, but it cannot prove its absence.

A sample of case reports that **no panel model can have seen** answers both points.

## 2. Cases

**Why these cases cannot be in any training data.** Every panel model was live on Bedrock, with
the exact model IDs pinned in `config/models.lock.yaml`, by 6 Nov 2026 (the probe). We use case
reports **first published electronically on or after 7 Nov 2026**. None of them can be in any
panel model's training data, whatever each model's cutoff.

**Selection** (`ccrl data select-fresh`, `src/ccrl/data/fresh.py`; deterministic, no model
involved):

- **Source:** PubMed Central, publication type *Case Reports*, open access under **CC BY**
  (redistributable with attribution; every row keeps its licence and link).
- **Publication window:** 7–27 Nov 2026, which gave 1,041 articles.
- **What is kept:** the case-presentation section, verbatim (figure and table captions and
  citation marks removed).
- **Exclusions:**

  | reason | articles |
  |---|---|
  | no identifiable case-presentation section | 643 |
  | section longer than 6,000 characters | 115 |
  | more than one patient (title or text) | 24 |
  | published before 7 Nov 2026 by electronic date | 13 |
  | section shorter than 700 characters | 8 |

- **Result:** 238 eligible reports, ordered by a seeded hash of their PMCID. All are sent to
  preparation.

**Preparation** (`ccrl fresh run`, `src/ccrl/prep/fresh.py`):

1. **Cut (Kimi K3).** Given the numbered sentences, Kimi K3 states whether the report covers one
   patient, the final diagnosis, whether the case is genuinely diagnostic (the diagnosis is not
   known at presentation), and the last sentence that can be shown before the diagnosis is given
   away. **The vignette is the authors' own first *k* sentences.** Nothing is rewritten.
2. **Screen (deterministic).** A case is rejected if any of these holds:
   - it covers more than one patient;
   - it is not a diagnostic case;
   - it has no diagnosis;
   - the vignette is under 4 sentences or 400 characters;
   - a distinctive word of the diagnosis appears in the vignette.
3. **Extract (Kimi K3) and verify (GLM-5),** with the **same prompts** as the MedCaseReasoning
   panel.
4. **Finalise,** with the **same deterministic rules** as the panel (`finalize_case`, 10
   candidates).
5. **Sample size.** **Every** passing case is used, up to the 238 selected. (The first draft of
   this amendment, 27 Nov 2026, capped this at 150; it was raised the same day, before any call,
   so that the replication rests on as many real cases as exist.) If fewer than 60 pass, the run
   stops before the panel.
6. **Build** items with the panel's renderer (the same 10 calls per case). Their SHA-256s are
   written to `artifacts/frozen/fresh/FREEZE_fresh.json` **before** the first panel call.

## 3. Panel arm

**`fresh_panel`:** every passing case (at most 238) × 10 calls × all 6 panel models, at provider-default
decoding, as in the main panel.

## 4. Pre-specified analyses

These are the same estimands as the main panel, per model (`ccrl analyze`; job
`fresh_panel`):

- the retest flip rate and ensemble gain;
- AUROC for confidence versus disagreement;
- NOE and the paraphrase, duplication and cue effects;
- the authority exchange rate, top-1, Brier score and ECE.

**Claims tested:**

1. **Replication of instability on real cases the models cannot have seen.** For each model,
   the retest flip rate on fresh cases has its 95% CI lower bound above 0. It is compared with
   the same model's MedCaseReasoning panel rate (unpaired; bootstrap CI of the difference).
2. **Contamination check by comparison.** The accuracy difference, MedCaseReasoning panel minus
   fresh, is reported per model with a bootstrap CI. A large drop on fresh cases at similar
   difficulty would suggest that panel accuracy was inflated. Differences in case mix
   (specialty, length) are reported alongside, so the comparison is descriptive, not causal.
3. **Directional replications.** These have no new thresholds:
   - disagreement beats confidence for the noisy models (DeepSeek V3.2, Qwen3-32B);
   - Qwen3-32B over-weights authority (exchange rate > 1);
   - the ensemble gain is positive.

   Each is reported as replicated or not.

## 5. Budget

| part | expected cost |
|---|---|
| Cut, 238 Kimi K3 calls | ~$3–13 |
| Extraction and verification | ~$12 |
| Panel, ~170 cases (if ~70% pass) × 10 calls × 6 models | ~$68, of which Kimi K3 is ~95% |
| **Total** | **~$90**; at most ~$125 if every case passes |

The job caps are $40, $40, $5 and $130. Calls must finish by 18 Dec 2026.
