# Final plan: methodology and budget (27 Nov 2026)

Two papers from one study, as before. This plan sets what each paper claims, the experiments
that test those claims, and what fits in **up to $350 of Bedrock and $100 of Lambda**. The
Lambda $100 includes the $18 already spent on 26–27 Nov, so $82 is left. More GPU money follows
only if stage A works (§3.5). Where this plan differs from [papers.md](papers.md), it replaces it.

OSF updates are set aside for now. Every design change is still dated in this repository before
the runs it affects, so the registration can be brought up to date in one step later.

---

## 1. The two papers

| | Paper 1: clinical evaluation | Paper 2: method |
|---|---|---|
| question | Is an open-weight model's differential a stable, evidence-driven belief? | How do you train a model to be consistent without making it uninformative? |
| data | Bedrock panel (6 models), DDXPlus + MedCaseReasoning + post-freeze case reports | Qwen3-4B trained with GRPO on DDXPlus; same frozen items for evaluation |
| status | main and secondary arms done ($560); Amendments 2–3 not yet run | diagnosis, fix and CPU audit done (Amendment 4); no successful training run yet |
| money | Bedrock up to $350 (§2) | Lambda $82 left of $100 (§3); more if stage A works |
| target | npj Digital Medicine / JAMIA | TNNLS (31 Mar 2027), NeurIPS 2027 |

---

## 2. Paper 1 on Bedrock (up to $350)

### What it buys

The two amendments as written forecast about $156. Up to $200 runs them **in full, Kimi K3
included**. The training paraphrase bank needs no Bedrock call after all (§3.4, P2). Kimi K3 runs
last because it is about 95% of the real-case cost ($0.038 per call against about $0.00035 for
each of the other five models), so a cap can only shorten its share.

| order | job | models | design | expected | cap |
|---|---|---|---|---|---|
| 1 | `temp0_panel` + `temp0_ext` | 5 models (Bedrock refuses temperature 0 for Kimi K3) | 900 cases × 10 calls at temperature 0 | ~$26 | $30 + $15 |
| 2 | `prep_fresh_cut` | Kimi K3 | 238 post-freeze case reports | ~$12 | $15 |
| 3 | `prep_fresh_extract` + `verify` | Kimi K3, GLM-5 | passing cases | ~$13 | $16 + $5 |
| 4 | `fresh_panel` | all 6 | up to 150 passing cases × 10 calls | ~$60 | $70 |
| 5 | `contamination` | all 6 | 300 real cases × 4 probes | ~$40 | $70 |
| 6 | `openended` + `openended_t0` (**Amendment 6**) | 6 / 5 | the same vignettes, free-text five-item differential, no candidate list and no probabilities; 75 DDXPlus + 300 MedCaseReasoning + all post-freeze cases × 4 renderings | ~$90 | $110 + $12 |
| 7 | `openended_judge` | GLM-5, DeepSeek V3.2 | blind mapping of every free-text diagnosis to candidate codes | ~$8 | $20 |
| 8 | `drift` | all 6 | core 150 × (base, retest), about three weeks after the panel | ~$12 | $18 |

- **Why jobs 6–8 ([amendment_6.md](amendment_6.md)).** Bedrock calls end on 18 Dec, so anything a
  reviewer could ask that needs a call must run now.
  - Jobs 6–7 answer "is the instability an artefact of asking for probabilities over a given
    list?", in the format a clinician would use.
  - Job 8 answers "is it stable over time?".
- **Expected total:** about $260. The **hard cap is $911**: the $561.55 already spent plus $350
  (`config/budget.yaml`). The registered jobs 1–5 run first, so a cap can only shorten jobs 6–8.
- **One command** runs everything in this order unattended: `bash scripts/final_bedrock.sh start`.
- **Deadlines:** everything finishes by 18 Dec 2026, the last day for paid calls. None of it
  blocks the GPU work any more.
- **Dated records:** [amendment_5.md](amendment_5.md) §A and [amendment_6.md](amendment_6.md).

### What Paper 1 then claims

- **F1–F8 as now** ([papers.md](papers.md)).
- **Full-scale temperature 0.** Temperature 0 reveals the presentation effects that sampling noise
  hides. This now rests on 900 cases, up from 150.
- **Replication on real cases no model can have seen.** Instability, ensemble gain and the
  disagreement alarm are re-tested on these cases, and their accuracy is compared with that on
  older published cases.
- **Contamination:** direct probes for all six models, cross-checked against the old-versus-new
  accuracy comparison.
- **Clinician ratings (no cost):** whether a flip changes the next management step.

---

## 3. Paper 2 on Lambda ($82 now; more if stage A works)

### 3.1 Contributions and what would refute each

| | claim | evidence | refuted if |
|---|---|---|---|
| C1 | A distributional consistency reward collapses in two independent ways. One is the readout: renormalised filler is a near-free flattening channel. The other is the estimator: group-relative advantages freeze batch-level gates | registered credit + lenient readout (done: collapsed at step 21); the two mechanism cells (§3.2) | registered credit + strict readout does not collapse (the readout was the only cause); or team credit + lenient readout collapses (credit alone is not enough) |
| C2 | Team credit with the strict readout trains without collapse and reduces nuisance incoherence | coherence arm vs RLVR, 3 seeds each | the coherence arm's nuisance flip rate and normalised incoherence are not lower than RLVR's, or its safeguard fires |
| C3 | The coherence term does the work, not the Brier term inside it | coherence arm vs **Brier-only** | Brier-only matches the coherence arm on the nuisance endpoints |
| C4 | Coherence training is *selective*: it removes nuisance sensitivity and keeps the response to evidence, which consistency fine-tuning does not (Gupta et al. 2026) | coherence arm vs BCT and self-consistency distillation, on `ablate` items | the fine-tuning baselines match it on both nuisance and evidence response |
| C5 | One answer captures the ensemble's gain | H7: gap closure | closure < 50% |
| C6 | The gains transfer | held-out pathologies, MedCaseReasoning, post-freeze real cases | under 50% of the gain retained |
| C7 | A CPU estimator audit predicts the GPU outcome | audit predictions committed **before** the GPU runs (§3.4), compared with what the runs do | predictions and runs disagree (reported either way) |

### 3.2 Arms

All arms train Qwen3-4B with LoRA, thinking off, on DDXPlus training patients. Every arm uses the
same rollouts per step (16 patients, 4 + 4 + 2) and runs 300 steps unless stated. The reward
constants stay as fixed before any run (τ_I 0.788, δ_S 0.234, m0 0.459, γ 0.5); nothing is tuned.

| arm | reward | credit | readout | seeds | role |
|---|---|---|---|---|---|
| **CCR** (`ccr_acc_fid`) | Brier + coherence + fidelity | team | strict | 3 | the method |
| **RLVR** (`rlvr`) | binary top-1 | (= reward) | strict | 3 | standard baseline |
| **Brier-only** (new `brier_only`) | Brier | (= reward) | strict | 1 (3 if money is left) | isolates the coherence term (C3) |
| mechanism cell A (new) | as CCR | team | lenient | 1, ≤ 60 steps | C1 |
| mechanism cell B (new) | as CCR | registered | strict | 1, ≤ 60 steps | C1 |
| (done) | as CCR | registered | lenient | attempt 2 | C1: collapsed at step 21 |
| BCT (`bct`) | SFT: own clean answer as target for perturbed inputs | – | – | 1 | consistency fine-tuning baseline (C4) |
| self-consistency distillation (new `sft_selfcons`) | SFT on the base model's own 8-rendering mean posterior | – | – | 1 | label-free "just distil the ensemble" baseline (C4, C5) |
| base model at temperature 0.7, at temperature 0, and its 8-answer ensemble | – | – | – | – | inference-time baselines |

**Mechanism cells.** Collapse showed by step 21 when it happened. Each cell runs at most 60
steps, under the same safeguard, and is compared with CCR's step-50 checkpoint.

### 3.3 Endpoints

The registered endpoints (H2–H7) are all reported as registered. They are not enough, because
**raw consistency can be met by flattening**: RLVR's 36-step model cut raw incoherence by 34%
while normalised incoherence did not move.

**Primary.** DDXPlus confirmatory items, registered decoding (temperature 0.7), with
temperature 0 as a sensitivity analysis.

1. **Nuisance flip rate.** The share of meaning-preserving renderings whose top-1 answer
   differs from the base rendering's. The renderings are retest, three orders, duplication,
   paraphrase and cue. Flattening or sharpening an answer never moves its argmax, so this cannot
   be gamed by hedging.
2. **Normalised incoherence.** The mean JS divergence between a case's base answer and its
   nuisance renderings, divided by the mean JS divergence to other patients' base answers. This
   puts within-patient disagreement on the scale of between-patient disagreement.
3. **Selectivity**, on the `ablate` item (one informative finding removed):
   - **evidence response:** JS(base, ablate) minus JS(base, retest), relative to the untrained
     model's;
   - **direction:** the share of cases whose change in gold probability has the sign that the
     KB-Bayes reference gives.

**Secondary:**

- top-1 and top-3, Brier score with its reliability and resolution parts, ECE;
- ensemble gap (H7);
- authority deference, a perturbation family that is never trained;
- MMLU-Pro and GSM8K subsets (500 + 300 questions), non-inferiority to the base model (the
  registered H3 second part, now implemented);
- training curves of M, entropy and sum-to-100 compliance.

**Transfer:** held-out pathologies (8), MedCaseReasoning (300), post-freeze real cases (Paper 1
§2).

**Change to Amendment 4's H2 guard (A4.4; not yet filed).**

- **The guard as written:** the coherence arm must keep ≥ 90% of RLVR's cross-patient
  information M.
- **Why it fails.** Under the strict readout, RLVR's binary reward favours one-hot answers,
  which always sum to 100. Overconfidence inflates M, so the guard would penalise a better
  calibrated model.
- **The replacement:** H2 counts as supported only if the raw reduction comes with lower values
  of endpoints 1 and 2 than RLVR, both with one-sided 95% intervals. M and entropy are
  reported against the untrained model, with M ≥ m0 checked.

**Statistics:**

- Per-case metrics are averaged over seeds. Arms are compared by paired case bootstrap (2,000
  resamples).
- Each seed is also shown on its own. A result is called consistent only if every seed has the
  same sign.
- Holm correction is applied over the three primary endpoints.
- Non-inferiority margins are as registered: 2 points top-1, 1.5 points general.

### 3.4 Before any GPU run (no cost)

Status 27 Nov 2026: **all done** ([amendment_5.md](amendment_5.md) §B; tests pass on transformers 4.57.6 and 5.22).

| | what | why |
|---|---|---|
| P1 | **Shared-prompt training pass.** Each step trains 160 sequences but has only 48 distinct prompts, and prompts are about 70% of the tokens. Each prompt is computed once and shared by its 2–4 rollouts. The result is identical: a CPU test checks that per-token log-probs and LoRA gradients match the unshared pass. Gradient checkpointing is turned off (the H100 has the memory) | halves the training part of the step (planned 40–50 s instead of 78 s). This is the difference between 1 and 3 seeds |
| P2 | **Disjoint perturbation banks.** Training today uses the same 230 paraphrase templates and 40 irrelevant sentences as the test items. Training now uses the next verified paraphrase from the archived outputs (213 of 230 differ from the test bank; the other 17 keep the base wording; no Bedrock call) and 40 new irrelevant sentences (linted with the panel's blacklist); the test items stay frozen | otherwise the DDXPlus paraphrase and cue scores measure memorised wording |
| P3 | Configs `brier_only`, the two mechanism cells, `sft_selfcons` (new SFT target) | §3.2 |
| P4 | Keep the adapters at steps 50, 100, …, 300. Evaluate each on the 30 pilot cases (≈ 1 min) | training curves; the go/no-go check |
| P5 | Endpoints 1–3, the Brier decomposition, seed pooling, the revised guard, and the MMLU-Pro/GSM8K subset evaluation in `ccrl train hypotheses` | §3.3 |
| P6 | **Audit predictions.** The estimator audit is extended with the readout channel. Its prediction for every arm (collapse or not; direction of entropy, M and nuisance divergence) is committed here before any run | makes C7 a real test |
| P7 | One job queue per machine, a budget guard that stops at the cap, then copy back and terminate | the $100 is a hard limit |
| P8 | The Bedrock caps of §2 and the dated record ([amendment_5.md](amendment_5.md)) | §2 |

**What the audit predicts** ([audit_predictions.md](audit_predictions.md), committed before any run):

- **The completed GPU collapse is reproduced:** step 22 in the audit, step 21 on the GPU.
- **The credit is the decisive mechanism.** Cell B (registered credit, strict readout) still
  collapses, by stated hedging. Cell A (team credit, lenient readout) does not.
- **CCR and RLVR train without collapse.** CCR is better calibrated, RLVR sharper.
- **Brier-only hedges and is stopped by the safeguard** at about step 23 (about $2 instead of $16).
- **The audit does not predict that CCR beats RLVR on presentation divergence.** In the
  simulation both reduce it equally, because there presentation sensitivity costs accuracy. A GPU
  difference would have to come from presentation effects that do not cost accuracy. This is the
  main scientific risk for C2.

### 3.5 GPU queue and what fits

**Basis.** All measured on 27 Nov 2026:

- **CCR step:** 78 s on the H100 (12 s generation, 66 s training).
- **RLVR step:** 36 s on the A100, which trains only the 4.7 of 16 groups with reward spread.
- **Evaluation of one model:** 12,000 generations in about 10 minutes.
- **Prices:** H100 $3.79/h, A100 40 GB $1.99/h; check `lambda_api.py types` before launch.
- **After P1:** the planned CCR step is 40–50 s; P1's GPU check measures it. Each line includes
  its evaluation.

| # | run | GPU | cost (40 s / 50 s per step) | running total (40 s / 50 s) |
|---|---|---|---|---|
| 0 | setup, equivalence and timing check, untrained-model evaluations | both | $5.7 | $5.7 / $5.7 |
| 1 | CCR seed 0, **go/no-go at step 50** | H100 | $13.8 / $16.9 | $19.5 / $22.6 |
| 2 | RLVR seed 0 | A100 | $6.0 | $25.5 / $28.6 |
| 3 | Brier-only seed 0 | H100 | $13.8 / $16.9 | $39.3 / $45.5 |
| 4 | mechanism cells A and B | H100 | $5.8 / $7.0 | $45.1 / $52.5 |
| 5 | BCT and self-consistency distillation | A100 | $3.5 | $48.6 / $56.0 |
| 6 | CCR + RLVR seed 1 | both | $19.8 / $22.9 | $68.4 / $78.9 |
| R | reserve: later post-freeze evaluation of every adapter (~$4) and overrun | | | |

Runs 0–6 are **stage A**, which the $82 pays for.

- **Go/no-go after run 1, step 50** (about $9 spent). Continue only if all of these hold:
  - sum-to-100 compliance is at least 0.8;
  - the safeguard has not fired;
  - on the pilot cases, entropy is at most 0.5 nats above the untrained model's;
  - the nuisance flip rate is not above the untrained model's + 0.05 (about one standard error
    on 30 pilot cases).

  Otherwise the queue stops, the rest of the money is kept, and the failure is diagnosed on the
  CPU first.
- **The cut line is set from the step time measured in run 0.** The forecast must stay at or
  below $78, keeping $4 in reserve:
  - at 40 s per step, all of stage A fits ($68);
  - at 50 s, the mechanism cells are cut to 40 steps ($77).
- **Without P1** (78 s per step), stage A does not fit: one seed of each arm, plus Brier-only and
  the mechanism cells, is the most it buys.

**Stage B, funded only if stage A works.** "Works" is fixed now, before any run:

- the go/no-go check passed;
- in **both** seeds, CCR has a lower nuisance flip rate and lower normalised incoherence than
  RLVR;
- top-1 is non-inferior to RLVR.

If stage A fails, the paper is the diagnosis (C1, C7) with a negative C2, and no more GPU money
is asked for. If stage A works, these are added, in order of what each adds to the paper:

| order | runs | cost | what it adds |
|---|---|---|---|
| B0 | real-case pair `ccr_mcr` + `rlvr_mcr` (2,000 labelled MedCaseReasoning training cases), if stage A's budget guard did not already run it | ~$22 | answers "trained only on synthetic data" (Amendment 6 §3) |
| B1 | CCR + RLVR seed 2 | ~$21–23 | three seeds, the usual minimum for a training claim |
| B2 | Qwen3-8B, CCR + RLVR, 2 seeds (measures its own τ_I first) | ~$90 | scale; H6 (a trained 8B model against the untrained 32B and the panel) |
| B3 | a second model family at about 4B, CCR + RLVR, 1 seed | ~$40 | the result is not specific to Qwen |
| B4 | Brier-only seeds 1–2 | ~$30 | C3 on three seeds |
| B5 | no-fidelity and no-invariance ablations, 1 seed each | ~$30 | which term does what |
| B6 | label-free arm on unlabelled real case reports (H5) | ~$15 | the label-free claim on real data |

Anything not run is reported as not run, with the reason.

---

## 4. Order of work

| when | who | what |
|---|---|---|
| 27 Nov | training side | P1–P8 done; audit predictions committed |
| early Dec | training side | stage A, GPU runs 0–6 (about 1 day of wall-clock on two machines) |
| after stage A | team | decide stage B on the rule in §3.5 |
| by 18 Dec | Bedrock runner | jobs 2–7 |
| Dec–Jan | clinicians | Paper 1 ratings |
| after the post-freeze items exist | training side | evaluate every adapter on them (reserve) |
| late Jan 2027 | team | Paper 1 preprint |
| by 31 Mar 2027 | team | Paper 2 submission |
