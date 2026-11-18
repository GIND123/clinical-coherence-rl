# Two papers from one study

*18 Nov 2026, after the main panel. This replaces the single-paper framing
of `problem_statement.md` §3: the grant needs two papers. Results so far:
[results_panel_main.md](results_panel_main.md).*

One Bedrock run and one item set feed both papers:

- **Paper 1** is a clinical evaluation of open-weight models as
  diagnosticians, built mainly on the Bedrock panel.
- **Paper 2** is a method paper: a label-free training signal, its theory,
  and GPU training of small models evaluated on the same frozen items.

They cite each other but make disjoint claims. Paper 1 is preprinted first.

---

## Paper 1 — clinical evaluation

**Working title.** *Same patient, same question, different diagnosis:
run-to-run instability, not presentation, limits open-weight language models
as diagnosticians.*

**Targets.** npj Digital Medicine; JAMIA; Lancet Digital Health. Backup: JBI,
JAMIA Open.

**The question.** Open-weight models are the ones a hospital can run on its
own hardware with patient data. When such a model states a differential
diagnosis, is that differential a stable, evidence-driven belief?

### Claims and their evidence

| # | claim | evidence | status |
|---|---|---|---|
| F1 | Accuracy on synthetic and real cases spans 51–84% and 24–61% top-1; calibration is poor (ECE 0.07–0.53) | panel_main | done |
| F2 | Asked the identical question twice, models change their leading diagnosis in 9–43% of cases | panel_main (retest) | done |
| F3 | Beyond that noise, order, duplication, paraphrase and irrelevant sentences add little (≤ 0.09 of a different patient; most CIs include 0) | panel_main; replicated on 300 new DDXPlus cases | panel_ext pending |
| F4 | The noise costs accuracy: averaging equivalent answers gains up to 12.8 points top-1; 4–24% of Brier error is variance | panel_main | done |
| F5 | "Ask twice" is a usable alarm. When two identical queries disagree, the first answer is wrong 55–86% of the time (vs 12–72% when they agree), and disagreement catches 17–71% of errors. For the noisier models it beats stated confidence (AUROC +0.20) | panel_main | done |
| F6 | Do models update on evidence once the noise is averaged out? Direction and size of the mean update after removing one finding, against the tempered-Bayes reference; half-evidence vs full-evidence posteriors | resample; sequential | pending |
| F7 | Does deterministic decoding remove the instability, and does it reveal systematic presentation effects hidden by noise? Does more reasoning effort reduce instability? | temp0; effort_arm | pending |
| F8 | Deference to a wrong attending suggestion: most models under-weight it relative to Bayes, while Qwen3-32B over-weights it (top-1 51% → 22%) | panel_main | done |

### Clinician component (what lifts it to a clinical journal)

Two clinicians, blinded to the model, rate:

- **(a)** the paraphrase and irrelevant-sentence banks: the review CSVs
  already produced by preprocessing;
- **(b)** 100 sampled retest flips: *"Would this change in the leading
  diagnosis change the next management step?"* (yes / no / unsure);
- **(c)** 50 cases where the averaged answer is right and the single answer
  wrong.

This turns "the top diagnosis changes" into "the clinically relevant decision
changes", and gives inter-rater agreement. The cases are public and synthetic
or published (no patient data). Confirm with the IRB whether this counts as
human-subjects research.

### Figures

1. Retest flip rate by model and dataset, with accuracy.
2. Decomposition of total incoherence into noise (retest) and systematic
   effects (NOE, paraphrase, duplication, cue).
3. Ensemble gain and variance share.
4. ROC: disagreement vs stated confidence for error detection; the
   two-query rule's operating points.
5. Mean-posterior evidence updates vs the Bayes reference (resample).
6. Effort and temperature arms.

### Reporting

TRIPOD-LLM; DECIDE-AI stage 0 statement; data and model cards; Zenodo
archive of every request and response.

### Not claimed

- That instability is unavoidable: decoding settings are part of the
  question (F7).
- That a model "ignores evidence": that waits for F6.
- Any statement about closed models: they are not in the panel.

---

## Paper 2 — method

**Working title.** *Coherence as a reward: label-free reinforcement learning
that makes small clinical models stable and evidence-sensitive.*

**Targets.** IEEE TNNLS (submission by 31 Mar 2027); NeurIPS 2027 (May).
Backup: ML4H, JBI.

**Motivation, from Paper 1.**

- The incoherence that matters is answer-to-answer variance.
- Ensembling removes it, but at K times the inference cost.
- Proposition 5 (docs/theory.md) shows the variance is exactly the Brier
  excess of a single answer over the model's own mean belief.

The coherence reward's invariance term penalises the divergence between
rollouts across meaning-preserving inputs, so it penalises exactly that
variance. The information and sensitivity gates rule out the degenerate fix:
a constant or evidence-blind answer (Proposition 4).

**The claim to test.** Coherence training captures the ensemble's gain in a
single answer, without labels. Proposition 2 limits what it can do: it
cannot move the mean belief toward the truth, which is what the fidelity and
accuracy terms are for.

### Hypotheses

These are amended before any training run (docs/osf_amendment_1.md).

| | hypothesis | change |
|---|---|---|
| H2 | CCR+acc+fid reduces **total** incoherence vs RLVR by ≥ 25%. Total incoherence is the mean JS between meaning-preserving answers, retest included. NOE becomes secondary | primary endpoint now includes the noise, which NOE subtracts out |
| H3 | top-1 non-inferior to RLVR (Δ 2 points); MMLU-Pro / GSM8K non-inferior to base (Δ 1.5) | unchanged |
| H4 | gains transfer to the 8 held-out pathologies and to real cases (≥ 50% retained) | unchanged |
| H5 | label-free training on unlabeled real case reports reduces total incoherence vs base; top-3 non-inferior | endpoint as in H2 |
| H6 | the CCR-trained Qwen3-8B, in a single answer, has lower total incoherence and a lower retest-flip rate than untrained Qwen3-32B and the panel median | was NOE, which is ~0 for most panel models and would make H6 untestable |
| H7 (new) | CCR closes ≥ 50% of the base model's ensemble gap (single vs 8-answer mean, top-1 and Brier) | new |

### Baselines

The existing ones: RLVR, RLCR, SFT on the Bayes reference, SFT on Kimi K3
answers (`distill`), BCT and logit read-out.

New, because Paper 1 makes them the obvious competitors:

- **Inference-time:** base model at temperature 0, and the base model's
  8-answer ensemble. These are the "why train at all?" baselines.
- **Training:** self-consistency distillation, i.e. SFT on the base model's
  own 8-answer mean (label-free, like CCR).

### Compute

About 40 LoRA runs on Qwen3-4B / 8B, at 10–14 GPU-hours each with the
current transformers loop (docs/training.md). The decisive pair first:
`rlvr` vs `ccr_acc_fid`, 4B, seed 0.

---

## What runs on Bedrock next (all within the credit)

At measured per-call costs:

| job | for | calls | expected cost |
|---|---|---|---|
| effort_arm | P1 F7 | 2,700 | ~$2 |
| sequential | P1 F6 | 3,600 | ~$20 |
| temp0 *(amendment 1)* | P1 F7, P2 baseline | 5,400 | ~$30 |
| resample *(amendment 1)* | P1 F6 | 10,800 | ~$50 |
| panel_ext | P1 F3 replication | 18,000 | ~$84 |
| distill | P2 baseline | 2,000 | ~$51 |
| MedCaseReasoning training extraction | P2 H5 | 2,150 | ~$140 |
| **total** | | **44,650** | **~$377** |

That brings total real spend to about $617 of the $1,100 credit.

## Timeline

| when | who | what |
|---|---|---|
| this week | PI | `git pull`; confirm Flex billing; file OSF amendment 1; re-freeze; run the jobs above (~2 days of Kimi time); push `artifacts/frozen/` and `models.lock.yaml` |
| from now | training side | base-model evaluation on the frozen items as soon as they are pushed; the decisive 4B pair |
| by 18 Dec | PI | archive (`ccrl export`, `ccrl archive pack`) to Zenodo |
| Dec–Jan | clinicians | ratings (a)–(c) |
| late Jan 2027 | team | Paper 1 preprint and submission |
| by 31 Mar 2027 | team | Paper 2 submission (TNNLS) |
