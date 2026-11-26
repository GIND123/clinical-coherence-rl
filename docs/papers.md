# Two papers from one study

*18 Nov 2026, after the main panel; updated 25 Nov 2026 after the secondary arms. This replaces
the single-paper framing of `problem_statement.md` §3: the grant needs two papers. Results:
[results_panel_main.md](results_panel_main.md) and [results_secondary.md](results_secondary.md).
All Bedrock runs are finished ($560 of the $1,100 credit).*

One Bedrock run and one item set feed both papers:

- **Paper 1** is a clinical evaluation of open-weight models as
  diagnosticians, built mainly on the Bedrock panel.
- **Paper 2** is a method paper: a label-free training signal, its theory,
  and GPU training of small models evaluated on the same frozen items.

They cite each other but make disjoint claims. Paper 1 is preprinted first.

---

## Paper 1 — clinical evaluation

**Working title.** *Same patient, same question, different diagnosis:
run-to-run instability and hidden presentation sensitivity in open-weight
language models as diagnosticians.* (The earlier "instability, not
presentation" no longer holds: at temperature 0, presentation effects appear;
docs/results_secondary.md §3.)

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
| F3 | At provider defaults, order, duplication, paraphrase and irrelevant sentences add little beyond that noise (≤ 0.09 of a different patient; most CIs include 0) | panel_main; panel_ext (300 new cases) | done; replicates |
| F4 | The noise costs accuracy: averaging equivalent answers gains up to 12.8 points top-1; 4–24% of Brier error is variance | panel_main | done |
| F5 | "Ask twice" is a usable alarm. When two identical queries disagree, the first answer is wrong 55–86% of the time (vs 12–72% when they agree), and disagreement catches 17–71% of errors. For the noisier models it beats stated confidence (AUROC +0.20) | panel_main | done |
| F6 | Even with the noise averaged out (5 vs 4 answers per case), removing an informative finding moves the gold diagnosis detectably in only 11–24% of cases, in the right direction in 49–60%, uncorrelated with the Bayes reference. Reversing the order of two evidence halves changes the final top diagnosis in 8–44% of cases | resample; sequential | done |
| F7 | Temperature 0 cuts the flip rate (DeepSeek V3.2 49% → 5%) but is not deterministic (0–41% byte-identical responses), restores DeepSeek V3.2's lost accuracy (+19 points), and **unmasks systematic order and paraphrase effects** hidden by the noise (Qwen3-32B, DeepSeek V3.2, GLM-5). Reasoning effort does not stabilise the top answer | temp0; effort_arm | done |
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

## Bedrock: finished

Every arm ran on 24–25 Nov 2026 (docs/results_secondary.md), except `temp0` on Kimi K3, which
Bedrock does not allow at temperature 0. Total reported spend: $560.02. About $540 of credit
remains until 31 Dec 2026, with paid calls possible until 18 Dec. It is not needed for either
paper; it is held for a reviewer-requested arm, should one come up in time.

## Timeline

| when | who | what |
|---|---|---|
| done | PI | all Bedrock runs; frozen inputs pushed; archive verified |
| now | training side | base-model evaluation on the frozen items; the decisive 4B pair (`rlvr` vs `ccr_acc_fid`) |
| 26 Nov 2026 | team | **done:** Paper 2 training hypotheses registered on OSF before training (https://osf.io/4xmq7) |
| Dec–Jan | clinicians | ratings (a)–(c) |
| before 18 Dec | PI | Zenodo deposit of the archive |
| late Jan 2027 | team | Paper 1 preprint and submission |
| by 31 Mar 2027 | team | Paper 2 submission (TNNLS) |
