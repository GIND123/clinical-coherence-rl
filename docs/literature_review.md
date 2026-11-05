# Literature review and positioning

Searched 5 November 2026 (alphaXiv index and primary PDFs). Every work in the
tables below was opened and checked against its own text; arXiv IDs are
given so each claim can be re-verified. Works cited in the planning document
that could **not** be verified in this session are listed separately at the
end and must be checked before citation.

## 1. The question in one paragraph

Clinical LLMs can name the right diagnosis yet state probabilities that do not
behave like beliefs: in our previous study (eleven open-weight models,
DDXPlus) the named diagnosis beat the stated posterior's argmax by
0.058–0.427 top-1, belief-update direction agreed with the corpus at
0.45–0.55 (chance), and reordering identical findings moved posteriors well
beyond each model's own retest noise. This project asks whether the
**principles of rational belief revision** — invariance to meaning-preserving
re-presentation, responsiveness to evidence, and (where a likelihood is
known) correct update size — can be turned into a training signal that needs
no diagnosis labels, and what such a signal can and cannot guarantee.

## 2. Measuring probabilistic coherence and Bayesian updating in LLMs

| work | what it shows | relation to us |
|---|---|---|
| Chen, Jörke et al. 2026, *LLMs are not (consistently) Bayesian* (arXiv:2605.06915) | information-processing gap Δ(q) = KL(q‖Bayes(π, ℓ)) with elicited prior/likelihood; near-Bayes in step-wise ("belief propagation") mode, far from it in batch mode; on MediQ, forcing Bayes **hurts** task performance | label-free Bayes-consistency *measure* (evaluation only); their batch-vs-sequential gap motivates our sequential arm; their MediQ result is a warning that coherence ≠ accuracy (our Prop. 2) |
| Kim et al. 2026, *From Drift to Coherence* (arXiv:2606.17832) | martingale property of predictive beliefs fails under resampling; fixed by a seed-answer prompt and a fine-tuned self-consistency loss | label-free coherence *training*, but martingale/resampling coherence on QA — no evidence-order/paraphrase/cue invariance, no differential, no clinical setting |
| Falck et al. 2024, *Is in-context learning Bayesian? A martingale perspective* (arXiv:2406.00793) | martingale violations in ICL | lineage of Kim et al. |
| Zhu & Griffiths 2024, *Incoherent probability judgments in LLMs* (arXiv:2401.16646) | systematic probability-axiom violations | synchronic coherence; ours is diachronic (revision) |
| Paleka et al. 2025, *Consistency checks for LM forecasters* (ICLR 2025) | arbitrage-based consistency metrics; consistency predicts accuracy | forecasting, evaluation |
| Pal et al. 2025, *Incoherent beliefs & inconsistent actions* (arXiv:2511.13240) | stated confidence and actions diverge | our knows-vs-states ratio is the clinical analogue |
| Yamin et al. 2026, *When agents say one thing and do another* (arXiv:2602.06286) | validating elicited beliefs against decisions | belief-validity framing |
| BayesBench (arXiv:2606.30850) | multi-turn belief trajectories vs closed-form Bayes (coin, recommender) incl. a medical-triage style-invariance environment; larger models over-update | evaluation only; we use its public tasks for non-medical transfer |
| *From Prompts to Posteriors* (arXiv:2610.29331); *Graded belief stability* (arXiv:2609.34158); *Diagnosing and improving probabilistic reasoning* (arXiv:2609.38005) | recent evaluations of in-context Bayesian inference, belief stability and decision-theoretic probabilistic reasoning | breadth of the evaluation literature; none trains a clinical differential for invariance |

## 3. Training toward Bayesian or coherent beliefs

| work | method | supervision | relation to us |
|---|---|---|---|
| Qiu et al., *Bayesian teaching* (arXiv:2503.17523; Nat. Commun. 2026) | SFT on a Bayesian assistant's predictions (flight recommendation) | Bayesian teacher | our `sft_kb_bayes` baseline; needs a teacher |
| Tsvilodub et al. 2026 (arXiv:2610.00679) | probes/patching show Bayes-SFT installs Bayes-consistent beliefs "as Bayesian as their beliefs allow"; oracle-label SFT does not; evidence-order invariance confirmed on a shuffled set | Bayesian teacher | evidence that the *signal* matters; supports our fidelity arm |
| **Yamin et al. 2026, *Training LMs to be coherent decision-makers* (arXiv:2610.09164)** | SFT on the model's **own** elicited belief + expected-utility action, across 8 framings and 20 domains; measures "excess belief distortion" above a **sampling-noise floor**; transfer to held-out domains and framings | self (no labels) | **closest new work.** Same belief-preservation idea and noise-floor normalisation, but: SFT not RL; binary events not K-way differentials; framing of utilities, not evidence order/paraphrase/irrelevant cues; no evidence-sensitivity or anti-collapse term; no non-identifiability analysis. Our `bct` baseline is the posterior analogue of their Direct SFT and must be reported head-to-head |
| Andrews 2026, *Revealed Rationality* (arXiv:2608.05015) | position paper: representation theorems (de Finetti, Afriat, Echenique–Saito) give label-free penalties that are zero iff behaviour is rationalisable; "coherence does not restrict which objective rationalizes behavior" | none (no training reported) | theoretical framing of our Prop. 2 |
| Self-CTRL (arXiv:2606.18327) | RL for self-consistency of self-descriptions | self | adjacent consistency-RL |
| Betz & Richardson 2023 (PLOS ONE) | self-training improves probabilistic coherence on synthetic data | self | early precedent |

## 4. Consistency training and label-free RL — and their failure modes

| work | idea | relation |
|---|---|---|
| Chua et al. 2025 (BCT); Irpan, Turner et al. 2025, *Consistency training helps stop sycophancy and jailbreaks* (arXiv:2510.27062) | train the model's response to a cue-wrapped prompt to match its own clean response (BCT, tokens) or activations (ACT) | **closest mechanism**; our `bct` baseline (posterior targets) |
| *Consistency training along the transformer stack* (arXiv:2606.05817) | broadens consistency training targets | related |
| *Consistency training can entrench misalignment* (arXiv:2606.03810) | consistency can lock in bad behaviour | risk we address with fidelity/accuracy arms, direction agreement and non-inferiority tests |
| Co-rewarding (arXiv:2508.00410) | self-supervised RL from agreement across rephrased questions (I) or an EMA teacher (II); avoids the collapse of single-view self-rewards | **closest RL framing**: invariance across views as reward; ours is over full distributions with explicit anti-collapse gates and theory |
| Label-free RLVR collapse: TTRL regimes (arXiv:2611.01307), CoVerRL (arXiv:2603.17775), *How far can unsupervised RLVR scale* (arXiv:2603.08660), *Wrong votes, right gradients* (arXiv:2610.20898), outcome-masked GRPO (arXiv:2608.03119) | majority-vote / self-certainty rewards collapse to consistent-but-wrong answers | motivates Proposition 4 (constant, clustered, one-finding and noise policies get zero coherence reward) |
| *Consistency has a computable blind spot* (arXiv:2608.05675); *When judges agree and are wrong* (arXiv:2610.21128) | invariance cannot detect systematic errors that survive every perturbation | the empirical face of our Proposition 2 |

## 5. Calibration and distributional rewards

| work | relation |
|---|---|
| Damani et al., RLCR, *Beyond binary rewards* (arXiv:2507.16806; ICLR 2026) | correctness + Brier; bounded proper scoring rules give accurate and calibrated models — our `rlcr` baseline and the basis of Prop. 3's accuracy term |
| *Reaching beyond the mode: RL for distributional reasoning* (arXiv:2603.24844) | RL over answer distributions rather than the mode |
| *How proper scoring rules shape LLM forecasting* (arXiv:2608.28482); *Verifiable rewards for calibrated forecasting* (arXiv:2607.00164) | scoring-rule choice as RL reward |
| *Calibrating reasoners with differentiable readouts* (arXiv:2609.34857); *Decoupling reasoning and confidence* (arXiv:2603.09117) | readout-based calibration; relates to our logit-readout baseline |

## 6. Clinical belief updating and diagnostic reasoning

| work | relation |
|---|---|
| Zeng & Zhang 2026 (arXiv:2610.02684) | ICU trajectories (MIMIC-IV, 8 open models 4–32B): conditioning on the model's previous judgment worsens error more often than it helps; asymmetric response to worsening vs improving evidence; prior 10 %→90 % shifts estimates 26.2 points; prompting does not fix it; EVLU (rule-based, no training) trades coverage for reliability — **clinical scoop risk, evaluation only** |
| *Decide, ask, or defer* (arXiv:2610.04542); *Evidence-value misalignment* (arXiv:2609.35627); *From hindsight to foresight* (arXiv:2610.18605); *Auditing evidence use* (arXiv:2607.20848) | evidence sufficiency and evidence use in diagnosis; evaluation-focused |
| *The patient is always right?* (arXiv:2610.34013) | sycophancy to graded patient beliefs — our authority cue is the clinician-side analogue with a stated reliability |
| MTDiag (arXiv:2608.25085); RL for evidence-seeking diagnosis (arXiv:2607.02983); TrustMed-RL (arXiv:2610.04387) | multi-turn and RL-trained diagnostic agents; reward correctness/process, not belief coherence |
| Nori et al. 2025, *Sequential diagnosis with LMs* (arXiv:2506.22405) | NEJM-based sequential diagnosis (not open data) |
| DDXPlus (arXiv:2205.09148); MedCaseReasoning (arXiv:2505.11733) | our two corpora |

## 7. Positioning

What no verified work does, and this project does:

1. **RL over full stated differentials** (49-way or case-specific 10-way),
   rewarding invariance to evidence order within exchangeable blocks,
   re-statement, paraphrase and certified-irrelevant cues — **without labels**.
2. **Anti-collapse by construction** (Prop. 4): constant, clustered,
   chief-complaint-only and noisy policies receive zero coherence reward,
   checked by a synthetic-policy suite before any GPU run.
3. **A non-identifiability result** (Prop. 2) and its numerical
   demonstration, plus the conditions under which an added fidelity term
   identifies the Bayes posterior (Prop. 3); and two design findings
   (docs/theory.md) showing that a per-rollout sensitivity term — the obvious
   design — prefers over-reacting reasoners under redundant evidence.
4. A **frozen, archived frontier panel** (11 hosted models, effort and
   sequential arms) measured with the same estimands as the trained models,
   normalised to each model's retest floor.

What we must not claim: "first coherence training" (Kim et al.; Yamin et al.;
Irpan et al.; Co-rewarding), "first clinical belief-updating evaluation"
(Zeng & Zhang), or that coherence makes a model Bayesian (our own Prop. 2).

**Scoop risk (Nov 2026).** High from groups combining clinical updating
evaluations (Zeng & Zhang) with consistency training (Yamin et al., Kim et
al.). A preprint with the panel, the theory and the 4B results in late
January 2027 remains the right plan; the OSF pre-registration (before the
main panel) timestamps the design.

## 8. Not verified in this session (check before citing)

From the planning document: MAPLE (2603.08987); PROSE (2609.16660);
DistriTTRL (2603.16500); DDRL (2604.21327); Cordero-Encinar (2510.17472);
Li et al. (2610.05292); Ortega (2602.02912); *The Art of Interrogation*
(2606.11918); Imran et al. (2507.17951); Tak et al. (2601.22329); Sheppert
et al. (PubMed 42335861 — PubMed access needs authorisation). The planning
document's attribution of arXiv:2506.19702 to "Kang et al." is a mismatch
(it is a LoRA-tuned DDXPlus pre-diagnosis platform); cite only after checking.
