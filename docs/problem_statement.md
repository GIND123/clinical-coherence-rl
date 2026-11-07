# Problem statement

**Coherence as a Reward: Label-Free Reinforcement Learning for Rational
Belief Revision in Open-Weight Clinical Reasoning Models**

*Version 2, 6 Nov 2026. Supersedes the planning document's closed-model
design (see "What changed and why" at the end).*

## 1. The problem

A diagnostic language model that names the right diagnosis can still hold
beliefs that are not beliefs. Put the same findings in a different order,
reword them, repeat one, or add an irrelevant sentence, and the stated
differential moves. Remove a finding that matters and it often doesn't move,
or moves the wrong way. In our previous study (eleven open-weight models,
DDXPlus) every model showed order effects well above its own retest noise.
Belief-update direction agreed with the corpus only at chance (0.45–0.55).
The named diagnosis beat the stated posterior's argmax by 0.06–0.43 top-1.

For clinical use this matters more than a few points of accuracy. A
differential that depends on the order a clinician typed the history in
cannot be audited, combined with other evidence, or trusted as a probability.

The models this applies to most are **open-weight models**. They are the ones
a hospital can run on its own hardware without sending patient data to a
vendor, and the only ones whose behaviour can be pinned, inspected and
retrained. They are also the models a research group can fix.

## 2. The question

> Can the principles of rational belief revision be turned into a training
> signal that needs **no diagnosis labels**, and does training a small
> open-weight model on that signal make its differential diagnoses behave
> like coherent beliefs, without losing accuracy and better than simply
> using a much larger model?

The principles are:

- **Invariance:** the posterior is unchanged by meaning-preserving
  re-presentation (order, duplication, paraphrase, irrelevant cues).
- **Sensitivity:** the posterior moves when evidence changes.
- **Fidelity:** where a likelihood is known, the update has the right
  direction and size.

## 3. Contributions (one paper)

1. **Measurement.** A pre-registered panel of six open-weight models served
   on Amazon Bedrock: Kimi K3, GLM-5, DeepSeek V3.2, Qwen3-Next-80B-A3B,
   gpt-oss-120b and Qwen3-32B.
   - It covers 600 confirmatory cases (300 DDXPlus, 300 real published case
     reports from MedCaseReasoning), each presented 10 ways.
   - It adds a 300-case DDXPlus replication sample, a reasoning-effort
     dose-response arm and a two-stage sequential-evidence arm.
   - Every effect is normalised between the model's own retest noise (0) and
     a different patient (1), so models with different sampling noise are
     comparable.
2. **Theory.** Four propositions (`docs/theory.md`):
   - invariance implies order, duplication and commutativity guarantees;
   - label-free coherence cannot identify the correct posterior
     (non-identifiability);
   - adding update fidelity restores identification;
   - the batch-level sensitivity gate and the cross-patient information gate
     rule out collapse to a constant answer.
3. **Method.** A coherence-gated reward (invariance × sensitivity gate ×
   information gate, plus optional fidelity and accuracy terms) trained with
   paired-group GRPO on Qwen3-4B and Qwen3-8B (LoRA). It is compared with:
   - accuracy-only RLVR, RLCR and BCT;
   - SFT on the Bayes reference and SFT on a teacher (Kimi K3 answers);
   - a logit read-out;
   - inference-time prompting.
4. **Scale comparison.** Trained models are evaluated on **the same frozen
   items, with the same parser and estimands** as the Bedrock panel. One table
   then answers whether a coherence-trained 4B/8B model is more coherent than
   open-weight models many times its size, including its own family
   (Qwen3-32B, Qwen3-Next-80B).

## 4. Research questions and hypotheses

The confirmatory hypotheses are registered on OSF before the main panel
(`docs/preregistration.md`).

| | question | confirmatory hypothesis |
|---|---|---|
| RQ1 | Are open-weight models' differentials invariant to meaning-preserving re-presentation and sensitive to evidence? | **H1** every panel model's order effect (NOE) is above its retest floor (95% CI lower bound > 0) on both datasets; authority exchange rate ≠ 1 |
| RQ2 | Does the coherence reward improve invariance beyond accuracy-only RL, without losing accuracy? | **H2** CCR+acc+fid reduces free-text NOE vs RLVR by ≥ 25% (Holm over order, duplication, paraphrase, cue); **H3** top-1 non-inferior to RLVR (Δ = 2 points), MMLU-Pro/GSM8K non-inferior to the base model (Δ = 1.5) |
| RQ3 | Do the gains transfer? | **H4** on the 8 held-out pathologies and on MedCaseReasoning, the NOE improvement keeps ≥ 50% of the in-distribution improvement |
| RQ4 | Does training without labels on real case reports work? | **H5** label-free training on unlabeled MedCaseReasoning reduces NOE on held-out real cases vs the base model; top-3 non-inferior (Δ = 3) |
| RQ5 | Is a coherence-trained small model more coherent than large untrained ones? | **H6** the CCR-trained Qwen3-8B has lower free-text NOE than untrained Qwen3-32B and than the panel median (one-sided, case-clustered bootstrap) |

Secondary and pre-registered, but not confirmatory:

- fidelity raises DDXPlus update-direction agreement above chance; this is
  *not* predicted for the label-free arm, by Proposition 2;
- the effort dose-response in gpt-oss-120b;
- the sequential commutativity gap;
- replication of H1 on the 300-case replication sample.

## 5. Why the result is informative either way

Every outcome is a finding, because the design is pre-registered and every
effect is measured against a noise floor:

- **H2–H6 hold:** a label-free method makes small deployable models coherent,
  with a proof of what it cannot do.
- **H2 fails while H1 holds:** the incoherence is real and resists the most
  direct fix. The theory says why (Proposition 2), and the ablations say which
  term failed.
- **H6 fails:** scale, not training, buys coherence, which is a deployment
  recommendation in itself.

## 6. Scope and non-claims

- This is not a clinical deployment or decision-support study (DECIDE-AI
  stage 0). No patient data beyond two public, licensed datasets (DDXPlus,
  CC BY 4.0; MedCaseReasoning, MIT). No credentialed data.
- Coherence is not accuracy. The paper reports both, and Proposition 2 is
  stated as a limitation of any label-free signal, ours included.
- Closed partner models (Claude, GPT-6, Grok) are not part of the
  confirmatory study. If AWS later enables them on the account, they run as
  a separately labelled exploratory arm (`partner_panel`).

## 7. What changed and why

The planning document assumed eleven hosted models, five of them closed.
The PI's account cannot call partner models: AccessDeniedException "not
available for this account" (6 Nov 2026), apparently because the credit grant
excludes third-party providers. The six open-weight models served by Bedrock
itself all probe OK.

| | plan v1 | this version |
|---|---|---|
| panel | 11 models (5 closed, 6 open) | 6 open-weight models |
| confirmatory cases | 300 DDXPlus + 150 MedCaseReasoning | 300 + **300** (more real cases) |
| extension | 300 DDXPlus, optional (C) | 300 DDXPlus **replication sample**, pre-registered (B) |
| effort arm | Opus 5.5, GPT-6 Sol, medium/high | gpt-oss-120b, low/medium/high (dose-response) |
| ceiling arm | GPT-6 Astra | dropped |
| distillation teacher | GPT-6 Sol | Kimi K3 |
| preprocessing | GPT-6 Sol, verified by Sonnet 5.5 | Kimi K3, verified by GLM-5 (different vendor) |
| framing | frontier models in general | open-weight, locally deployable models; trained model compared with its own family at scale |
| expected Bedrock spend | ~$500 | ~$150–250 (`docs/budget.md`) |

The change makes the paper tighter rather than weaker. Every model in it
(panel and trained) is open-weight and reproducible. The comparison that
matters for deployment, a small trained model against large open ones, is now
in the paper. The money saved buys more real cases and a replication sample.
