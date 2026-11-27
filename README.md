# Coherence as a Reward

**Label-free reinforcement learning for rational belief revision in
open-weight clinical reasoning models.**

Clinical language models can name the right diagnosis yet state
probabilities that do not behave like beliefs. Reordering the same findings,
rewording them, or adding an irrelevant sentence moves their differential,
while removing real evidence often does not move it in the right direction.

This repository turns the principles of rational belief revision into a
training signal: a coherence-gated reward over full diagnostic posteriors
that needs no diagnosis labels. It proves what such a signal can and cannot
guarantee. It then tests whether a small open-weight model trained on it is
more coherent than much larger ones.

**Problem statement, research questions and hypotheses:
[docs/problem_statement.md](docs/problem_statement.md).**

## One paper, two parts that share one item set, one parser and one set of estimands

| | Part 1 — measurement panel | Part 2 — training |
|---|---|---|
| what | 6 open-weight models on Amazon Bedrock (Kimi K3, GLM-5, DeepSeek V3.2, Qwen3-Next-80B, gpt-oss-120b, Qwen3-32B), run **once**, archived | Qwen3-4B / 8B with LoRA: coherence-gated GRPO vs RLVR, RLCR, BCT, SFT, logit read-out |
| data | 600 confirmatory cases (300 DDXPlus + 300 real case reports) × 10 presentations; 300-case replication sample; effort and sequential arms | DDXPlus training split (8 pathologies held out); unlabeled MedCaseReasoning cases |
| joined by | the same frozen items: trained models enter the panel's tables (H6: trained 8B vs Qwen3-32B and the panel median) | |
| runs on | the PI's machine, PI's AWS credentials: **finished** 25 Nov 2026 ($560 of $1,100) | a GPU workstation, starting now |
| start here | **[RUNBOOK.md](RUNBOOK.md)** | **[docs/training.md](docs/training.md)** |

## Status (25 Nov 2026)

| component | state |
|---|---|
| **All Bedrock runs**: pilot, main panel (600 cases × 10 calls × 6 models), replication (300 new cases), `temp0`, `resample`, `sequential`, `effort_arm`, `distill`, training extraction | **done** on the PI's account ($560.02 of $1,100); archive verified; results in [docs/results_panel_main.md](docs/results_panel_main.md) and [docs/results_secondary.md](docs/results_secondary.md) |
| Plan | **two papers** ([docs/papers.md](docs/papers.md)): a clinical evaluation of open-weight models, and the coherence-reward method |
| Amendment 2 (Paper 1): temperature 0 on all 900 cases, contamination probe | specified and frozen ([docs/amendment_2.md](docs/amendment_2.md)); **to run on the PI's account before 18 Dec** (RUNBOOK §8b, ≈ $66) |
| Amendment 3 (Paper 1): 150 real case reports published after the models were frozen (contamination-free) | specified, cases selected and frozen ([docs/amendment_3.md](docs/amendment_3.md)); **to run before 18 Dec** (RUNBOOK §8c, ≈ $76–90) |
| Frozen inputs (`artifacts/frozen/`, `models.lock.yaml`) and 2,000 MedCaseReasoning training cases | in the repository |
| OSF | **training study (Paper 2) registered 26 Nov 2026, before any training: [osf.io/4xmq7](https://osf.io/4xmq7)**. The Bedrock panel was not registered before collection (stated in [docs/preregistration.md](docs/preregistration.md)) |
| Reward and synthetic-policy suite (Propositions 2 and 4) | done; 10/10 criteria pass |
| Trainer, baselines, local evaluation, trained-vs-panel comparison | code done; CPU smoke-tested; GPU runs pending |
| Tests | `python -m pytest -q` (CPU, ~3 min) |

**Headline.**

- **The answer is unstable.** At provider-default decoding, the leading diagnosis changes in
  9–48% of cases when the *identical* question is asked twice. This replicates on 300 new
  cases.
- **Averaging helps.** Averaging equivalent answers raises top-1 by up to 14 points, and
  disagreement between answers flags errors better than stated confidence for the noisier
  models.
- **Temperature 0 helps, but it is not deterministic.** The noise falls, and DeepSeek V3.2
  regains 19 points of accuracy.
- **Without the noise, presentation sensitivity appears.** Re-ordering and rewording the case
  then change the answer.
- **The evidence barely moves the answer.** Averaged over the noise, removing an informative
  finding moves the answer in the right direction in only 49–60% of cases.

## Quick start

```bash
bash scripts/setup.sh && source .venv/bin/activate
ccrl doctor          # environment, config, AWS identity, model lock (no cost)
ccrl rehearse        # the entire panel pipeline on an offline mock, ~10 min, $0
```

Then follow [RUNBOOK.md](RUNBOOK.md) step by step.

## What the design gets right

* **Every paid call is accounted for before it is sent.** A worst-case
  reservation is checked under one lock against three limits:
  - the hard cap ($1,000 of the $1,100 credit);
  - the job cap;
  - the per-model cap.

  Timeouts that may have been billed stay counted. Nothing is sent after
  18 Dec 2026. A request's ID is a hash of its exact text, so re-running
  never re-sends completed work.
* **The design cannot drift.** Every input that defines a prompt is hashed
  (`ccrl freeze`), and paid jobs refuse to run if anything changed. The model
  lock records a hash of each model's settings, so a stale probe is caught
  before money is spent.
* **Requests are validated before money is spent**, in two ways:
  - tests check them against the installed botocore service model;
  - a live per-model probe records which effort encoding, service tier,
    cache point and optional fields each model accepts
    (`config/models.lock.yaml`).
* **Noise-normalised estimands.** Every effect is scaled between the model's
  own retest noise (0) and a different patient (1). Hosted models and locally
  trained models with different sampling noise are therefore directly
  comparable.
* **Two findings shaped the reward** ([docs/theory.md](docs/theory.md)):
  - On DDXPlus the reference posterior is near-deterministic, so
    single-finding sensitivity is degenerate (median ablation JS = 0). The
    contrast is therefore the triage rendering.
  - A per-rollout sensitivity term prefers an over-reacting oracle to the
    Bayes oracle, so sensitivity is a batch-level anti-collapse gate.

## Repository map

| path | contents |
|---|---|
| `config/` | study design, model registry, prices, budget caps, jobs, cue bank, prompt templates, training configs |
| `artifacts/frozen/` | committed inputs (DDXPlus, MedCaseReasoning selection) and, after the paid preprocessing, the phrase bank, prepared cases, items and `FREEZE.json` |
| `src/ccrl/data/` | DDXPlus parsing, finding units, renderer, corpus likelihoods; MedCaseReasoning selection; downloads |
| `src/ccrl/prompts/` | prompt assembly, robust posterior parser |
| `src/ccrl/panel/` | prepared cases, the 10 transformations, jobs, forecast, freeze, pilot QA |
| `src/ccrl/bedrock/` | Converse client, mock, ledger, runner, model resolution and probing, pricing |
| `src/ccrl/prep/` | paid preprocessing (phrase bank, MedCaseReasoning) |
| `src/ccrl/analysis/` | metrics, estimands, statistics, report |
| `src/ccrl/reward/` | the coherence-gated reward and the synthetic-policy suite |
| `src/ccrl/train/` | paired-group GRPO, SFT/BCT, logit read-out, local evaluation, trained-vs-panel comparison |
| `docs/` | [two-paper plan](docs/papers.md), [main-panel results](docs/results_panel_main.md), [secondary-arm results](docs/results_secondary.md), [OSF amendment 1](docs/osf_amendment_1.md), [problem statement](docs/problem_statement.md), [design](docs/design.md), [theory](docs/theory.md), [literature review](docs/literature_review.md), [pre-registration](docs/preregistration.md), [budget](docs/budget.md), [training](docs/training.md), [data cards](docs/data_cards.md) |
| `tests/` | CPU tests |

## Data and licences

DDXPlus (CC BY 4.0) and MedCaseReasoning (MIT); see
[docs/data_cards.md](docs/data_cards.md). No credentialed clinical data
(MIMIC/PhysioNet) is used in, or may be added to, this repository. Code is
MIT-licensed.

## Lineage

This builds on the team's previous study, *Probabilistic Coherence as a Test
of Clinical Reasoning in Language Models*
([repo](https://github.com/GIND123/Probabilistic-Coherence-as-a-Test-of-Clinical-Reasoning-in-Language-Models)),
which measured eleven open-weight models. Its instrument-calibration lessons
(C1–C11) are built into the parser and the item design. Training utilities
are adapted from the team's DEFER-Dx trainer
([repo](https://github.com/GIND123/Clinical-Agent-Stanford), Apache-2.0).
