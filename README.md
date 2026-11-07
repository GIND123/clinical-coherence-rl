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
| runs on | the PI's machine, PI's AWS credentials, before 18 Dec 2026 (≈$180 expected) | a GPU workstation, starting now |
| start here | **[RUNBOOK.md](RUNBOOK.md)** | **[docs/training.md](docs/training.md)** |

## Status (6 Nov 2026)

| component | state |
|---|---|
| Problem statement, pre-registration draft | rewritten for the open-weight panel |
| DDXPlus inputs (KB, corpus likelihoods, 2,620 selected cases, held-out pathologies) | **built and committed** (md5-checked source) |
| MedCaseReasoning selection (2,586 cases queued for preprocessing) | **built and committed** (sha256-checked source) |
| Bedrock harness: model resolution and probing, budget-guarded idempotent runner, ledger, pilot gates, Cost Explorer reconciliation, export and archive | **done; rehearsed end to end on an offline mock** |
| Model access | 6 panel models probed OK on the PI's account; closed partner models unavailable (optional arm only) |
| Preprocessing (phrase bank, MedCaseReasoning extraction and verification) | code done; **paid run pending** (RUNBOOK §4) |
| Items, freeze, pilot, panel, arms | code done; **paid runs pending** (RUNBOOK §5–8) |
| Reward and synthetic-policy suite (Propositions 2 and 4) | **done; 10/10 criteria pass** |
| Trainer, baselines, local evaluation, trained-vs-panel comparison | code done; CPU smoke-tested; GPU runs pending |
| Tests | `python -m pytest -q` (CPU, ~2 min) |

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
| `docs/` | [problem statement](docs/problem_statement.md), [design](docs/design.md), [theory](docs/theory.md), [literature review](docs/literature_review.md), [pre-registration](docs/preregistration.md), [budget](docs/budget.md), [training](docs/training.md), [data cards](docs/data_cards.md) |
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
