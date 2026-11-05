# Coherence as a Reward

**Label-free reinforcement learning for rational belief revision in clinical reasoning models.**

Clinical language models can name the right diagnosis yet state probabilities
that do not behave like beliefs: reordering the same findings, rewording them,
or adding an irrelevant sentence moves their differential, while removing
real evidence often does not move it in the right direction. This repository
turns the principles of rational belief revision into a training signal — a
coherence-gated reward over full diagnostic posteriors that needs no diagnosis
labels — and proves what such a signal can and cannot guarantee.

It has two tracks that share one item set, one parser and one set of estimands:

| | Track A — frozen panel | Track B — local training |
|---|---|---|
| what | 11 hosted frontier and open-weight models on Amazon Bedrock, run **once** and archived | Qwen3-4B / 8B with LoRA, paired-group GRPO and baselines |
| data | 300 DDXPlus + 150 MedCaseReasoning cases × 10 calls, plus effort, sequential and ceiling arms | DDXPlus training split (8 pathologies held out); unlabeled MedCaseReasoning cases |
| runs on | the PI's machine, PI's AWS credentials, before 18 Dec 2026 | a GPU workstation, from 19 Dec 2026 |
| start here | **[RUNBOOK.md](RUNBOOK.md)** | **[docs/training.md](docs/training.md)** |

## Status (5 Nov 2026)

| component | state |
|---|---|
| DDXPlus inputs (KB, corpus likelihoods, 2,620 selected cases, held-out pathologies) | **built and committed** (md5-checked source) |
| MedCaseReasoning selection (2,376 cases queued for preprocessing) | **built and committed** (sha256-checked source) |
| Bedrock harness: model resolution + probing, budget-guarded idempotent runner, ledger, pilot QA gates, Cost Explorer reconciliation, export and archive | **done; rehearsed end to end on an offline mock** |
| Preprocessing (phrase bank, MedCaseReasoning extraction + cross-vendor verification) | code done; **paid run pending** (RUNBOOK §4) |
| Items, freeze, pilot, panel, arms | code done; **paid runs pending** (RUNBOOK §5–9) |
| Reward + synthetic-policy suite (Propositions 2 and 4) | **done; 10/10 criteria pass** |
| Track B trainer, SFT/BCT/logit-readout baselines, local evaluation | code done; CPU smoke-tested; GPU runs pending |
| Tests | 41 passing (`python -m pytest -q`, ~1 min, CPU) |

## Quick start

```bash
bash scripts/setup.sh && source .venv/bin/activate
ccrl doctor          # environment, config, AWS identity (no cost)
ccrl rehearse        # the entire Track A pipeline on an offline mock, ~10 min, $0
```

Then follow [RUNBOOK.md](RUNBOOK.md) step by step.

## What the design gets right (and what it found)

* **Every paid call is accounted before it is sent.** A worst-case
  reservation is checked against the hard cap ($1,000 of $1,100 credit), the
  job cap and the per-model cap under one lock; timeouts that may have been
  billed stay counted; nothing is sent after 18 Dec 2026; a request's ID is a
  hash of its exact text, so re-running never re-sends completed work.
* **The design cannot drift.** Every input that defines a prompt is hashed
  (`ccrl freeze`); paid jobs refuse to run if anything changed.
* **Requests are validated before money is spent**: against the installed
  botocore service model (tests), and by a live per-model probe that records
  which effort encoding, service tier, cache point and optional fields each
  model accepts (`config/models.lock.yaml`).
* **Two findings changed the reward** ([docs/theory.md](docs/theory.md)):
  on DDXPlus the reference posterior is near-deterministic, so single-finding
  sensitivity is degenerate (median ablation JS = 0) — the contrast is the
  triage rendering; and a per-rollout sensitivity term prefers an
  over-reacting oracle to the Bayes oracle — sensitivity is a batch-level
  anti-collapse gate.

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
| `src/ccrl/train/` | paired-group GRPO, SFT/BCT, logit readout, local evaluation |
| `docs/` | [design](docs/design.md), [theory](docs/theory.md), [literature review](docs/literature_review.md), [pre-registration](docs/preregistration.md), [budget](docs/budget.md), [training](docs/training.md), [data cards](docs/data_cards.md) |
| `tests/` | 41 CPU tests |

## Data and licences

DDXPlus (CC BY 4.0) and MedCaseReasoning (MIT) — see
[docs/data_cards.md](docs/data_cards.md). No credentialed clinical data
(MIMIC/PhysioNet) is used in, or may be added to, this repository. Code: MIT.

## Lineage

Builds on the team's previous study, *Probabilistic Coherence as a Test of
Clinical Reasoning in Language Models*
([repo](https://github.com/GIND123/Probabilistic-Coherence-as-a-Test-of-Clinical-Reasoning-in-Language-Models)),
whose instrument-calibration lessons (C1–C11) are built into the parser and
the item design. Training utilities are adapted from the team's DEFER-Dx
trainer ([repo](https://github.com/GIND123/Clinical-Agent-Stanford), Apache-2.0).
