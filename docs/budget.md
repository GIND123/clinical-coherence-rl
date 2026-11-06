# Budget

Credit $1,100 (expires end of December 2026); harness hard cap $1,000;
no paid call after 18 Dec 2026 23:59.

## Calls per job (design in `config/jobs.yaml`)

| job | priority | design | calls |
|---|---|---|---|
| probe | A | ~5 tiny calls × 12 models | ~60 |
| prep_phrasebank | A | 10 batches generate + 10 verify | 20 |
| prep_mcr_extract | A | 226 pilot/panel + 2,150 training cases | 2,376 |
| prep_mcr_verify | A | 226 pilot/panel | 226 |
| pilot | A | 30 cases × (10 calls + 1 schema variant) × 11 models | 3,630 |
| panel_main | A | 450 cases × 10 × 11 | 49,500 |
| effort_arm | B | 150 × 6 × 2 efforts × 2 models | 3,600 |
| sequential | B | 150 × 2 orders × 2 turns × 7 models | 4,200 |
| distill | B | 2,000 × GPT-6 Sol | 2,000 |
| astra_ceiling | C1 | 150 × 6 × GPT-6 Astra | 900 |
| panel_ext | C2 | 300 × 10 × 4 models | 12,000 |

## Pre-pilot forecast (USD)

Computed by `ccrl plan` from the exact prompt texts (conservative ~3
characters per token) and the plan's output-token assumptions (low effort
500/800/1,500; provider default 600/1,200/2,500; medium 1,000/1,600/2,500;
high 2,000/3,000/5,000) plus 5/8/12 % retries. Flex discount applied to
low/expected only.

| | low | expected | high |
|---|---|---|---|
| pilot | ~9 | ~15 | ~33 |
| panel_main | 155 | 244 | 576 |
| effort_arm | 92 | 140 | 232 |
| sequential | 21 | 33 | 74 |
| distill | 15 | 22 | 38 |
| preprocessing (all) | ~30 | ~45 | ~70 |
| **A + B total** | **~320** | **~500** | **~1,020** |
| astra_ceiling (C1) | 33 | 49 | 86 |
| panel_ext (C2) | 88 | 137 | 286 |

Largest expected lines: Opus 5.5 panel ($86) and effort arm ($92), Kimi K3
panel ($50), GPT-6 Sol panel ($48), Sonnet 5.5 panel ($43).

**Price update (5 Nov 2026).** The AWS Price List (us-east-1, Standard) returned prices for 7 of the
panel models. Kimi K3 ($3.30/$16.50 per 1M) and Grok 4.6 ($2.20/$6.60) are about 2.2x the
planning figures; the other five match or are slightly lower. With these prices `ccrl plan` gives
A+B ≈ $370 expected / $650 high before preprocessing (≈ $415 / $720 with it). Opus 5.5,
Sonnet 5.5 and GPT-6 Sol/Luna are not yet in the Price List API and still use list prices.

After the pilot, `ccrl report pilot` re-forecasts every remaining job from
measured mean (expected) and p90 (high) output tokens per model.

## Caps (enforced on every call)

hard $1,000 · job caps ≈ the high scenario (`config/budget.yaml`) · per-model
cap = max(1.3 × expected, high) of the job's forecast · priority C only at
≤ $700 committed and from 10 Dec · alerts at $250/500/750/900.

## If the pilot forecast for A + B (high) exceeds the cap — cut in this order

1. `distill` (−$22 expected; the SFT-from-frontier baseline becomes optional).
2. `sequential` restricted to Opus, Sol, Kimi, GLM (`--models`; −$12).
3. `effort_arm` at `high` only (edit `efforts` before freezing; −$70).
4. MedCaseReasoning authority and duplication calls for the 4 cheapest models
   (requires a re-freeze before the panel).
Never cut the core panel models or the retest call (the floor).
