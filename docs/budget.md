# Budget

Credit $1,100 ($1,000 "RSA pilot Fall 2025" + $100 Free Tier), expiring end
of December 2026. The harness hard cap is $1,000, and no paid call is made
after 18 Dec 2026 23:59.

The credit covers the open-weight models served by Bedrock. It does not
appear to cover partner models (Claude, GPT-6, Grok): those return
AccessDeniedException on the PI's account. So the panel is the six
open-weight models.

## Calls per job (`config/jobs.yaml`)

| job | priority | design | calls |
|---|---|---|---|
| probe | A | a few tiny calls × 6 models | ~40 |
| prep_phrasebank | A | 10 batches generate (Kimi K3) + 10 verify (GLM-5) | 20 |
| prep_mcr_extract | A | 436 pilot/panel + 2,150 training cases (Kimi K3) | 2,586 |
| prep_mcr_verify | A | 436 pilot/panel (GLM-5) | 436 |
| pilot | A | 30 cases × (10 calls + 1 schema variant) × 6 models | 1,980 |
| panel_main | A | 600 cases (300 + 300) × 10 × 6 | 36,000 |
| panel_ext | B | 300 DDXPlus (replication) × 10 × 6 | 18,000 |
| effort_arm | B | 150 × 6 calls × 3 efforts × gpt-oss-120b | 2,700 |
| sequential | B | 150 × 2 orders × 2 turns × 6 models | 3,600 |
| distill | B | 2,000 × Kimi K3 | 2,000 |
| partner_panel | C (optional) | 600 × 10 × whichever partner models AWS enables | ≤ 30,000 |

## Pre-pilot forecast (USD)

`ccrl plan` computes this from:

- **input:** the exact prompt texts (≈ 3 characters per token,
  conservative);
- **output:** 600 / 1,200 / 2,500 tokens per call at provider default
  (low / expected / high). gpt-oss-120b uses the effort-specific
  assumptions (low 500 / 800 / 1,500; medium 1,000 / 1,600 / 2,500; high
  2,000 / 3,000 / 5,000);
- **retries:** +5 / 8 / 12 %;
- **Flex discount:** applied to the low and expected scenarios only.

Prices are the AWS Price List values (us-east-1, Standard) the PI retrieved
on 5 Nov 2026.

| | low | expected | high |
|---|---|---|---|
| pilot | 1.5 | 2 | 6 |
| panel_main | 47 | 64 | 182 |
| panel_ext | 24 | 33 | 94 |
| effort_arm | 1 | 1.4 | 4 |
| sequential | 5 | 7 | 20 |
| distill | 10 | 14 | 41 |
| preprocessing (all) | ~30 | ~60 | ~150 |
| **A + B total** | **~120** | **~180** | **~500** |
| partner_panel (C, only if enabled) | 190 | 280 | 570 |

Kimi K3 is about two-thirds of every panel job: it has the highest output
price ($16.50 per 1M). The other five models together cost less than Kimi
alone.

After the pilot, `ccrl report pilot` re-forecasts every remaining job from
the measured mean (expected) and p90 (high) output tokens per model. Expect
the "high" column to be the realistic one for reasoning-heavy models.

## Caps (enforced on every call)

- **Hard cap:** $1,000 across everything.
- **Job caps:** about 1.5 × the high scenario (`config/budget.yaml`).
- **Per-model cap:** max(1.3 × expected, high) of the job's forecast.
- **Priority C:** only after `panel_main`, and only at ≤ $700 committed.
- **Alerts** at $100 / 250 / 500 / 750 / 900.

## Headroom

A + B leaves roughly $500–800 of the cap unused even in the high scenario.
It is held, in this order, for:

1. Real token counts exceeding the forecast. Reasoning models can be several
   times more verbose than assumed; the pilot will tell.
2. Re-runs of failed items.
3. The optional `partner_panel`, if AWS enables partner models in time to
   finish by 18 Dec (about 2 days of running).

Nothing is added to the confirmatory design after the OSF registration.
