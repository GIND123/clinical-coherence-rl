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
| temp0 | B (amendment 1) | core 150 × 6 calls × 6 models at temperature 0 | 5,400 |
| resample | B (amendment 1) | 300 DDXPlus × (base, ablate) × 3 replicates × 6 models | 10,800 |
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

## Measured (7–18 Nov 2026): $240.36 spent

From the PI's ledger (reported cost; AWS's Credits page matched the ledger
within 1% on 7 Nov).

| job | calls | reported cost |
|---|---|---|
| preprocessing (phrase bank, 436 extractions + verifications) | 905 | $29.73 |
| pilot | 2,014 | $10.41 |
| panel_main | 36,504 | $200.19 |
| probes | 143 | $0.02 |

Measured cost per call (Flex), main panel:

| model | DDXPlus | real cases | output tokens (mean) |
|---|---|---|---|
| Kimi K3 | $0.0255 | $0.0376 | 3,661 (reasoning) |
| GLM-5 | $0.0009 | $0.0004 | 189 |
| DeepSeek V3.2 | $0.0006 | $0.0003 | 219 |
| gpt-oss-120b | $0.0005 | $0.0003 | 1,170 (reasoning) |
| Qwen3-Next-80B | $0.0002 | $0.0001 | 188 |
| Qwen3-32B | $0.0001 | $0.0001 | 178 |

Kimi K3 is 95% of the panel cost. The main panel came in above the pre-pilot
"high" forecast ($182), because Kimi K3 writes about 3,700 tokens of
reasoning per answer, against the 2,500 assumed.

## Remaining (forecast from measured per-call costs)

| job | calls | expected |
|---|---|---|
| effort_arm | 2,700 | $2 |
| temp0 (amendment 1) | 5,400 | $30 |
| sequential | 3,600 | $20 |
| resample (amendment 1) | 10,800 | $50 |
| distill | 2,000 | $51 |
| panel_ext | 18,000 | $84 |
| MedCaseReasoning training extraction | 2,150 | $140 |
| **total remaining** | **44,650** | **≈ $377** |

**Projected total ≈ $617 of $1,100**, leaving about $480 of credit unused.
That reserve covers re-runs and the optional partner arm.

## Caps (enforced on every call)

**Guard cost vs real cost.** Caps are checked against the conservative
*guard* cost, never the reported cost. Until Flex billing is confirmed, the
guard takes the full Standard price for Flex calls, so it runs about 2× the
real spend. At the end of the main panel it showed $482 committed for $240
spent. This is why the pilot and panel job caps were hit early.

`ccrl billing confirm-flex --start 2026-11-06` fixes it. It compares Cost
Explorer with the ledger on complete days, records the evidence in
`runs/flex_confirmed.json`, and re-counts past Flex calls at their real
price. Nothing in `config/` changes, and no re-freeze is needed. Run it
before the remaining jobs. Without it, the $1,000 hard cap would stop them at
about $760 of real spend.

- **Hard cap:** $1,000 across everything.
- **Job caps:** about 1.5 × the high scenario (`config/budget.yaml`).
- **Per-model cap:** max(1.3 × expected, high) of the job's forecast.
- **Priority C:** only after `panel_main`, and only at ≤ $700 committed.
- **Alerts** at $100 / 250 / 500 / 750 / 900.

## Headroom

About $480 of credit stays unused after everything planned. It is held, in
this order, for:

1. Real token counts exceeding the forecast (Kimi K3 already did once).
2. Re-runs of failed items.
3. The optional `partner_panel`, if AWS enables partner models in time to
   finish by 18 Dec (about 2 days of running).

Additions after the main panel are filed as OSF amendments before they run (docs/osf_amendment_1.md).
