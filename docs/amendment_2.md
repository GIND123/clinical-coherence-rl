# Amendment 2 (26 Nov 2026): temperature 0 at full scale, and a contamination probe

*Written and committed publicly **before** any of the arms below ran. Freeze of the design these
arms run under (SHA-256 of `artifacts/frozen/FREEZE.json`):*
`0327bd84632b9f04a50cc8eecd5fcd85ec00e0725cc5e69550e87e4b19770b80`.
*The previous freeze (`ea6f2a24…`, Amendment 1) is kept in `artifacts/frozen/freeze_history/`.*

This amendment concerns Paper 1 (the Bedrock panel) only. It does not change anything in the
training study registered on OSF (osf.io/4xmq7).

## 1. Why

The secondary arms (docs/results_secondary.md) produced two results that the paper's main claims
now rest on, and one obvious objection.

1. **The temperature-0 result rests on few cases.** At temperature 0, re-ordering and rewording
   the case changes the answer beyond the model's own noise. This was seen on the core 150 cases
   only (75 per dataset). The intervals are wide, and the effect was not tested on the
   replication sample.
2. **Contamination is the obvious objection to the real cases.** The 300 MedCaseReasoning cases
   are published case reports (2005–2025), all from before the panel models' training cutoffs.
   A reviewer will ask whether real-case accuracy reflects memorisation. A before/after-cutoff
   split is impossible, because only 9 cases are from 2025.

## 2. Arms

| arm | design | calls | forecast (expected / high) |
|---|---|---|---|
| `temp0_panel` | all 600 confirmatory cases × all 10 calls × 5 models at temperature 0 | 30,000 | $17 / $45 |
| `temp0_ext` | the 300 replication cases × 10 calls × 5 models at temperature 0 | 15,000 | $9 / $23 |
| `contamination` | 300 MedCaseReasoning panel cases × 4 probes × 6 models, provider default | 7,200 | $40 / $176 |

**Kimi K3 is not in the temperature-0 arms**, because Bedrock rejects the `temperature` field for
that model. Everything else is identical to the main panel: items, prompts, parser and models.
The core 150 cases are asked again under the new job names. Together with `temp0`, this gives a
second independent temperature-0 answer for those cases.

**Contamination probes** (`src/ccrl/panel/contamination.py`; prompts `config/prompts/contam_*.txt`):

| probe | the model sees | answer |
|---|---|---|
| `id` | PMCID, journal and year of the case report, and its 10 candidate diagnoses; no clinical details | posterior JSON, same system prompt and parser as the panel |
| `noid` | the same candidates, with the identity withheld | posterior JSON |
| `guided` | source identity and the first part of the case presentation (whole sentences, at least half the characters); asked to continue "exactly as it appears in the original publication" | free text, about as long as the withheld part (40–150 words) |
| `general` | the same first part, with no source; asked to continue | free text |

## 3. Pre-specified analyses

**Temperature 0 (full scale).** The same estimands as the main panel (`ccrl analyze`), and the
paired comparison with the main panel on identical cases (`ccrl.analysis.secondary`):

- the retest flip rate and top-1;
- the excess order and paraphrase JS beyond the condition's own retest JS.

`temp0_ext` is compared with `panel_ext`.

**Claim tested:** at temperature 0 the excess order or paraphrase effect is above zero (95% CI
lower bound > 0) for models whose effect at default decoding includes 0. It is reported per model
and dataset, without a directional prediction for any single model. Replication: the same
comparison on `temp0_ext`.

**Contamination.** All of these are per model, with paired case bootstrap CIs (2,000 resamples):

- **Primary:** top-1 under `id` minus top-1 under `noid`. Above 0 means the model recalls the
  diagnosis from the publication's identity.
- **Secondary:**
  - the gold probability under `id` minus `noid`;
  - verbatim overlap with the withheld text, `guided` minus `general`: ROUGE-2 F1, the share of
    8-grams found in the withheld part, and the longest shared token sequence;
  - the share of cases with a shared sequence of at least 13 tokens (verbatim recall);
  - whether the continuation names the final diagnosis, `guided` minus `general`.
- **Link to the panel.** A case is *suspect* for a model if either holds:
  - it answers correctly under `id` and wrongly under `noid`;
  - or its `guided` continuation shares a sequence of at least 13 tokens with the original.

  We report main-panel top-1 and retest flip rate on suspect cases versus the rest, and real-case
  top-1 with suspect cases excluded, as a sensitivity analysis.

**How the results are read:**

- **If `id` − `noid` and `guided` − `general` include 0 for a model,** there is no evidence that
  the model memorised these case reports.
- **If they are above 0,** the paper reports the size of the effect and the sensitivity analysis.

The stability findings (retest flips and presentation effects) do not depend on accuracy, and are
reported whichever way this comes out.

## 4. Budget

The expected cost is about $66 and the high scenario about $244. The job caps are $55, $30 and
$200 (`config/budget.yaml`), all within the remaining hard-cap headroom ($438 at $561.55
committed). Calls must finish by 18 Dec 2026.
