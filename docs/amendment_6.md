# Amendment 6: arms that answer the likely reviewer questions (27 Nov 2026)

Made **before any call below**. Paid Bedrock calls end on 18 Dec 2026 and the credit expires on
31 Dec, so anything a reviewer could ask that needs a Bedrock call must run now. The ceiling for
Amendments 2, 3 and 6 together is **$350** on top of the $561.55 already spent: the hard cap in
`config/budget.yaml` is $911. The registered arms (Amendments 2–3) run first, so a cap can only
shorten these new ones (`scripts/final_bedrock.sh`).

## 1. Open-ended differential format (Paper 1)

**The question it answers.** The panel asks for a probability on every diagnosis of a given
candidate list. A reviewer can fairly ask:

- whether the instability is an artefact of that format;
- whether a clinician, who has no candidate list, would see it.

**Design** (`ccrl openended`, `src/ccrl/panel/openended.py`):

| | |
|---|---|
| prompt | `config/prompts/open_system.txt`: "the five diagnoses you consider most likely, the most likely first", JSON list of names; no candidate list, no probabilities |
| vignettes | **byte-identical** to the panel's, extracted from the frozen items (checked: the DDXPlus user turns are identical; the MedCaseReasoning ones lose only the candidate block) |
| cases | 75 DDXPlus core cases, 300 MedCaseReasoning panel cases, and every post-freeze real case (Amendment 3) once its items exist |
| renderings | base, retest (identical request), perm1 (finding order), para (paraphrase) |
| models and decoding | `openended`: 6 panel models at provider defaults; `openended_t0`: the 5 that accept temperature 0 |
| judges | `openended_judge`: GLM-5 and DeepSeek V3.2. Each maps every free-text diagnosis to the case's candidate codes (synonyms, subtypes and more specific forms count as the same condition) or to X-labels shared within the request (`config/prompts/open_judge_system.txt`). There is one call per case × model × decoding, and the four answers appear in a seeded random order under neutral ids, so the judge cannot tell which is the base rendering |

**Endpoints.** These are computed per case and model, under the judge that is not that model.
For GLM-5's and DeepSeek's own answers, the other judge is used.

- top-1 flip relative to the base rendering, for retest, order and paraphrase;
- top-1 and top-5 accuracy (mapped code equals the gold code);
- top-5 Jaccard overlap with the base answer;
- judges' agreement on the mapped top-1.

**Main comparison.** The retest and presentation flip rates in the open format against the
panel's top-1 flip rates on the same cases, models and renderings. Paired by case, with a case
bootstrap.

**Judge validation, with no further Bedrock call:**

- the two judges' agreement;
- a clinician check of 100 random mappings, added to Paper 1's clinician component.

The judge labels also serve later as the reference for validating a local judge. Paper 2 can
then score its trained models in the open format after Bedrock access has ended.

## 2. Temporal reproducibility (Paper 1)

**Job `drift`** re-sends the identical base and retest requests of the 150 core cases, under a
new job name, about three weeks after the main panel. Request IDs depend on the job, so these are
new calls.

**What it shows:** whether accuracy and flip rates on the same Bedrock model IDs are stable over
time.

## 3. Real-case training (Paper 2, GPU only)

**The objection.** The training study trains on DDXPlus, which is synthetic.

**The arms:**

- `ccr_mcr`: the method's reward (coherence and Brier accuracy) trained on the 2,000 labelled
  MedCaseReasoning training cases;
- `rlvr_mcr`: RLVR on the same cases.

**The data.** These cases are real published case reports with 10 candidates each, disjoint from
the 300 panel cases and from the post-freeze cases. The fidelity term is off (γ = 0), because no
Bayes reference exists for real cases.

**When they run:**

- at the end of the stage A H100 queue, if the budget guard finds the money (Brier-only is
  predicted to stop early);
- otherwise first in stage B.

**What they need:** no Bedrock call.

## 4. Costs and caps

| job | calls | expected | cap |
|---|---|---|---|
| `openended` | (375 + post-freeze cases) × 4 × 6 | ~$85 (Kimi K3 ~$0.038 per call is ~90%) | $110 |
| `openended_t0` | same × 5 models | ~$5 | $12 |
| `openended_judge` | 2 judges × cases × 11 model-decodings | ~$8 | $20 |
| `drift` | 150 × 2 × 6 | ~$12 | $18 |

With Amendments 2–3 (about $150 expected), the total expected is about $260 against the $350
ceiling.

## 5. Considered and not added

| idea | why not |
|---|---|
| authority-cue dose-response (other stated reliabilities, a correct suggestion) | F8 is a secondary finding; it needs new items, and Kimi K3 would make it about $25 |
| a second post-freeze wave (case reports published 28 Nov–12 Dec) | it needs a second Bedrock session after about 12 Dec; Amendment 3 already uses every passing case |
| demographic perturbations | no agreed normative target (age and sex legitimately change priors) |
| closed models | not available on the PI's Bedrock account |

## 6. Freeze

- **What is added:** the new prompts (`open_system`, `open_judge_system`, `open_judge_user`), the
  `drift` job and the `base_retest` transform set.
- **What is unchanged:** every earlier input. No earlier item, prompt or request ID changes.
- **The re-freeze:** it keeps the previous FREEZE.json in `freeze_history/`.
