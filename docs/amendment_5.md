# Amendment 5: final design before stage A (27 Nov 2026)

Made **before any training run under it and before any Bedrock call of Amendments 2–3**. It puts
into effect the plan in [plan_final.md](plan_final.md). OSF updates are set aside for now
(osf.io/4xmq7); this file is the dated record to file later in one step. Nothing in the frozen
panel items, prompts, models or readout of Paper 1 changes.

## A. Paper 1 (Bedrock)

| | was | now |
|---|---|---|
| ceiling for Amendments 2–3 | job caps summing to about $500, under a hard cap of $1,000 | **$200 on top of the $561.55 already spent**: hard cap $760 (`config/budget.yaml`) |
| job caps | temp0_panel 55, temp0_ext 30, contamination 200, prep_fresh_cut 40, prep_fresh_extract 40, fresh_panel 130 | 30, 15, 70, 15, 16, 70 (expected total about $150) |
| run order | §8b (Amendment 2) then §8c (Amendment 3) | temperature 0 jobs, then the post-freeze preparation and `fresh_panel`, then `contamination` last, because Kimi K3 is about 95% of its cost and a cap can then only shorten the probe |
| models | unchanged: all six in `fresh_panel` and `contamination`; five at temperature 0 (Bedrock refuses temperature 0 for Kimi K3) | unchanged |

`budget.yaml` is a spending control outside the freeze, so this needs no re-freeze. The planned
Bedrock job for a training paraphrase bank is not needed (B.2).

## B. Paper 2 (training), before stage A

### B.1 Shared-prompt training pass

Each GRPO step trains 160 sequences but only 48 distinct prompts, and prompts are about 70% of
the tokens. Each distinct prompt is now run once, and its key/value cache is shared by its
rollouts (`shared_prompt_logprobs`, `config/train/base.yaml: shared_prompt: true`).

- **The loss and gradient are unchanged.** CPU tests on transformers 4.57.6 (the Lambda
  version) and 5.22 give identical log-probs, entropies, k3 KL and LoRA gradients
  (`tests/test_shared_prompt.py`).
- **A GPU check runs before the first paid run** (`python -m ccrl.train.check_shared`): bf16,
  real prompts. It requires a gradient cosine ≥ 0.999 and a relative difference ≤ 1%, otherwise
  training falls back to the unshared pass.
- **Revised the same day, after the first GPU check and before any training run.**
  - **Memory.** The check ran out of memory at 8,192 tokens per micro-batch. The size is now
    4,096, and the check caps itself at the memory vLLM leaves in training, stepping down until a
    size fits. At 4,096 it fits (45 GB) and the step is 1.4× faster than the unshared pass.
  - **Pass rule.** The gradient cosine was 0.9992 and the relative difference 3.9%. A fixed 1% had
    no basis: in bf16, regrouping the same sequences into different micro-batches also changes the
    sum. The rule is now relative to that noise floor on both measures: cosine ≥
    min(0.999, 1 − 1.5 × (1 − floor cosine)) and relative difference ≤ max(1%, 1.5 × floor), where
    the floor compares two groupings of the unshared pass. The fp32 CPU test still requires exact
    equality.
  - **Measured:**

    | | shared vs unshared | unshared vs regrouped unshared (floor) |
    |---|---|---|
    | gradient cosine | 0.99890 | 0.99904 |
    | relative gradient difference | 4.8% | 4.4% |
    | mean per-token log-prob difference | 0.0200 | 0.0206 |

    Under the noise-floor rule this passes.
- **Gradient checkpointing is off** for GRPO, as the shared pass requires. The SFT baselines and
  the A100 (RLVR, 40 GB) keep the unshared pass with checkpointing (`--no-shared`).

### B.2 Training perturbation banks disjoint from the test items

Training used the same 230 paraphrases and 40 irrelevant sentences as the frozen test items, so
the DDXPlus paraphrase and cue scores partly measured memorised wording.

- **Paraphrases.** The training bank `artifacts/frozen/banks/ddx_phrasebank_train.json` uses, for
  each phrase, the next verified alternative after the one the test bank chose.
  - These are the archived generator and verifier outputs (`phrasebank_review.csv`), so no new
    model call is needed.
  - 213 of 230 phrases get a different paraphrase. The other 17 keep the base wording.
  - No training paraphrase equals a test paraphrase (`tests/test_cues.py`).
- **Irrelevant cues.** 40 new training-only sentences are in `config/cues_train.yaml`.
  - They follow the panel's rules and pass its lint and diagnosis-name check.
  - None repeats a test cue.
- **Effect.** Training configs set `cue_bank: train` and `phrasebank_path` to the training bank.
  Evaluation items are unchanged.

### B.3 New arms

| config | what | role |
|---|---|---|
| `brier_only` | the method's accuracy term (negative Brier) alone, same rollouts, credit and readout | does the coherence term do the work (C3) |
| `mech_team_lenient` | the method with the registered (renormalising) readout, 60 steps | mechanism 1 alone (C1) |
| `mech_registered_strict` | the method with the registered credit, 60 steps | mechanism 2 alone (C1) |
| `sft_selfcons` | SFT on the untrained model's mean posterior over 8 renderings (4 clean, 4 transformed) | the label-free "distil your own ensemble" baseline (C4, C5) |

SFT targets for `bct` and `sft_selfcons` are generated with vLLM at the registered decoding
(temperature 0.7, top-p 0.95).

### B.4 Primary endpoints and the H2 guard

The registered hypotheses H2–H7 are all still reported as registered. Raw consistency can be met
by flattening, so three endpoints are added as primary, each tested with a paired case bootstrap,
one-sided, with Holm over the three (`hypotheses.primary`).

| endpoint | definition | better |
|---|---|---|
| nuisance flip rate | share of the 7 meaning-preserving renderings whose top-1 answer differs from the base rendering's | lower |
| normalised incoherence | mean total incoherence divided by mean JS to other patients' base answers | lower |
| selectivity | mean JS to the answer with one informative finding removed, divided by mean nuisance JS | higher |

**Comparisons:**

- coherence arm vs RLVR (C2), vs Brier-only (C3), vs BCT and self-consistency distillation (C4);
- every arm vs the untrained model;
- each seed on its own: a result is called consistent only if every seed has the same sign.

**What the stopped runs show about these endpoints.** On the hedged 21-step coherence adapter:

- the flip rate barely moved (0.32 → 0.28);
- normalised incoherence still fell (0.36 → 0.27). Normalised incoherence is unaffected by
  tempering an answer, but not by mixing it with the uniform distribution, which is what filler
  does;
- that adapter's M was 0.12.

**Revised H2 guard.** The Amendment 4 guard (M ≥ 90% of RLVR's) would penalise a calibrated
coherence model against an overconfident RLVR, whose one-hot answers inflate M. It is replaced:

- **New rule.** The raw H2 reduction counts only if two conditions hold:
  - the flip rate and normalised incoherence are both lower than RLVR's (one-sided 95%);
  - the coherence arm's M is at least m0 = 0.459. The hedged adapter fails on this condition.
- **Still reported.** The Amendment 4 rule is shown as `supported_a4`.

**Secondary endpoints:**

- Brier reliability and resolution (top-label Murphy decomposition);
- authority deference (never trained);
- training curves of M, entropy and sum-to-100 compliance.

### B.5 General capability (registered H3, second part)

- **Items.** MMLU-Pro: 2,000 questions, a seeded sample stratified by category. GSM8K: all 1,319
  test questions.
- **Source.** Dataset revisions pinned in `ccrl/train/general.py`.
- **Decoding.** Temperature 0, thinking off.
- **Test.** Each trained arm against the untrained model, paired by question: one-sided 95% bound
  above −1.5 points.

### B.6 Milestones and go/no-go

- **Milestone adapters.** Kept at steps 50, 100, …, 300; the last two checkpoints still hold the
  optimizer state. The coherence runs' milestones are evaluated on the 30 pilot cases.
- **Go/no-go after step 50 of the first coherence run** (`ccrl train gonogo`). All four must hold:
  - rollout sum-to-100 compliance is at least 0.8 (steps 41–50);
  - the safeguard has not fired;
  - pilot entropy is at most the untrained model's + 0.5 nats;
  - pilot flip rate is at most the untrained model's + 0.05. The 0.05 is about one standard
    error on 30 cases.

  A no-go stops the queue, and the failure is diagnosed on the CPU first.
- **Stage B.** Funded only if stage A works, by the rule fixed in plan_final.md §3.5.

### B.7 Audit predictions (claim C7)

The estimator audit gains the readout channel: a filler knob that the lenient readout
renormalises and the strict readout rejects (`python -m ccrl.reward.audit_arms`). Its prediction
for every stage A arm is in [audit_predictions.md](audit_predictions.md). It is committed with this
amendment, before any stage A run, and compared with the GPU runs afterwards whichever way they
turn out.

### B.8 Unchanged

- **Reward constants:** τ_I 0.788, δ_S 0.234, m0 0.459, γ 0.5.
- **Training setup:** LoRA, learning rate, KL, rollouts per step and 300 steps.
- **Amendment 4 mechanisms:** strict readout, team credit and the collapse safeguard.
- **Evaluation:** items, decoding and readout.
