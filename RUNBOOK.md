# RUNBOOK — running the Bedrock panel

For the person running the paid panel from their own terminal with their own
AWS credentials. Every step gives the exact command, what it costs, how long
it takes, and what "OK" looks like. Nothing in this file needs Python
knowledge beyond copying commands.

> **Hard deadline: no paid call after 18 Dec 2026, 23:59** (the harness
> enforces it). Credit: $1,100; hard cap in the harness: $1,000.
> **Panel: 6 open-weight models** (Kimi K3, GLM-5, DeepSeek V3.2,
> Qwen3-Next-80B, gpt-oss-120b, Qwen3-32B).
> Expected total spend for everything required: about **$180** (high
> scenario ≈ $500).

---

## Already cloned and probed once? Start here

```bash
cd clinical-coherence-rl
git pull                       # the design changed to the 6 open-weight models
source .venv/bin/activate
ccrl doctor                    # must show no FAIL except possibly "probe results are stale"
ccrl rehearse                  # $0, ~10 min, must end with "REHEARSAL PASSED"
ccrl models resolve            # $0
ccrl models probe              # ~$0.10: re-probes the 6 panel models under the new settings
```

Then continue at step 3. Step 2 explains what the probe output means.

---

## 0. Five rules

1. **Never delete `runs/`.** It holds the ledger (every request, response and
   cent), and everything resumes from it. A backup copy is written after
   every job.
2. **Run long jobs detached:** `bash scripts/run_detached.sh run <job>`. This
   survives a closed terminal and keeps the Mac awake (`caffeinate`).
3. **To stop a job, press Ctrl-C once** (or `kill -INT $(cat runs/logs/last.pid)`).
   In-flight calls finish and are recorded. Re-running the same command later
   continues where it stopped and never re-sends completed requests.
4. **Rehearse first.** `ccrl rehearse` runs the whole pipeline on an offline
   mock in about 10 minutes at zero cost. If it passes on your machine, the
   paid run will behave the same way.
5. **Do not edit `config/` or `artifacts/frozen/` after `ccrl freeze`.** The
   panel refuses to run if anything that defines the prompts changed.

---

## 1. One-time setup (≈10 minutes, $0)

```bash
git clone https://github.com/GIND123/clinical-coherence-rl.git
cd clinical-coherence-rl
python3 --version          # needs 3.10 or newer (conda base is fine)
bash scripts/setup.sh      # creates .venv and installs pinned dependencies
source .venv/bin/activate  # do this in every new terminal
ccrl doctor                # checks Python, packages, config, AWS identity, Bedrock reachability
ccrl rehearse              # full zero-cost dry run (~10 min); must end with "REHEARSAL PASSED"
                           # (a "NO-GO" line on MOCK data inside it is expected: the mock injects failures)
python -m pytest -q        # optional: the test suite, ~2 min
```

If `python3 --version` is below 3.10:
`conda create -n ccrl python=3.12 -y && conda activate ccrl`, then
`PYTHON=python bash scripts/setup.sh`.

**AWS:** the harness uses your normal AWS credentials (the same ones that made
`aws bedrock-runtime converse` work). If you use a named profile, add
`--profile <name>` right after `ccrl` in every command, e.g.
`ccrl --profile lab doctor`. Region is `us-east-1` (set in `config/models.yaml`).

**Recommended:** in the AWS console → Billing → Budgets, create a cost budget
of $1,000 with e-mail alerts at 25/50/75/90 %. The harness has its own caps;
this is a second, independent alarm.

---

## 2. Models: resolve and probe (≈10 minutes, ≈$0.10)

```bash
ccrl models list          # every model / inference profile your account can see ($0)
ccrl models resolve       # matches the study models to live IDs -> config/models.resolved.yaml ($0)
ccrl models probe         # a few tiny real calls per panel model -> config/models.lock.yaml
```

The probe prints one line per model, e.g.
`[probe] gpt-oss-120b: OK id=openai.gpt-oss-120b-1:0 effort=reasoning_field/medium tier=flex ...`.

* **`OK` with no `WARN`:** fine.
* **`WARN: service tier flex rejected; using default`:** fine (it costs a bit
  more, and is recorded).
* **`WARN: effort may be ignored` for gpt-oss-120b:** tell the team before
  the effort arm (step 8). Everything else can go ahead.
* **`FAILED` for any of the 6 panel models:** do not continue; send the
  printed line to the team.

Closed partner models (Claude, GPT-6, Grok) are **not** probed by default:
the account cannot call them. If AWS later enables some, probe just those:
`ccrl models probe --models opus-5.5,sonnet-5.5,gpt-6-sol,gpt-6-luna,grok-4.6`.

---

## 3. Verify prices (5 minutes, $0)

Open `config/prices.yaml`. The six panel rows are the AWS Price List values
from 5 Nov. Compare them with <https://aws.amazon.com/bedrock/pricing/>,
correct any number, and set `verified: true` and `checked: 2026-11-xx` on
rows you confirmed. Unverified rows are budgeted with a 25 % safety margin,
so this only makes the caps tighter, never looser.

---

## 4. Preprocessing (≈2–3 hours wall time, ≈$25 now + ≈$40–100 later)

```bash
ccrl prep phrasebank                              # DDXPlus paraphrase bank (~$1)
bash scripts/run_detached.sh prep mcr --stage all --roles pilot,panel   # 436 real cases, extract + verify (~$15–30)
ccrl prep finalize                                # deterministic; prints prepared counts
```

`finalize` must report at least **10 ok pilot** and **300 ok panel**
MedCaseReasoning cases (it prepares spares). If it reports fewer, send the
output to the team; do not build items.

Two review files are created for the clinicians. They can review in parallel
with the pilot; the panel does not wait for them.

* `artifacts/frozen/banks/phrasebank_review.csv`
* `artifacts/frozen/medcase/medcase_review.csv`

The training-set extraction (2,150 cases, ≈$40–100) is needed only for the
training part. Run it any time before 18 Dec, e.g. overnight:

```bash
bash scripts/run_detached.sh prep mcr --stage extract --roles train
```

---

## 5. Build, freeze, plan (2 minutes, $0)

```bash
ccrl build items          # every prompt the study will ever send
ccrl freeze               # hashes all inputs -> artifacts/frozen/FREEZE.json
ccrl plan                 # call counts and cost forecast per job
```

Record the freeze hash for the pre-registration:
`shasum -a 256 artifacts/frozen/FREEZE.json`.

---

## 6. Pilot (≈1 hour, ≈$2–6)

```bash
bash scripts/run_detached.sh run pilot
tail -f runs/logs/ccrl-run-pilot-*.log      # Ctrl-C stops watching, not the job
ccrl report pilot                           # after it finishes
```

The report ends with **GO** or **NO-GO** and the reasons. The gates, per
model:

| gate | threshold |
|---|---|
| parse rate | ≥ 98 % |
| refusals | ≤ 2 % |
| truncations | ≤ 1 % |
| timeouts | ≤ 2 % |
| informative answers | ≥ 90 % |
| parse-rate spread across transformations | ≤ 5 points |

The re-forecast of A+B must also be within the $1,000 cap.

On **NO-GO**: send `runs/reports/pilot_report.txt` to the team. Do not start
the panel.

**Billing check (24–48 h after the pilot):**

```bash
ccrl billing reconcile --start 2026-11-07
```

Or in the console: Billing → Cost Explorer, group by *Service* and by *Charge
type*. Every model line should be matched by a **Credit** line. If anything
is billed to the card, stop and tell the team.

---

## 7. Pre-register (OSF), then the main panel (≈1 day, ≈$65–180)

Upload `docs/preregistration.md`, with the freeze hash filled in, to OSF
**before** this step.

```bash
bash scripts/run_detached.sh run panel_main
ccrl status               # any time: progress and spend per model
```

It runs 600 cases × 10 calls × 6 models = 36,000 calls. If the machine
sleeps, the network drops, or you stop it, run the same command again.

---

## 8. Secondary arms (≈1 day, ≈$55–160)

Each starts only after `panel_main` is complete:

```bash
bash scripts/run_detached.sh run panel_ext    # replication sample: 300 more DDXPlus cases, 6 models
bash scripts/run_detached.sh run effort_arm   # gpt-oss-120b at low / medium / high reasoning effort
bash scripts/run_detached.sh run sequential   # two-stage evidence, 6 models
bash scripts/run_detached.sh run distill      # 2,000 Kimi K3 answers for a training baseline
```

---

## 9. Optional: partner models (only if AWS enables them)

```bash
ccrl models probe --models opus-5.5,sonnet-5.5,gpt-6-sol,gpt-6-luna,grok-4.6
ccrl status                                   # must show <= $700 committed
bash scripts/run_detached.sh run partner_panel
```

It runs only on partner models that passed the probe, and is reported
separately as exploratory.

---

## 10. Archive and hand back (before 18 Dec)

```bash
ccrl export               # ledger -> parquet + MANIFEST.json (SHA-256 per file)
ccrl archive pack         # runs/archive/ccrl-archive-<date>.tar.gz + .sha256
ccrl analyze              # first-look tables and figures in runs/reports/analysis/
```

Send the team:

* the `.tar.gz` and its `.sha256` (shared drive / Zenodo);
* if you cannot push to GitHub, the small frozen files:
  `config/models.lock.yaml` and the whole `artifacts/frozen/` folder.

With push access instead:

```bash
git add config/models.lock.yaml config/prices.yaml artifacts/frozen
git commit -m "Frozen panel inputs and model lock"
git push
```

---

## 11. Troubleshooting

| You see | Meaning | Do |
|---|---|---|
| `Freeze check failed: changed: [...]` | an input changed after `ccrl freeze` | undo the edit; if it was intended, tell the team before re-freezing |
| `config/models.yaml changed for [...] since they were probed` | the probe results are older than the model settings | run the printed `ccrl models probe --models ...` (a few cents) |
| `hard cap $1000.00` / `job cap` in the summary | a cap was reached | `ccrl status`; tell the team; do not edit caps without agreement |
| `model_halted` / `ValidationException` | Bedrock rejected a parameter for one model | other models continue; re-run `ccrl models probe --models <key>` and send the output |
| `effort_arm: gpt-oss-120b has no working reasoning-effort control` | the probe found the effort setting ignored | nothing to do; tell the team (the arm is skipped) |
| `AccessDeniedException`, `ExpiredToken` | credentials | refresh credentials (`aws sso login` / new keys), re-run the same command |
| `AccessDeniedException ... not available for this account` | model not covered by the account or credit | expected for partner models; for a panel model, tell the team |
| many `ThrottlingException` lines | account quota | nothing; the harness backs off and continues |
| `Hard stop ... has passed` | after 18 Dec | expected; no paid calls are possible |
| `requires panel_main to be complete` | order rule | finish the earlier job first |
| laptop slept / Wi-Fi dropped | — | re-run the same command; completed work is never repeated |
| `REHEARSAL` fails | local environment problem | send the last 30 lines of the output |

Every attempt for a specific request is in `runs/ledger.sqlite`, mirrored to
`runs/raw/<job>/<model>.jsonl` (request, response, usage, cost).
