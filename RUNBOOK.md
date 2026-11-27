# RUNBOOK — running the Bedrock panel

For the person running the paid panel from their own terminal with their own
AWS credentials. Every step gives the exact command, what it costs, how long
it takes, and what "OK" looks like. Nothing in this file needs Python
knowledge beyond copying commands.

> **Hard deadline: no paid call after 18 Dec 2026, 23:59** (the harness
> enforces it). Credit: $1,100; hard cap in the harness: $1,000.
> **Panel: 6 open-weight models** (Kimi K3, GLM-5, DeepSeek V3.2,
> Qwen3-Next-80B, gpt-oss-120b, Qwen3-32B).
> **Status 18 Nov 2026:** preprocessing (pilot + panel), pilot and the main
> panel are done ($240 spent). **Continue at §8.** The remaining jobs cost
> about $377 (total ≈ $617 of the $1,100 credit).

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
   (Exceptions: `config/prices.yaml` and `config/budget.yaml` are spending
   controls and can be corrected at any time.)

**Keeping a detached job alive on a Mac:** plug in the power, keep the lid
open (closing it sleeps the Mac even with `caffeinate`), and start jobs from
a normal Terminal window rather than from inside an AI assistant's shell,
which may stop background processes when its session ends. If a job stops
anyway, run the same command again: finished calls are never repeated.

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

## 4. Preprocessing (≈2–3 hours wall time, ≈$30 now + ≈$140 later)

```bash
ccrl prep phrasebank                              # DDXPlus paraphrase bank (~$1)
bash scripts/run_detached.sh prep mcr --stage all --roles pilot,panel   # 436 real cases, extract + verify (~$30)
ccrl prep finalize                                # deterministic; prints prepared counts
```

If the job stops before it finishes (check with `ccrl status` or the end of
the log), run the same `run_detached.sh prep mcr ...` command again. Finished
extractions are skipped and it continues with what is left.

`finalize` must report at least **10 ok pilot** and **300 ok panel**
MedCaseReasoning cases (it prepares spares). If it reports fewer, send the
output to the team; do not build items.

Two review files are created for the clinicians. They can review in parallel
with the pilot; the panel does not wait for them.

* `artifacts/frozen/banks/phrasebank_review.csv`
* `artifacts/frozen/medcase/medcase_review.csv`

The training-set extraction (2,150 cases, ≈$140 at the measured
$0.065 per case) is needed only for the training part. Run it any time before 18 Dec, e.g. overnight:

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

**If the AWS total matches the ledger's reported cost** (`ccrl status`,
"reported" column), AWS is billing Flex calls at the Flex rate. Then set
`guard_counts_flex: true` in `config/prices.yaml`; this needs no re-freeze.
From then on the budget guard counts finished Flex calls at their real price
instead of double. Without it, the caps fill about twice as fast as the money
is actually spent.

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

## 8. After the main panel: secondary arms (≈1–2 days of mostly waiting, ≈$377)

> **Done 24–25 Nov 2026** ($560.02 in total; docs/results_secondary.md). The training cases were
> finalised from the archive on the analysis machine (`medcase/prepared_train.jsonl`). Do not run
> `ccrl prep finalize` on the panel machine.

**8.1 Update (2 min, $0).** Your local edits to `config/budget.yaml` (raised
caps) and `config/prices.yaml` (verified prices) are already in the repo
version. Discard them, then pull:

```bash
cd clinical-coherence-rl && source .venv/bin/activate
git checkout -- config/budget.yaml config/prices.yaml
git pull
ccrl doctor               # no FAIL, except "freeze" until 8.4
```

**8.2 Confirm Flex billing (1 min, ~$0.01).** Until this is done, the guard
counts every Flex call at the full price. The pilot and panel caps were hit
at half their real spend for this reason, and the remaining jobs would stop
early. The command compares Cost Explorer with the ledger on complete days.
If they match, it records the evidence and re-counts past Flex calls at
their real price (committed drops from about $482 to about $242).

```bash
ccrl billing confirm-flex --start 2026-11-06
```

**No Cost Explorer access** (e.g. only a Bedrock API key on this machine)?
Open AWS console → Billing → Credits and read **Total estimated amount used**.
Pass that figure instead; the newest paid call must be at least 48 h old:

```bash
ccrl billing confirm-flex --aws-total 241.50     # use the number shown on the Credits page
```

| it prints | meaning | do |
|---|---|---|
| `[confirmed]` and `applied: committed total now ...` | AWS bills Flex at the Flex rate | continue |
| `[inconclusive]` | Cost Explorer has not caught up | try again tomorrow; do 8.3–8.4 meanwhile, but do not start 8.5 until it says confirmed |
| `[standard]` | AWS bills at full price | stop; tell the team |

**8.3 Free disk space ($0).** Each job refuses to start with less than 5 GB
free. The main panel run crashed once on a full disk.

```bash
rm -rf runs/dryrun                    # rehearsal leftovers
ls -lh runs/backups/                  # keep the newest one; delete the others
df -h .                               # want >= 10 GB free
```

**8.4 Re-freeze for amendment 1 and register it ($0).** Two arms were added
after the main panel (`temp0`, `resample`; see `docs/osf_amendment_1.md`).

```bash
ccrl freeze                                     # the old FREEZE.json is kept in artifacts/frozen/freeze_history/
shasum -a 256 artifacts/frozen/FREEZE.json      # paste into docs/osf_amendment_1.md
ccrl plan                                       # call counts and cost per job
```

File `docs/osf_amendment_1.md` on OSF (as an update to the registration)
**before** 8.5.

**8.5 Run the remaining arms, one after another, in one detached process.**
Plug the Mac in, keep the lid open, and start it from a normal Terminal
window.

```bash
bash scripts/run_detached.sh queue effort_arm temp0 sequential resample distill panel_ext
tail -f runs/logs/ccrl-queue-*.log      # Ctrl-C stops watching only
ccrl status                             # progress and spend per model, any time
```

| job | what | calls | ≈ cost | ≈ time |
|---|---|---|---|---|
| effort_arm | gpt-oss-120b at low / medium / high effort | 2,700 | $2 | 20 min |
| temp0 | 150 cases at temperature 0, 6 models | 5,400 | $30 | 1 h |
| sequential | two-stage evidence, 6 models | 3,600 | $20 | 1 h |
| resample | 300 DDXPlus cases, base + ablated × 3 repeats | 10,800 | $50 | 1.5 h |
| distill | 2,000 Kimi K3 answers (training baseline) | 2,000 | $51 | 1.5 h |
| panel_ext | replication: 300 more DDXPlus cases × 10 calls | 18,000 | $84 | 2.5 h |

If it stops (sleep, network, a cap), run the same `queue` command again.
Finished calls are never repeated, and finished jobs are skipped in seconds.

**8.6 Training-set extraction (overnight, ≈ $140, ≈ 14 h).** This is needed
only for the label-free training arm.

```bash
bash scripts/run_detached.sh prep mcr --stage extract --roles train
```

**8.7 Push the frozen inputs (2 min).** The training side evaluates on exactly
these items.

```bash
git add config/models.lock.yaml artifacts/frozen
git commit -m "Frozen panel inputs, model lock and amendment-1 freeze"
git push
```

Without push access, zip `config/models.lock.yaml` and the whole
`artifacts/frozen/` folder and send them.

---

> **Order and ceiling (27 Nov 2026, docs/amendment_5.md §A and docs/amendment_6.md).** Run these in
> this order:
>
> 1. the 8b temperature 0 queue;
> 2. all of 8c (`fresh run`);
> 3. the 8b `contamination run`;
> 4. Amendment 6: `openended run`, `openended judge`, `queue drift`.
>
> Together they may spend at most **$350** on top of the $561.55 already spent: `config/budget.yaml`
> sets the hard cap to $911, with a cap per job. The registered arms run first, so a cap can only
> shorten the newest work.
>
> **One command does all three in this order, then exports and packs:**
> `git pull && bash scripts/final_bedrock.sh start`. Watch it with
> `tail -f runs/logs/final_bedrock.log`.
>
> `ccrl freeze --check` now lists **46 files**: the two training-only banks of Amendment 5 and the
> three open-ended prompts of Amendment 6.

## 8b. Amendment 2: temperature 0 at full scale and a contamination probe (≈ $66, before 18 Dec)

Design and reasons: docs/amendment_2.md. Expected cost is about $66 and the worst case about
$244. This is well inside the remaining hard-cap headroom.

```bash
cd clinical-coherence-rl && source .venv/bin/activate
git pull
ccrl freeze --check            # must print OK (46 files, frozen 2026-11-27...)
ccrl status                    # committed about $561.55
ccrl plan | grep temp0_        # temp0_panel ~$17, temp0_ext ~$9 expected
ccrl contamination plan        # ~$40 expected (Kimi K3 is most of it)
bash scripts/run_detached.sh queue temp0_panel temp0_ext       # ~45,000 cheap calls, a few hours
```

When the log shows the queue has finished:

```bash
bash scripts/run_detached.sh contamination run                 # 7,200 calls, mostly waiting on Kimi K3
```

When both have finished, pack and send the archive as before:

```bash
ccrl status
ccrl export && ccrl archive pack
```

Upload the newest `runs/archive/*.tar.gz` and its `.sha256` to the shared folder.

- **Do not add `temp0` to the queue.** Its 900 Kimi K3 requests can never run, because Bedrock
  rejects temperature for Kimi K3.
- **If anything stops,** rerun the same command. Finished calls are never sent again.

---

## 8c. Amendment 3: real case reports published after the models were frozen (≈ $76–90, before 18 Dec)

Design and reasons: docs/amendment_3.md. One command runs every stage; each stage resumes if
interrupted:

1. Kimi K3 marks where each case stops before its diagnosis.
2. Deterministic screening.
3. Extraction (Kimi K3) and verification (GLM-5).
4. Finalisation.
5. Building the items.
6. The panel calls, 10 per case × 6 models.

Run it **after** 8b has finished (it can also run on its own):

```bash
cd clinical-coherence-rl && source .venv/bin/activate
git pull
ccrl freeze --check            # OK (46 files)
ccrl fresh plan                # ~238 reports; expected total about $76
bash scripts/run_detached.sh fresh run
tail -f runs/logs/ccrl-fresh-run-*.log
```

- **Checkpoints in the log:**
  - `[fresh] screen: N usable of 238`;
  - `[fresh] finalize: {...}`;
  - `[fresh] items: {...}`;
  - then the panel progress.
- **If fewer than 60 cases pass,** it stops before the panel. That costs about $15–25 and is not
  an error: send the archive.
- **If anything stops,** rerun `bash scripts/run_detached.sh fresh run`.

When it finishes:

```bash
ccrl status
ccrl export && ccrl archive pack
```

Then upload the new archive and its `.sha256`, as before. Also push the prepared files:

```bash
git add artifacts/frozen/fresh && git commit -m "Amendment 3 prepared cases and items" && git push
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
| `committed` in the log is about 2× the AWS bill | the guard counts Flex calls at full price until Flex billing is confirmed | `ccrl billing confirm-flex --start 2026-11-06` (§8.2) |
| `Only ... GB free on the disk` | the disk check before a job | free space (§8.3), then re-run the same command |
| `WARNING: raw mirror not written` | the disk filled during a run; the ledger still has everything | free space soon; the run continues |
| a detached job is no longer running, no error in the log | the Mac slept (lid closed) or the shell that started it ended | run the same command again from a normal Terminal window |
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
