# RUNBOOK — running the Bedrock panel (Track A)

For the person running the paid panel from their own terminal with their own
AWS credentials. Every step gives the exact command, what it costs, how long
it takes, and what "OK" looks like. Nothing in this file needs Python
knowledge beyond copying commands.

> **Hard deadline: no paid call after 18 Dec 2026, 23:59** (the harness
> enforces it). Credit: $1,100; hard cap in the harness: $1,000.
> Expected total spend for priorities A+B: about **$500** (high scenario ≈ $1,000).

---

## 0. Five rules

1. **Never delete `runs/`.** It holds the ledger (every request, response and
   cent). Everything resumes from it. A backup copy is written after every job.
2. **Long jobs run detached:** `bash scripts/run_detached.sh run <job>`. This
   survives a closed terminal and keeps the Mac awake (`caffeinate`).
3. **To stop a job, press Ctrl-C once** (or `kill -INT $(cat runs/logs/last.pid)`).
   In-flight calls finish and are recorded. Re-running the same command later
   continues where it stopped and never re-sends completed requests.
4. **Rehearse first.** `ccrl rehearse` runs the whole pipeline on an offline
   mock in ~10 minutes at zero cost. If it passes on your machine, the paid
   run will behave the same way.
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
python -m pytest -q        # optional: 61 tests, ~1 min
```

If `python3 --version` is below 3.10:
`conda create -n ccrl python=3.12 -y && conda activate ccrl`, then
`PYTHON=python bash scripts/setup.sh`.

AWS: the harness uses your normal AWS credentials (the same ones that made
`aws bedrock-runtime converse` work). If you use a named profile, add
`--profile <name>` right after `ccrl` in every command, e.g.
`ccrl --profile lab doctor`. Region is `us-east-1` (set in `config/models.yaml`).

**Optional but recommended:** in the AWS console → Billing → Budgets, create a
cost budget of $1,000 with e-mail alerts at 25/50/75/90 %. The harness has its
own caps; this is a second, independent alarm.

---

## 2. Models: resolve and probe (≈10 minutes, ≈$0.50)

```bash
ccrl models list          # every model / inference profile your account can see ($0)
ccrl models resolve       # matches the 12 study models to live IDs -> config/models.resolved.yaml ($0)
```

Check the printed table: every panel model needs a `chosen id`. If one says
`None`, enable access to it in the Bedrock console (Model access) and re-run.

```bash
ccrl models probe         # a few tiny real calls per model (~$0.50 total)
```

This writes `config/models.lock.yaml` and prints one line per model, e.g.
`[probe] opus-5.5: OK id=us.anthropic.claude-opus-5-5 effort=output_config/low tier=default cache=True ...`.

* `OK` with no `WARN` — fine.
* `WARN: service tier flex rejected; using default` — fine (costs a bit more; recorded).
* `WARN: effort may be ignored` for Opus or GPT-6 Sol — **tell the team before
  running the effort arm** (the medium/high comparison would be meaningless).
* `FAILED` for a panel model — do not continue; send the printed line to the team.

To re-probe one model: `ccrl models probe --models kimi-k3`.

---

## 3. Verify prices (5 minutes, $0)

Open `config/prices.yaml`. Compare each row with
<https://aws.amazon.com/bedrock/pricing/> (and the Marketplace listing for
Claude / GPT / Grok / Kimi). Correct any number, and set `verified: true` and
`checked: 2026-11-xx` on rows you confirmed. Unverified rows are budgeted with a
25 % safety margin, so this only makes the caps tighter, never looser.

---

## 4. Preprocessing (≈1–2 hours wall time, ≈$10 now + ≈$40 later)

```bash
ccrl prep phrasebank                              # DDXPlus paraphrase bank (~$1)
ccrl prep mcr --stage all --roles pilot,panel     # MedCaseReasoning: 226 cases extract + verify (~$8)
ccrl prep finalize                                # deterministic; prints prepared counts
```

`finalize` must report at least **10 ok pilot** and **150 ok panel** MedCaseReasoning
cases (it prepares spares). Two review files are created for the clinicians
(they can review in parallel with the pilot; the panel does not wait for them):

* `artifacts/frozen/banks/phrasebank_review.csv`
* `artifacts/frozen/medcase/medcase_review.csv`

The training-set extraction (2,150 cases, ≈$35) is needed only for Track B and
can run any time before 18 Dec, e.g. overnight:

```bash
bash scripts/run_detached.sh prep mcr --stage extract --roles train
```

---

## 5. Build, freeze, plan (2 minutes, $0)

```bash
ccrl build items          # every prompt the study will ever send (~13k items)
ccrl freeze               # hashes all inputs -> artifacts/frozen/FREEZE.json
ccrl plan                 # call counts and cost forecast per job
```

Record the freeze hash for the pre-registration:
`shasum -a 256 artifacts/frozen/FREEZE.json`.

---

## 6. Pilot (≈1 hour, ≈$15–30)

```bash
bash scripts/run_detached.sh run pilot
tail -f runs/logs/ccrl-run-pilot-*.log      # Ctrl-C stops watching, not the job
ccrl report pilot                           # after it finishes
```

The report ends with **GO** or **NO-GO** and the reasons. Gates per model:
parse rate ≥ 98 %, refusals ≤ 2 %, truncation ≤ 1 %, timeouts ≤ 2 %,
informative answers ≥ 90 %, parse rate not differing by more than 5 points
across transformations, and the re-forecast of A+B within the $1,000 cap.

On **NO-GO**: send `runs/reports/pilot_report.txt` to the team. Do not start
the panel.

**Billing check (24–48 h after the pilot):**

```bash
ccrl billing reconcile --start 2026-11-07
```

or in the console: Billing → Cost Explorer, group by *Service* and by *Charge
type*. Every model line should be matched by a **Credit** line. If a model is
billed to the card, stop and tell the team.

---

## 7. Pre-register (OSF), then the main panel (≈1–2 days, ≈$250–$600)

Upload `docs/preregistration.md` (with the freeze hash) to OSF **before** this step.

```bash
bash scripts/run_detached.sh run panel_main
ccrl status               # any time: progress and spend per model
```

It runs 450 cases × 10 calls × 11 models = 49,500 calls. If the machine
sleeps, the network drops, or you stop it, simply run the same command again.

---

## 8. Priority B (≈1 day, ≈$200)

Each starts only after `panel_main` is complete:

```bash
bash scripts/run_detached.sh run effort_arm   # Opus 5.5 and GPT-6 Sol at medium/high effort
bash scripts/run_detached.sh run sequential   # two-stage evidence, 7 models
bash scripts/run_detached.sh run distill      # 2,000 GPT-6 Sol answers for a training baseline
```

---

## 9. Priority C (from 10 Dec, only if spend ≤ $700)

```bash
ccrl status                                   # check committed spend
bash scripts/run_detached.sh run astra_ceiling
bash scripts/run_detached.sh run panel_ext
```

The harness refuses priority C before 10 Dec or above $700 committed.

---

## 10. Archive and hand back (before 18 Dec)

```bash
ccrl export               # ledger -> parquet + MANIFEST.json (SHA-256 per file)
ccrl archive pack         # runs/archive/ccrl-archive-<date>.tar.gz + .sha256
ccrl analyze              # first-look tables and figures in runs/reports/analysis/
```

Send the team: the `.tar.gz` and its `.sha256` (shared drive / Zenodo), and the
small frozen files if you cannot push to GitHub:
`config/models.lock.yaml`, `artifacts/frozen/` (whole folder).
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
| `hard cap $1000.00` / `job cap` in the summary | a cap was reached | `ccrl status`; tell the team; do not edit caps without agreement |
| `model_halted` / `ValidationException` | Bedrock rejected a parameter for one model | other models continue; re-run `ccrl models probe --models <key>` and send the output |
| `AccessDeniedException`, `ExpiredToken` | credentials | refresh credentials (`aws sso login` / new keys), re-run the same command |
| many `ThrottlingException` lines | account quota | nothing; the harness backs off and continues |
| `Hard stop ... has passed` | after 18 Dec | expected; no paid calls are possible |
| `requires panel_main to be complete` | order rule | finish the earlier job first |
| laptop slept / Wi-Fi dropped | — | re-run the same command; completed work is never repeated |
| `REHEARSAL` fails | local environment problem | send the last 30 lines of the output |

Questions about a specific request: every attempt is in `runs/ledger.sqlite`
and mirrored to `runs/raw/<job>/<model>.jsonl` (request, response, usage, cost).
