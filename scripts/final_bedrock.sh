#!/usr/bin/env bash
# The last Bedrock runs (docs/amendment_5.md §A; RUNBOOK §8b-8c), in the planned order, unattended:
#   1. temperature 0 on the full panel and the replication sample (5 models)
#   2. post-freeze real case reports: preparation, then the 6-model panel
#   3. the contamination probe, last (Kimi K3 is most of its cost)
#   4. export and pack the archive
# Spend is capped by config/budget.yaml: at most $200 on top of the $561.55 already spent.
# Each step resumes where it stopped; finished calls are never sent again, so rerunning is safe.
#
#   git pull && bash scripts/final_bedrock.sh start      # starts detached, survives a closed terminal
#   tail -f runs/logs/final_bedrock.log                  # watch
set -uo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = start ]; then
  mkdir -p runs/logs
  if command -v caffeinate >/dev/null 2>&1; then
    nohup caffeinate -ims bash "$0" > runs/logs/final_bedrock.log 2>&1 < /dev/null &
  else
    nohup bash "$0" > runs/logs/final_bedrock.log 2>&1 < /dev/null &
  fi
  echo "started (pid $!). Watch: tail -f runs/logs/final_bedrock.log"
  exit 0
fi
[ -f .venv/bin/activate ] && source .venv/bin/activate
echo "=== start $(date)"
ccrl freeze --check || { echo "STOP: freeze check failed. Run 'git pull' first, then rerun."; exit 1; }
ccrl status
for step in "queue temp0_panel temp0_ext" "fresh run" "contamination run"; do
  echo "=== ccrl $step   $(date)"
  ccrl $step || echo "!!! 'ccrl $step' exited with code $? - continuing; rerun this script later to resume it"
done
echo "=== export and archive   $(date)"
ccrl status
ccrl export && ccrl archive pack
echo "=== FINISHED $(date)"
echo "Next: upload the newest runs/archive/*.tar.gz and its .sha256 to the shared folder, then:"
echo "  git add artifacts/frozen/fresh && git commit -m 'Amendment 3 prepared cases and items' && git push"
