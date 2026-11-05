#!/usr/bin/env bash
# Run a long ccrl command so it survives a closed terminal and Mac sleep.
#   bash scripts/run_detached.sh run panel_main
#   tail -f runs/logs/<latest>.log          # watch progress
# Stop it gracefully:  kill -INT <pid>     (in-flight calls finish and are recorded)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/logs
stamp=$(date +%Y%m%d-%H%M%S)
log="runs/logs/ccrl-${1:-cmd}-${2:-}-$stamp.log"
CCRL=".venv/bin/ccrl"
[ -x "$CCRL" ] || CCRL="ccrl"
if command -v caffeinate >/dev/null 2>&1; then
  # macOS: -i prevents idle sleep, -s keeps the machine awake on AC power, -m keeps the disk awake
  nohup caffeinate -ims "$CCRL" "$@" > "$log" 2>&1 < /dev/null &
else
  nohup "$CCRL" "$@" > "$log" 2>&1 < /dev/null &
fi
pid=$!
echo "$pid" > runs/logs/last.pid
echo "started pid $pid -> $log"
echo "watch:  tail -f $log"
