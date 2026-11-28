#!/usr/bin/env bash
# Local orchestrator for one stage A machine (docs/plan_final.md §3.5). Run once per instance, after
# lambda_setup.sh has finished there:
#
#   (LAMBDA_API_KEY and HF_TOKEN: from this shell, else from the Windows user environment)
#   PY=/e/ccrl-venv/Scripts/python.exe bash scripts/stage_a_local.sh h100 <instance-id> <ip> 3.79 61
#   PY=/e/ccrl-venv/Scripts/python.exe bash scripts/stage_a_local.sh a100 <instance-id> <ip> 1.99 18.5
#
# It starts the machine's queue detached, copies runs/ back every 10 minutes to
# /e/ccrl-data/lambda_runs/stage_a_<machine>/ (DEST_ROOT), and terminates the instance (ours only: lambda_api.py
# refuses any instance not named govind_llm) when the queue writes DONE or STOPPED_*, when the
# queue process dies, or when the hard cap is reached: hours since start x price >= budget + $1.
set -euo pipefail
machine=$1 id=$2 ip=$3 price=$4 budget=$5
PY=${PY:-python}                  # a Python with huggingface_hub (e.g. PY=/e/ccrl-venv/Scripts/python.exe)
# keys saved as Windows user environment variables are read from there if this shell lacks them
for v in LAMBDA_API_KEY HF_TOKEN; do
  if [ -z "${!v:-}" ] && command -v powershell.exe >/dev/null; then
    export "$v=$(powershell.exe -NoProfile -Command "[Environment]::GetEnvironmentVariable('$v','User')" | tr -d '\r')"
  fi
done
[ -n "${LAMBDA_API_KEY:-}" ] || { echo "LAMBDA_API_KEY not set"; exit 1; }
key=${SSH_KEY:-~/.ssh/govind_llm}
dest="${DEST_ROOT:-/e/ccrl-data/lambda_runs}/stage_a_${machine}"
mkdir -p "$dest"
ssh_() { ssh -i "$key" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "ubuntu@$ip" "$@"; }
# interim pulls: logs, evaluations and configs only (no weights); the final pull adds the adapters
pull() { ssh_ "cd ~/clinical-coherence-rl && tar czf - --exclude='*.safetensors' --exclude='*.pt' \
               --exclude='runs/train/*/rollout_adapter' runs" | tar xzf - -C "$dest" || true; }
pull_full() { ssh_ "cd ~/clinical-coherence-rl && tar czf - --exclude='*.pt' \
                    --exclude='runs/train/*/rollout_adapter' runs" | tar xzf - -C "$dest" || true; }
env_vars="PRICE_H=$price BUDGET_USD=$budget"
[ "$machine" = a100 ] && env_vars="$env_vars GPU_MEM=0.45 VLLM_SLEEP=1 NO_SHARED=1"
env_vars="$env_vars ${EXTRA_ENV:-}"       # e.g. EXTRA_ENV="GPU_MEM=0.45 VLLM_SLEEP=1 CHECK_MEM=0.8" on a 40 GB card
ssh_ "cd ~/clinical-coherence-rl && mkdir -p runs/env && (pgrep -f '^bash scripts/[l]ambda_stage_a' >/dev/null || \
      (nohup env $env_vars bash scripts/lambda_stage_a.sh $machine > runs/env/queue_$machine.log 2>&1 < /dev/null &))"
t0=$(date +%s)
unreachable=0
echo "queue started on $machine ($ip); copying to $dest"
while :; do
  sleep 600
  pull
  hours=$("$PY" -c "import time; print((time.time()-$t0)/3600)")
  cost=$("$PY" -c "print(round($hours*$price, 2))")
  state=$(ssh_ "cd ~/clinical-coherence-rl/runs/env && ls DONE STOPPED_* 2>/dev/null | head -1; \
                pgrep -f '^bash scripts/[l]ambda_stage_a' >/dev/null && echo RUNNING || echo DEAD" | tr '\n' ' ' || echo "UNREACHABLE")
  echo "$(date -Is) ~\$$cost  $state  $(tail -1 "$dest/runs/env/queue_$machine.log" 2>/dev/null | cut -c1-120)"
  if [[ "$state" == *UNREACHABLE* ]]; then unreachable=$((unreachable + 1)); else unreachable=0; fi
  # three unreachable checks in a row (30 min): stop paying for a machine we cannot reach
  if [[ "$state" == *DONE* || "$state" == *STOPPED* || "$state" == *DEAD* ]] || [ "$unreachable" -ge 3 ] || \
     "$PY" -c "import sys; sys.exit(0 if $cost >= $budget + 1 else 1)"; then
    pull_full
    echo "terminating $machine ($id): $state  ~\$$cost"
    "$PY" scripts/lambda_api.py terminate "$id" --yes
    # adapter weights only, from this machine (the HF token never goes to the shared instance)
    if [ -n "${HF_TOKEN:-}" ]; then
      "$PY" scripts/upload_adapters.py "$dest/runs" || echo "adapter upload failed: rerun scripts/upload_adapters.py $dest/runs"
    fi
    break
  fi
done
