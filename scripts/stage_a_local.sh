#!/usr/bin/env bash
# Local orchestrator for one stage A machine (docs/plan_final.md §3.5). Run once per instance, after
# lambda_setup.sh has finished there:
#
#   export LAMBDA_API_KEY=...        # in this shell only; never written to disk
#   bash scripts/stage_a_local.sh h100 <instance-id> <ip> 3.79 60
#   bash scripts/stage_a_local.sh a100 <instance-id> <ip> 1.99 18
#
# It starts the machine's queue detached, copies runs/ back every 10 minutes to
# $CCRL_DATA/lambda_runs/stage_a_<machine>/, and terminates the instance (ours only: lambda_api.py
# refuses any instance not named govind_llm) when the queue writes DONE or STOPPED_*, when the
# queue process dies, or when the hard cap is reached: hours since start x price >= budget + $2.
set -euo pipefail
machine=$1 id=$2 ip=$3 price=$4 budget=$5
# keys saved as Windows user environment variables are read from there if this shell lacks them
for v in LAMBDA_API_KEY HF_TOKEN; do
  if [ -z "${!v:-}" ] && command -v powershell.exe >/dev/null; then
    export "$v=$(powershell.exe -NoProfile -Command "[Environment]::GetEnvironmentVariable('$v','User')" | tr -d '\r')"
  fi
done
[ -n "${LAMBDA_API_KEY:-}" ] || { echo "LAMBDA_API_KEY not set"; exit 1; }
key=${SSH_KEY:-~/.ssh/govind_llm}
dest="${CCRL_DATA:-E:/ccrl-data}/lambda_runs/stage_a_${machine}"
mkdir -p "$dest"
ssh_() { ssh -i "$key" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "ubuntu@$ip" "$@"; }
pull() { ssh_ "cd ~/clinical-coherence-rl && tar czf - runs --exclude='runs/train/*/rollout_adapter' \
               --exclude='runs/train/*/checkpoints/*/trainer_state.pt'" | tar xzf - -C "$dest" || true; }
env_vars="PRICE_H=$price BUDGET_USD=$budget"
[ "$machine" = a100 ] && env_vars="$env_vars GPU_MEM=0.45 VLLM_SLEEP=1 NO_SHARED=1"
ssh_ "cd ~/clinical-coherence-rl && mkdir -p runs/env && (pgrep -f lambda_stage_a.sh >/dev/null || \
      (nohup env $env_vars bash scripts/lambda_stage_a.sh $machine > runs/env/queue_$machine.log 2>&1 &))"
t0=$(date +%s)
echo "queue started on $machine ($ip); copying to $dest"
while :; do
  sleep 600
  pull
  hours=$(python -c "import time; print((time.time()-$t0)/3600)")
  cost=$(python -c "print(round($hours*$price, 2))")
  state=$(ssh_ "cd ~/clinical-coherence-rl/runs/env && ls DONE STOPPED_* 2>/dev/null | head -1; \
                pgrep -f lambda_stage_a.sh >/dev/null && echo RUNNING || echo DEAD" | tr '\n' ' ' || echo "UNREACHABLE")
  echo "$(date -Is) ~\$$cost  $state  $(tail -1 "$dest/runs/env/queue_$machine.log" 2>/dev/null | cut -c1-120)"
  if [[ "$state" == *DONE* || "$state" == *STOPPED* || "$state" == *DEAD* ]] || \
     python -c "import sys; sys.exit(0 if $cost >= $budget + 2 else 1)"; then
    pull
    echo "terminating $machine ($id): $state  ~\$$cost"
    python scripts/lambda_api.py terminate "$id" --yes
    # adapter weights only, from this machine (the HF token never goes to the shared instance)
    if [ -n "${HF_TOKEN:-}" ]; then
      python scripts/upload_adapters.py "$dest/runs" || echo "adapter upload failed: rerun scripts/upload_adapters.py $dest/runs"
    fi
    break
  fi
done
