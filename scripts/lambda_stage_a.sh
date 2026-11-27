#!/usr/bin/env bash
# Stage A of Paper 2 (docs/plan_final.md §3.5): one job queue per machine, run detached after
# lambda_setup.sh. Each phase is skipped if its output exists, so a rerun resumes.
#
#   H100:  PRICE_H=3.79 BUDGET_USD=60 nohup bash scripts/lambda_stage_a.sh h100 > runs/env/queue_h100.log 2>&1 &
#   A100:  PRICE_H=1.99 BUDGET_USD=18 GPU_MEM=0.45 VLLM_SLEEP=1 NO_SHARED=1 nohup bash scripts/lambda_stage_a.sh a100 > runs/env/queue_a100.log 2>&1 &
#   or any phases:  bash scripts/lambda_stage_a.sh train:rlvr:1 post:rlvr:1
#
# Phases
#   check                    GPU equivalence + timing of the shared-prompt pass (fallback: --no-shared)
#   prep                     untrained-model evaluations: pilot, confirmatory (T 0.7 and T 0), general, fresh
#   train:<cfg>:<seed>[:N]   GRPO (N = stop after N steps; a later train phase resumes)
#   gonogo:<cfg>:<seed>      pilot eval of the step-50 adapter + the go/no-go rule; a no-go stops the queue
#   post:<cfg>:<seed>        evaluations of the final adapter (+ milestone pilot evals if MILESTONES=1)
#   sft:<cfg>:<seed>         SFT baseline (targets generated with vLLM), then post
# Budget guard: before each phase, (hours since queue start) x PRICE_H must be below BUDGET_USD
# minus the phase's own estimate; otherwise the queue stops and writes runs/env/STOPPED_BUDGET.
# The local orchestrator copies runs/ back and terminates the instance when DONE or STOPPED_* appears.
set -euo pipefail
cd ~/clinical-coherence-rl && source .venv/bin/activate
M=Qwen/Qwen3-4B
mkdir -p runs/env runs/local_eval
START=runs/env/queue_start
[ -f "$START" ] || date +%s > "$START"
PRICE_H=${PRICE_H:-3.79}
BUDGET_USD=${BUDGET_USD:-1000}
mem_args=()
[ -n "${GPU_MEM:-}" ] && mem_args+=(--gpu-mem "$GPU_MEM")
[ -n "${VLLM_SLEEP:-}" ] && mem_args+=(--vllm-sleep)
shared_opts() {  # decided when each training run starts (the check may have run earlier in this queue)
  if [ -f runs/env/NO_SHARED ] || [ -n "${NO_SHARED:-}" ]; then echo "--no-shared"; return; fi   # A100: unshared
  [ -f runs/env/shared_check.json ] && python -c "import json; t=json.load(open('runs/env/shared_check.json')).get('shared_micro_batch_tokens'); print(f'--shared-tokens {t}' if t else '')"
  return 0
}

spent() { python -c "import time; print(round((time.time()-$(cat $START))/3600*$PRICE_H, 2))"; }
guard() {  # $1 = estimated $ of the next phase
  local s; s=$(spent)
  if python -c "import sys; sys.exit(0 if $s + $1 <= $BUDGET_USD else 1)"; then
    echo "[budget] spent \$$s of \$$BUDGET_USD; next phase ~\$$1"
  else
    echo "[budget] STOP: spent \$$s + next ~\$$1 > \$$BUDGET_USD" | tee runs/env/STOPPED_BUDGET; exit 0
  fi
}
ev() { [ -f "runs/local_eval/$2.parquet" ] || ccrl train eval --backend vllm --model "$M" --name "$2" "${@:3}" ${1:+--adapter "$1"}; }
gen() { [ -f "runs/local_eval/general_$2.parquet" ] || ccrl train general --model "$M" --name "$2" ${1:+--adapter "$1"} \
          ${GPU_MEM:+--gpu-mem "$GPU_MEM"}; }
fresh() { [ -f artifacts/frozen/fresh/items/sets.json ] && ev "$1" "$2_fresh" --set fresh || true; }

check() {
  [ -f runs/env/shared_check.json ] && return 0
  if ! python -m ccrl.train.check_shared --config config/train/ccr_acc_fid.yaml --out runs/env/shared_check.json; then
    echo "shared-prompt check FAILED: training falls back to the unshared pass" | tee runs/env/NO_SHARED
  fi
}

prep() {
  ev "" base4b_pilot --set pilot
  ev "" base4b
  ev "" base4b_t0 --temperature 0
  gen "" base4b
  fresh "" base4b
}

finished() {  # cfg seed [steps]: the log reached `steps` (default: the config's) or the safeguard stopped the run
  python - "$1" "$2" "${3:-}" <<'PY'
import json, sys
from pathlib import Path
from ccrl.train.grpo import load_config
cfg, seed, steps = sys.argv[1], sys.argv[2], sys.argv[3]
want = int(steps) if steps else int(load_config(f"config/train/{cfg}.yaml").get("steps", 300))
log = Path(f"runs/train/{cfg}_s{seed}/train_log.jsonl")
rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()] if log.exists() else []
last = max((r["step"] for r in rows if "sec" in r), default=0)
stopped = any(r.get("event") == "early_stop_M_collapse" for r in rows)
sys.exit(0 if (last >= want or stopped) and Path(f"runs/train/{cfg}_s{seed}/final").is_dir() else 1)
PY
}

train() {  # cfg seed [steps]
  local cfg=$1 seed=$2 steps=${3:-}
  if finished "$cfg" "$seed" "$steps"; then echo "already finished: ${cfg}_s${seed} ${steps}"; return 0; fi
  echo "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" > "runs/env/hardware_${cfg}_s${seed}.txt"
  ccrl train run --config "config/train/${cfg}.yaml" --seed "$seed" --backend vllm ${steps:+--steps "$steps"} \
      "${mem_args[@]}" $(shared_opts)
}

gonogo() {  # cfg seed
  local cfg=$1 seed=$2 run="runs/train/$1_s$2"
  local ck="$run/checkpoints/step_0050"
  [ -d "$ck" ] || { echo "no step-50 checkpoint in $run" | tee runs/env/STOPPED_GONOGO; exit 0; }
  ev "$ck" "${cfg}_s${seed}_step0050_pilot" --set pilot
  if ! ccrl train gonogo --run "$run" --pilot "runs/local_eval/${cfg}_s${seed}_step0050_pilot.parquet" \
       --base runs/local_eval/base4b_pilot.parquet; then
    echo "NO-GO at step 50 of $run (see $run/gonogo.json)" | tee runs/env/STOPPED_GONOGO; exit 0
  fi
}

post() {  # cfg seed
  local cfg=$1 seed=$2 a="runs/train/$1_s$2/final"
  [ -d "$a" ] || { echo "no adapter $a"; return 0; }
  ev "$a" "${cfg}_s${seed}"
  ev "$a" "${cfg}_s${seed}_t0" --temperature 0
  gen "$a" "${cfg}_s${seed}"
  fresh "$a" "${cfg}_s${seed}"
  if [ "${MILESTONES:-0}" = 1 ]; then
    for ck in runs/train/${cfg}_s${seed}/checkpoints/step_*; do
      [ -d "$ck" ] && ev "$ck" "${cfg}_s${seed}_$(basename "$ck" | tr -d _)_pilot" --set pilot
    done
  fi
}

sft() {  # cfg seed
  local cfg=$1 seed=$2
  [ -d "runs/train/${cfg}_s${seed}/final" ] || ccrl train run --config "config/train/${cfg}.yaml" --seed "$seed" \
      --backend vllm "${mem_args[@]}"
  post "$cfg" "$seed"
}

# per-machine stage A queues (plan_final.md §3.5): phase@estimated $ (50 s/step on the H100)
H100_Q=(check@1 prep@2 train:ccr_acc_fid:0:50@4 gonogo:ccr_acc_fid:0@0.5 train:ccr_acc_fid:0@13 post:ccr_acc_fid:0@1.5
        train:brier_only:0@16 post:brier_only:0@1 train:mech_team_lenient:0@3.5 post:mech_team_lenient:0@1
        train:mech_registered_strict:0@3.5 post:mech_registered_strict:0@1 train:ccr_acc_fid:1@16 post:ccr_acc_fid:1@1.5
        train:ccr_mcr:0@14 post:ccr_mcr:0@1 train:rlvr_mcr:0@7 post:rlvr_mcr:0@1)
# (the real-case pair at the end runs only if the budget guard finds the money: Brier-only is
#  predicted to be stopped early by the safeguard, which frees about $14)
A100_Q=(train:rlvr:0@6 post:rlvr:0@1 sft:bct:0@2 sft:sft_selfcons:0@2.5 train:rlvr:1@6 post:rlvr:1@1)

phases=("$@")
case "${phases[0]:-}" in
  h100) phases=("${H100_Q[@]}") ;;
  a100) phases=("${A100_Q[@]}") ;;
esac
for entry in "${phases[@]}"; do
  ph=${entry%@*}
  est=0
  [[ "$entry" == *@* ]] && est=${entry##*@}
  IFS=: read -r name a1 a2 a3 <<< "$ph"
  echo "=== $ph  $(date -Is)"
  guard "$est"
  case "$name" in
    check) check ;;
    prep) prep ;;
    train) train "$a1" "$a2" "${a3:-}" ;;
    gonogo) gonogo "$a1" "$a2" ;;
    post) if [ "$a1" = ccr_acc_fid ]; then MILESTONES=1 post "$a1" "$a2"; else post "$a1" "$a2"; fi ;;
    sft) sft "$a1" "$a2" ;;
    *) echo "unknown phase $ph"; exit 2 ;;
  esac
done
echo "spent ~\$$(spent) on this machine" | tee runs/env/DONE
