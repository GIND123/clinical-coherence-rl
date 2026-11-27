#!/usr/bin/env bash
# The decisive pair (docs/training.md "Order of runs"; OSF osf.io/4xmq7): Qwen3-4B, seed 0,
# accuracy-only RLVR vs the coherence reward (CCR+acc+fid), plus the untrained baselines.
# Run on an instance after lambda_setup.sh and lambda_smoke.sh, detached:
#
#   nohup bash scripts/lambda_decisive.sh > runs/env/decisive.log 2>&1 &          # everything, one GPU
#   nohup bash scripts/lambda_decisive.sh prep train:rlvr > ... &                  # split across machines:
#   nohup bash scripts/lambda_decisive.sh train:ccr_acc_fid > ... &                # (copy runs/env/tau_i.json first)
#   bash scripts/lambda_decisive.sh compare                                        # where all evals are
#
# Memory: 80 GB cards need nothing. On a 40 GB A100 set  GPU_MEM=0.45 VLLM_SLEEP=1.
# Re-running skips finished pieces; training resumes from its newest checkpoint (every 25 steps).
set -euo pipefail
cd ~/clinical-coherence-rl && source .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
M=Qwen/Qwen3-4B
mkdir -p runs/env
ev() { [ -f "runs/local_eval/$2.parquet" ] || ccrl train eval --backend vllm --model "$M" --name "$2" "${@:3}" ${1:+--adapter "$1"}; }
mem_args=()
[ -n "${GPU_MEM:-}" ] && mem_args+=(--gpu-mem "$GPU_MEM")
[ -n "${VLLM_SLEEP:-}" ] && mem_args+=(--vllm-sleep)

prep() {
  # pre-specified reward constant: tau_I = 95th pct of the untrained model's pilot retest JS
  ev "" base4b_pilot --set pilot
  [ -f runs/env/tau_i.json ] || ccrl train tau-i --eval runs/local_eval/base4b_pilot.parquet > runs/env/tau_i.json
  # untrained baselines on the confirmatory items: registered decoding, and temperature 0
  ev "" base4b
  ev "" base4b_t0 --temperature 0
}

train() {
  local cfg=$1
  [ -f runs/env/tau_i.json ] || { echo "runs/env/tau_i.json missing: run 'prep' here or copy it from the prep machine"; exit 1; }
  local tau
  tau=$(python -c "import json; print(json.load(open('runs/env/tau_i.json'))['tau_I'])")
  echo "tau_I = $tau  ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))" | tee -a "runs/env/hardware_${cfg}_s0.txt"
  [ -d "runs/train/${cfg}_s0/final" ] || ccrl train run --config "config/train/${cfg}.yaml" --seed 0 --backend vllm \
      --tau-i "$tau" "${mem_args[@]}"
  ev "runs/train/${cfg}_s0/final" "${cfg}_s0"
}

compare() {
  ccrl train compare --ref runs/local_eval/rlvr_s0.parquet   --new runs/local_eval/ccr_acc_fid_s0.parquet
  ccrl train compare --ref runs/local_eval/base4b.parquet    --new runs/local_eval/ccr_acc_fid_s0.parquet
  ccrl train compare --ref runs/local_eval/base4b_t0.parquet --new runs/local_eval/ccr_acc_fid_s0.parquet
}

phases=("$@")
[ ${#phases[@]} -eq 0 ] && phases=(prep train:rlvr train:ccr_acc_fid compare)
for ph in "${phases[@]}"; do
  echo "=== $ph  $(date -Is)"
  case "$ph" in
    prep) prep ;;
    train:*) train "${ph#train:}" ;;
    compare) compare ;;
    *) echo "unknown phase $ph"; exit 2 ;;
  esac
done
echo "DONE: ${phases[*]}"
