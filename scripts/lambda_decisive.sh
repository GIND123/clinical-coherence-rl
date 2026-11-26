#!/usr/bin/env bash
# The decisive pair (docs/training.md "Order of runs"; OSF osf.io/4xmq7): Qwen3-4B, seed 0,
# accuracy-only RLVR vs the coherence reward (CCR+acc+fid), plus the untrained baselines.
# Run on the instance after lambda_setup.sh and lambda_smoke.sh, detached:
#   nohup bash scripts/lambda_decisive.sh > runs/env/decisive.log 2>&1 &
# Re-running skips finished pieces (training resumes from its newest checkpoint).
set -euo pipefail
cd ~/clinical-coherence-rl && source .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
M=Qwen/Qwen3-4B
ev() { [ -f "runs/local_eval/$2.parquet" ] || ccrl train eval --backend vllm --model "$M" --name "$2" "${@:3}" ${1:+--adapter "$1"}; }

# 1. pre-specified reward constant: tau_I = 95th pct of the untrained model's pilot retest JS
ev "" base4b_pilot --set pilot
[ -f runs/env/tau_i.json ] || ccrl train tau-i --eval runs/local_eval/base4b_pilot.parquet > runs/env/tau_i.json
TAU=$(python -c "import json; print(json.load(open('runs/env/tau_i.json'))['tau_I'])")
echo "tau_I = $TAU"

# 2. untrained baselines on the confirmatory items (registered decoding, and temperature 0)
ev "" base4b
ev "" base4b_t0 --temperature 0

# 3. the decisive pair
for cfg in rlvr ccr_acc_fid; do
  [ -d "runs/train/${cfg}_s0/final" ] || ccrl train run --config "config/train/${cfg}.yaml" --seed 0 --backend vllm --tau-i "$TAU"
  ev "runs/train/${cfg}_s0/final" "${cfg}_s0"
done

# 4. the comparison that decides whether to continue
ccrl train compare --ref runs/local_eval/rlvr_s0.parquet --new runs/local_eval/ccr_acc_fid_s0.parquet
ccrl train compare --ref runs/local_eval/base4b.parquet  --new runs/local_eval/ccr_acc_fid_s0.parquet
ccrl train compare --ref runs/local_eval/base4b_t0.parquet --new runs/local_eval/ccr_acc_fid_s0.parquet
echo "DECISIVE PAIR DONE"
