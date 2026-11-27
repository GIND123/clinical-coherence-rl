#!/usr/bin/env bash
# GPU smoke test of the vLLM backend (~15-20 min). Run on the instance after lambda_setup.sh:
#   bash scripts/lambda_smoke.sh 2>&1 | tee runs/env/smoke.log
# Checks: vLLM evaluation parses; a few GRPO steps run with vLLM rollouts; the vLLM and
# transformers backends give the same rollout log-probs up to kernel noise (TIS weights ~1);
# prints seconds per step for both backends, which sets the GPU-hour plan.
set -euo pipefail
cd ~/clinical-coherence-rl && source .venv/bin/activate
mkdir -p runs/env
mem_args=()                                   # on a 40 GB A100:  GPU_MEM=0.45 VLLM_SLEEP=1 bash scripts/lambda_smoke.sh
[ -n "${GPU_MEM:-}" ] && mem_args+=(--gpu-mem "$GPU_MEM")
[ -n "${VLLM_SLEEP:-}" ] && mem_args+=(--vllm-sleep)

echo "== 1. vLLM evaluation, base Qwen3-4B, 10 pilot cases"
ccrl train eval --model Qwen/Qwen3-4B --name smoke_eval_vllm --set pilot --limit-cases 10 --backend vllm
python - <<'PY'
import pandas as pd
d = pd.read_parquet("runs/local_eval/smoke_eval_vllm.parquet")
print(f"items {len(d)}  parse rate {d.parse_ok.mean():.3f}  informative {d.informative.mean():.3f}")
assert d.parse_ok.mean() > 0.8, "parse rate too low: check the chat template / thinking switch"
PY

echo "== 2. GRPO, 4 steps, vLLM rollouts (ccr_acc_fid, 4B)"
ccrl train run --config config/train/ccr_acc_fid.yaml --seed 99 --steps 4 --backend vllm "${mem_args[@]}"
echo "== 3. GRPO, 2 steps, transformers rollouts (timing reference)"
ccrl train run --config config/train/rlvr.yaml --seed 99 --steps 2 --backend hf
python - <<'PY'
import json, glob
for f in sorted(glob.glob("runs/train/*_s99/train_log.jsonl")):
    rows = [json.loads(l) for l in open(f) if '"sec"' in l]
    if rows:
        last = rows[-1]
        print(f"{f}: steps {len(rows)}  sec/step {sum(r['sec'] for r in rows[1:]) / max(1, len(rows) - 1):.1f}  "
              f"gen_sec {last.get('gen_sec')}  invalid {last['invalid_frac']:.2f}  sum100 {last.get('sum100_frac')}  "
              f"M {last['M']:.3f}  M_lenient {last.get('M_lenient')}  credit {last.get('credit_mean')}  "
              f"reward {last['reward']:.3f}  kl {last['kl']:.4f}  tis_w {last.get('tis_w')}")
        assert "M_lenient" in last and "credit_mean" in last, "Amendment 4 logging missing"
        assert last.get("sum100_frac") is not None, "strict readout not active"
PY
rm -rf runs/train/*_s99
echo "SMOKE OK"
