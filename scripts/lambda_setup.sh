#!/usr/bin/env bash
# One-time setup of a Lambda GPU instance for Track B (Paper 2). Run ON the instance:
#   curl -fsSL https://raw.githubusercontent.com/GIND123/clinical-coherence-rl/main/scripts/lambda_setup.sh | bash
# or, after cloning:  bash scripts/lambda_setup.sh
# Creates ~/clinical-coherence-rl/.venv with vLLM (which brings its matching torch), the
# training stack and this package; downloads DDXPlus and the base models; runs the CPU checks.
set -euo pipefail
REPO=https://github.com/GIND123/clinical-coherence-rl.git
cd ~
[ -d clinical-coherence-rl ] || git clone "$REPO"
cd clinical-coherence-rl && git pull --ff-only
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install vllm                                   # pulls the torch build vLLM is compiled against
pip install -r requirements/train.txt
pip install -e . --no-deps
mkdir -p runs/env
pip freeze > "runs/env/pip-freeze-$(date +%Y%m%d-%H%M%S).txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > runs/env/gpu.txt
python -c "import torch, vllm, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| vllm', vllm.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ('Qwen/Qwen3-4B', 'Qwen/Qwen3-8B')]"

ccrl data download ddxplus                         # ~180 MB, md5-checked
ccrl train build-pool                              # 60k training patients, held-out pathologies excluded
ccrl train reward-suite                            # must print 10 PASS
ccrl freeze --check                                # the evaluation items are exactly the frozen ones
python -m pytest -q tests/test_train_smoke.py tests/test_vllm_backend.py
echo "SETUP OK"
