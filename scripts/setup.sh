#!/usr/bin/env bash
# One-time setup on the machine that runs the Bedrock panel (macOS or Linux).
#   bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
ver=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$ver" in
  3.10|3.11|3.12|3.13|3.14) ;;
  *) echo "Python $ver found; need 3.10+. Try: conda create -n ccrl python=3.12 && conda activate ccrl"; exit 1;;
esac
"$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -r requirements/analysis.txt
pip install -e . --no-deps
pip install pytest
echo
echo "Installed. Activate with:  source .venv/bin/activate"
echo "Next:                      ccrl doctor   then   ccrl rehearse"
