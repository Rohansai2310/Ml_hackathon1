#!/usr/bin/env bash
# Run this in VS Code's integrated terminal. Reuses the already fitted forest.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    source ../.venv/bin/activate
fi
export PYTHONPATH="$repo_root/src"
export MPLCONFIGDIR=/tmp/pathsnap-mpl
run_name="${1:-evaluator_test_rf_hmm_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    printf 'Run name may contain letters, digits, underscores and hyphens only.\n' >&2
    exit 2
fi
set -x
python -m pytest -q tests
python -m pathsnap.evaluate_forest \
    --model-dir outputs/training/random_forest_selected_001 \
    --data-root .. \
    --output "outputs/evaluation/$run_name" \
    --with-hmm
python scripts/report_evaluator_forest.py \
    --model-dir outputs/training/random_forest_selected_001 \
    --evaluation-dir "outputs/evaluation/$run_name" \
    --archive "outputs/experiments/$run_name"
