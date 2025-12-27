#!/usr/bin/env bash
set -euo pipefail

# Background runner for XGBoost training with logs.
#
# Example:
#   bash /home/shadeform/models/vista-3d/scripts/start_xgb_training_bg.sh \
#     --data /home/shadeform/models/vista-3d/outputs/ml/radiomics_ml_table.csv \
#     --label egfr_mutated \
#     --out-dir /home/shadeform/models/vista-3d/outputs/ml/models/egfr_mutated

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 --data <ml_table.csv> --label <label_col> --out-dir <output_dir> [extra train_xgboost.py args...]" >&2
  exit 2
fi

VENV="/home/shadeform/.venvs/vista-3d/bin/activate"
if [[ -f "$VENV" ]]; then
  # shellcheck disable=SC1090
  source "$VENV"
fi

LOG_DIR="/home/shadeform/models/vista-3d/outputs/ml/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/xgb_${TS}.log"

nohup python -u /home/shadeform/models/vista-3d/scripts/train_xgboost.py "$@" >"$LOG" 2>&1 &
PID=$!
echo "Started XGBoost training (pid=$PID). Log: $LOG"


