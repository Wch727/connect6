#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Explicit opt-in: this script starts substantial training. Never run for installation.
PYTHON="${PYTHON:-python}"
ITERATIONS="${ITERATIONS:-50}"
if [[ ! -f checkpoints/sl_model.pt ]]; then
    "$PYTHON" -X utf8 gen_data.py
    "$PYTHON" -X utf8 train_sl.py
fi
# Identical evaluation settings; training assistance is the variable.
"$PYTHON" -X utf8 train_experiment.py --log-name baseline.csv --no-rule-based --eval-no-rule-based --iterations "$ITERATIONS"
"$PYTHON" -X utf8 train_experiment.py --log-name rules_only.csv --eval-no-rule-based --iterations "$ITERATIONS"
"$PYTHON" -X utf8 train_experiment.py --log-name bc_only.csv --pretrained checkpoints/sl_model.pt --no-rule-based --eval-no-rule-based --iterations "$ITERATIONS"
"$PYTHON" -X utf8 train_experiment.py --log-name ours.csv --pretrained checkpoints/sl_model.pt --eval-no-rule-based --iterations "$ITERATIONS"
"$PYTHON" -X utf8 plot_results.py
