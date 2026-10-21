#!/usr/bin/env bash
# Train and evaluate all three subtasks for English and Spanish.
# Usage: bash scripts/run_all.sh <data_dir> [text_col] [label_col]
# Expects <data_dir>/{en,es}/ with task{N}_train.csv (or train.csv), optional dev/test files.
set -euo pipefail

DATA=${1:?"data directory required"}
TEXT_COL=${2:-text}
LABEL_COL=${3:-label}

pick() {  # pick <dir> <task> <split>
  if [[ -f "$1/task$2_$3.csv" ]]; then echo "$1/task$2_$3.csv"
  elif [[ -f "$1/$3.csv" ]]; then echo "$1/$3.csv"; fi
}

for LANG in en es; do
  for TASK in 1 2 3; do
    DIR="$DATA/$LANG"
    TRAIN=$(pick "$DIR" "$TASK" train)
    [[ -z "$TRAIN" ]] && { echo "skip task $TASK ($LANG): no train file in $DIR"; continue; }
    DEV=$(pick "$DIR" "$TASK" dev); TEST=$(pick "$DIR" "$TASK" test)
    echo ">> Task $TASK · $LANG"
    python -m ssd.train --task "$TASK" --lang "$LANG" --train "$TRAIN" \
      ${DEV:+--dev "$DEV"} ${TEST:+--test "$TEST"} \
      --text-col "$TEXT_COL" --label-col "$LABEL_COL"
  done
done
echo "Done. Results in outputs/"
