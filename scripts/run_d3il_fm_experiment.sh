#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "dpcc" ]]; then
  if [[ -f /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh ]]; then
    source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
    conda activate dpcc
  elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate dpcc
  else
    echo "Could not find conda. Activate the dpcc environment before running this script." >&2
    exit 1
  fi
fi

export TRAIN_SEEDS="${TRAIN_SEEDS:-0,1,2}"
export TRAIN_EXPS="${TRAIN_EXPS:-avoiding-synthetic,avoiding-synthetic-fm}"
export EVAL_EXPS="${EVAL_EXPS:-$TRAIN_EXPS}"

echo "Generating three-scenario synthetic Avoiding dataset"
python scripts/generate_synthetic_data.py

echo "Auditing synthetic dataset quality"
python scripts/audit_synthetic_data.py

IFS=',' read -ra EXPS_TO_TRAIN <<< "$TRAIN_EXPS"
for exp in "${EXPS_TO_TRAIN[@]}"; do
  exp="${exp//[[:space:]]/}"
  echo "Training ${exp} with seeds ${TRAIN_SEEDS}"
  TRAIN_EXP="${exp}" python scripts/train.py 2>&1 | tee "train_${exp}.log"
done

echo "Evaluating avoiding-synthetic vs avoiding-synthetic-fm"
python scripts/eval.py 2>&1 | tee eval_synthetic_fm.log

echo "Done. Summaries:"
IFS=',' read -ra EXPS_TO_SUMMARIZE <<< "$EVAL_EXPS"
for exp in "${EXPS_TO_SUMMARIZE[@]}"; do
  exp="${exp//[[:space:]]/}"
  RESULT_EXP="${exp}" python scripts/load_results.py 2>&1 | tee "results_${exp}.log"
done
