#!/usr/bin/env bash
set -euo pipefail

export TRAIN_EXP="${TRAIN_EXP:-avoiding-synthetic-fm}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-0}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000}"
export TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-500}"
export EVAL_EXPS="${EVAL_EXPS:-$TRAIN_EXP}"
export EVAL_SEEDS="${EVAL_SEEDS:-$TRAIN_SEEDS}"
export EVAL_N_TRIALS="${EVAL_N_TRIALS:-5}"
export EVAL_DIFFUSION_EPOCH="${EVAL_DIFFUSION_EPOCH:-$TRAIN_STEPS}"
export EVAL_SAVE_TAG="${EVAL_SAVE_TAG:-pilot${TRAIN_STEPS}}"
export EVAL_PROJECTION_VARIANTS="${EVAL_PROJECTION_VARIANTS:-diffuser,dpcc-c-tightened}"

python scripts/generate_synthetic_data.py
python scripts/audit_synthetic_data.py

echo "Training $TRAIN_EXP seeds=$TRAIN_SEEDS steps=$TRAIN_STEPS"
python scripts/train.py 2>&1 | tee "train_${TRAIN_EXP}_${TRAIN_STEPS}.log"

echo "Evaluating $EVAL_EXPS trials=$EVAL_N_TRIALS checkpoint=$EVAL_DIFFUSION_EPOCH"
python scripts/eval.py 2>&1 | tee "eval_${TRAIN_EXP}_${TRAIN_STEPS}.log"
