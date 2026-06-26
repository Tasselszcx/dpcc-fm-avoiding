#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for multi-GPU machines such as 8x H800.
# Each model/seed is an independent experiment, so use one single-GPU process
# per job instead of DDP.

GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPUS <<< "$GPUS_CSV"
NUM_GPUS="${#GPUS[@]}"

MODE="${MODE:-all}"  # train | eval | all
EXPS_CSV="${EXPS:-avoiding-synthetic,avoiding-synthetic-fm}"
SEEDS_CSV="${SEEDS:-0,1,2}"
SCENES_CSV="${SCENES:-top-right-hard,top-left-hard,both-hard}"
PROJECTION_VARIANTS="${PROJECTION_VARIANTS:-diffuser,dpcc-c-tightened,post_processing}"

LOG_DIR="${LOG_DIR:-run_logs/h800_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

IFS=',' read -ra EXPS <<< "$EXPS_CSV"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra SCENES <<< "$SCENES_CSV"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$NUM_GPUS" ]; do
    wait -n
  done
}

gpu_for_job() {
  local job_idx="$1"
  echo "${GPUS[$((job_idx % NUM_GPUS))]}"
}

echo "GPUs: ${GPUS[*]}"
echo "Mode: $MODE"
echo "Experiments: ${EXPS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Scenes: ${SCENES[*]}"
echo "Projection variants: $PROJECTION_VARIANTS"
echo "Logs: $LOG_DIR"

python scripts/generate_synthetic_data.py | tee "$LOG_DIR/generate_data.log"
python scripts/audit_synthetic_data.py | tee "$LOG_DIR/audit_data.log"

job_idx=0

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
  echo "Starting parallel training jobs"
  for exp in "${EXPS[@]}"; do
    exp="${exp//[[:space:]]/}"
    for seed in "${SEEDS[@]}"; do
      seed="${seed//[[:space:]]/}"
      wait_for_slot
      gpu="$(gpu_for_job "$job_idx")"
      log="$LOG_DIR/train_${exp}_seed${seed}_gpu${gpu}.log"
      echo "GPU $gpu :: train $exp seed $seed"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export TRAIN_EXP="$exp"
        export TRAIN_SEEDS="$seed"
        python scripts/train.py
      ) > "$log" 2>&1 &
      job_idx=$((job_idx + 1))
    done
  done
  wait
  echo "Training jobs complete"
fi

job_idx=0

if [[ "$MODE" == "eval" || "$MODE" == "all" ]]; then
  echo "Starting parallel evaluation jobs"
  for exp in "${EXPS[@]}"; do
    exp="${exp//[[:space:]]/}"
    for seed in "${SEEDS[@]}"; do
      seed="${seed//[[:space:]]/}"
      for scene in "${SCENES[@]}"; do
        scene="${scene//[[:space:]]/}"
        wait_for_slot
        gpu="$(gpu_for_job "$job_idx")"
        log="$LOG_DIR/eval_${exp}_seed${seed}_${scene}_gpu${gpu}.log"
        echo "GPU $gpu :: eval $exp seed $seed scene $scene"
        (
          export CUDA_VISIBLE_DEVICES="$gpu"
          export EVAL_EXPS="$exp"
          export EVAL_SEEDS="$seed"
          export EVAL_HALFSPACE_VARIANTS="$scene"
          export EVAL_PROJECTION_VARIANTS="$PROJECTION_VARIANTS"
          python scripts/eval.py
        ) > "$log" 2>&1 &
        job_idx=$((job_idx + 1))
      done
    done
  done
  wait
  echo "Evaluation jobs complete"

  for exp in "${EXPS[@]}"; do
    exp="${exp//[[:space:]]/}"
    RESULT_EXP="$exp" python scripts/load_results.py 2>&1 | tee "$LOG_DIR/results_${exp}.log"
  done
fi

echo "Done. Logs are in $LOG_DIR"
