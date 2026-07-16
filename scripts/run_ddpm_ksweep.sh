#!/usr/bin/env bash
set -eo pipefail   # NOT -u: conda's MKL activation script references unbound vars

# ---------------------------------------------------------------------------
# DDPM K-sweep evaluation launcher (per-K trained models).
#
# Unlike FM (one model, K set at inference), each DDPM K has its OWN trained
# weights. This evaluates each K's checkpoint via EVAL_NDIFF=<K> (which rewrites
# loadpath/savepath to H<H>_K<K>_...). Set EVAL_HORIZON to pick the H=16 models.
#
# Parallelises over (K x seed x scene) across the given GPUs, one process/GPU.
# Usage: GPUS=0,1,2,3,4,5,6,7 EVAL_HORIZON=16 EXP=avoiding-d3il bash scripts/run_ddpm_ksweep.sh
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."
source scripts/env_h800.sh >/dev/null

GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS_CSV"
NUM_GPUS="${#GPU_ARR[@]}"

KS_CSV="${KS:-4,8,16,32}"
SEEDS_CSV="${SEEDS:-0,1,2}"
SCENES_CSV="${SCENES:-top-right-hard,top-left-hard,both-hard}"
VARIANT="${VARIANT:-dpcc-c-tightened}"
EXP="${EXP:-avoiding-d3il}"
NTRIALS="${NTRIALS:-100}"
HORIZON="${EVAL_HORIZON:-16}"

IFS=',' read -ra KS <<< "$KS_CSV"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra SCENES <<< "$SCENES_CSV"

LOG_DIR="${LOG_DIR:-run_logs/ddpm_ksweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

echo "GPUs: ${GPU_ARR[*]} | K: ${KS[*]} | seeds: ${SEEDS[*]} | scenes: ${SCENES[*]}"
echo "exp=$EXP H=$HORIZON variant=$VARIANT n_trials=$NTRIALS | logs: $LOG_DIR"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$NUM_GPUS" ]; do
    wait -n
  done
}

job_idx=0
for K in "${KS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for scene in "${SCENES[@]}"; do
      wait_for_slot
      gpu="${GPU_ARR[$((job_idx % NUM_GPUS))]}"
      log="$LOG_DIR/k${K}_seed${seed}_${scene}_gpu${gpu}.log"
      echo "GPU $gpu :: K=$K seed=$seed scene=$scene"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export EVAL_EXPS="$EXP"
        export EVAL_SEEDS="$seed"
        export EVAL_HALFSPACE_VARIANTS="$scene"
        export EVAL_PROJECTION_VARIANTS="$VARIANT"
        export EVAL_NDIFF="$K"
        export EVAL_HORIZON="$HORIZON"
        export EVAL_SAVE_TAG="h${HORIZON}_k${K}"
        export EVAL_N_TRIALS="$NTRIALS"
        python scripts/eval.py
      ) > "$log" 2>&1 &
      job_idx=$((job_idx + 1))
    done
  done
done
wait
echo "DDPM K-sweep (H=$HORIZON) complete. Results tagged h${HORIZON}_k<K> under each seed's results/ dir."
