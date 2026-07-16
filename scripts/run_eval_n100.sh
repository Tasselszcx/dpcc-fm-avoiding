#!/usr/bin/env bash
set -eo pipefail   # NOT -u: conda's MKL activation script references unbound vars

# ---------------------------------------------------------------------------
# Unified n_trials=100 three-way evaluation (inference only).
#
# Re-evaluates all three step-count families at a consistent n_trials=100 so
# FM / DDIM / native-DDPM are directly comparable at the SAME test workload:
#   FM     : avoiding-synthetic-fm  K via EVAL_FM_NTIMESTEPS   tag fmk<K>n100
#   DDIM   : avoiding-synthetic     K via EVAL_DDIM_STEPS(eta0) tag ddim<K>n100
#   DDPM   : avoiding-synthetic     K via EVAL_NDIFF (native)   tag retrain<K>n100
#
# The n100 tag suffix keeps these results in their own dirs so the earlier
# n_trials=50 sweeps (and the untagged K=20 baselines referenced elsewhere)
# are never overwritten. All K use the SAME 100 test scenarios (arange(100)),
# so every cell is a paired comparison.
#
# Usage:  GPUS=0,1,2,3,4,5,6,7 bash scripts/run_eval_n100.sh
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."
source scripts/env_h800.sh >/dev/null

GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS_CSV"
NUM_GPUS="${#GPU_ARR[@]}"

KS_CSV="${KS:-1,2,3,4,5,6,8,10,15,20}"
SEEDS_CSV="${SEEDS:-0,1,2}"
SCENES_CSV="${SCENES:-top-right-hard,top-left-hard,both-hard}"
METHODS_CSV="${METHODS:-fm,ddim,ddpm}"
VARIANT="${VARIANT:-dpcc-c-tightened}"
NTRIALS="${NTRIALS:-100}"

IFS=',' read -ra KS <<< "$KS_CSV"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra SCENES <<< "$SCENES_CSV"
IFS=',' read -ra METHODS <<< "$METHODS_CSV"

LOG_DIR="${LOG_DIR:-run_logs/evaln100_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

echo "GPUs: ${GPU_ARR[*]} | methods: ${METHODS[*]} | K: ${KS[*]} | seeds: ${SEEDS[*]}"
echo "variant=$VARIANT n_trials=$NTRIALS | logs: $LOG_DIR"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$NUM_GPUS" ]; do
    wait -n
  done
}

job_idx=0
for method in "${METHODS[@]}"; do
  for K in "${KS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for scene in "${SCENES[@]}"; do
        wait_for_slot
        gpu="${GPU_ARR[$((job_idx % NUM_GPUS))]}"
        log="$LOG_DIR/${method}_K${K}_seed${seed}_${scene}_gpu${gpu}.log"
        echo "GPU $gpu :: $method K=$K seed=$seed scene=$scene"
        (
          export CUDA_VISIBLE_DEVICES="$gpu"
          export EVAL_SEEDS="$seed"
          export EVAL_HALFSPACE_VARIANTS="$scene"
          export EVAL_PROJECTION_VARIANTS="$VARIANT"
          export EVAL_N_TRIALS="$NTRIALS"
          case "$method" in
            fm)
              export EVAL_EXPS="avoiding-synthetic-fm"
              export EVAL_FM_NTIMESTEPS="$K"
              export EVAL_SAVE_TAG="fmk${K}n100"
              ;;
            ddim)
              export EVAL_EXPS="avoiding-synthetic"
              export EVAL_DDIM_STEPS="$K"
              export EVAL_DDIM_ETA="0.0"
              export EVAL_SAVE_TAG="ddim${K}n100"
              ;;
            ddpm)
              export EVAL_EXPS="avoiding-synthetic"
              export EVAL_NDIFF="$K"
              export EVAL_SAVE_TAG="retrain${K}n100"
              ;;
          esac
          python scripts/eval.py
        ) > "$log" 2>&1 &
        job_idx=$((job_idx + 1))
      done
    done
  done
done
wait
echo "n100 three-way eval complete. Tags: fmk<K>n100 / ddim<K>n100 / retrain<K>n100."
