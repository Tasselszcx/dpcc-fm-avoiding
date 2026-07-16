#!/usr/bin/env bash
set -eo pipefail   # NOT -u: conda's MKL activation script references unbound vars

# ---------------------------------------------------------------------------
# FM K-sweep launcher (inference only, NO retraining).
#
# Flow Matching decouples the sampling step count K from the trained weights
# (sampling integrates the learned velocity field with dt = 1/n_timesteps), so
# the same K=20 checkpoint can be evaluated at any K just by overriding
# diffusion.n_timesteps at inference. This script sweeps K over the FM model
# under the *per-step* projection variant (dpcc-c-tightened) and writes each K
# into its own tagged results dir so nothing overwrites the existing K=20 run.
#
# K=20 is intentionally NOT re-run here: the existing per-step FM results
# (halfspace_<scene>/dpcc-c-tightened.npz) already are the K=20 point.
#
# Parallelises over (seed x scene) across the given GPUs, one process per GPU.
# Usage:  GPUS=3,4,5,6,7 bash scripts/run_fm_ksweep.sh
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."
source scripts/env_h800.sh >/dev/null

GPUS_CSV="${GPUS:-3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS_CSV"
NUM_GPUS="${#GPU_ARR[@]}"

KS_CSV="${KS:-15,10,5,2}"          # K values to sweep (K=20 reused from existing run)
SEEDS_CSV="${SEEDS:-0,1,2}"
SCENES_CSV="${SCENES:-top-right-hard,top-left-hard,both-hard}"
VARIANT="${VARIANT:-dpcc-c-tightened}"
EXP="${EXP:-avoiding-synthetic-fm}"
NTRIALS="${NTRIALS:-50}"

IFS=',' read -ra KS <<< "$KS_CSV"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra SCENES <<< "$SCENES_CSV"

LOG_DIR="${LOG_DIR:-run_logs/ksweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

echo "GPUs: ${GPU_ARR[*]} | K: ${KS[*]} | seeds: ${SEEDS[*]} | scenes: ${SCENES[*]}"
echo "exp=$EXP variant=$VARIANT n_trials=$NTRIALS | logs: $LOG_DIR"

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
        export EVAL_FM_NTIMESTEPS="$K"
        export EVAL_SAVE_TAG="k${K}"
        export EVAL_N_TRIALS="$NTRIALS"
        python scripts/eval.py
      ) > "$log" 2>&1 &
      job_idx=$((job_idx + 1))
    done
  done
done
wait
echo "FM K-sweep complete. Results tagged k<K> under each seed's results/ dir."
