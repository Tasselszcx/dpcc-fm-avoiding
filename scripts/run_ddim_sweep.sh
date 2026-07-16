#!/usr/bin/env bash
set -eo pipefail   # NOT -u: conda's MKL activation script references unbound vars

# ---------------------------------------------------------------------------
# DDIM fair-comparison sweep (inference only, NO retraining).
#
# This is the honest counterpart to the FM K-sweep. DDIM subsamples timesteps
# from the *already trained* DDPM K=20 schedule and integrates them
# deterministically (respaced ancestral sampling, eta=0). The question it
# answers is 学姐's todo: does determinism ALONE - without FM's flow objective
# - reproduce the few-step efficiency we see in Flow Matching?
#
# No weights are trained here: every K' loads the same
#   avoiding-synthetic  H8_K20_Dmodels.GaussianDiffusion  checkpoint
# and only overrides the number of deterministic sampling steps K' via
# EVAL_DDIM_STEPS. Results are tagged ddim<K'> so they never clobber the
# stochastic DDPM per-step baseline.
#
# Parallelises over (K x seed x scene) across the given GPUs, one proc per GPU.
# Usage:  GPUS=3,4,5,6,7 bash scripts/run_ddim_sweep.sh
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."
source scripts/env_h800.sh >/dev/null

GPUS_CSV="${GPUS:-3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS_CSV"
NUM_GPUS="${#GPU_ARR[@]}"

KS_CSV="${KS:-20,15,10,5,2}"      # deterministic sampling steps K'
SEEDS_CSV="${SEEDS:-0,1,2}"
SCENES_CSV="${SCENES:-top-right-hard,top-left-hard,both-hard}"
VARIANT="${VARIANT:-dpcc-c-tightened}"
EXP="${EXP:-avoiding-synthetic}"   # DDPM weights
NTRIALS="${NTRIALS:-50}"
DDIM_ETA="${DDIM_ETA:-0.0}"        # 0 = deterministic

IFS=',' read -ra KS <<< "$KS_CSV"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra SCENES <<< "$SCENES_CSV"

LOG_DIR="${LOG_DIR:-run_logs/ddimsweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

echo "GPUs: ${GPU_ARR[*]} | K': ${KS[*]} | seeds: ${SEEDS[*]} | scenes: ${SCENES[*]}"
echo "exp=$EXP variant=$VARIANT eta=$DDIM_ETA n_trials=$NTRIALS | logs: $LOG_DIR"

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
      log="$LOG_DIR/ddim${K}_seed${seed}_${scene}_gpu${gpu}.log"
      echo "GPU $gpu :: K'=$K seed=$seed scene=$scene"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export EVAL_EXPS="$EXP"
        export EVAL_SEEDS="$seed"
        export EVAL_HALFSPACE_VARIANTS="$scene"
        export EVAL_PROJECTION_VARIANTS="$VARIANT"
        export EVAL_DDIM_STEPS="$K"
        export EVAL_DDIM_ETA="$DDIM_ETA"
        export EVAL_SAVE_TAG="ddim${K}"
        export EVAL_N_TRIALS="$NTRIALS"
        python scripts/eval.py
      ) > "$log" 2>&1 &
      job_idx=$((job_idx + 1))
    done
  done
done
wait
echo "DDIM sweep complete. Results tagged ddim<K'> under each seed's results/ dir."
