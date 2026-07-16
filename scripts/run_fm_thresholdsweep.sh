#!/usr/bin/env bash
set -eo pipefail   # NOT -u: conda's MKL activation script references unbound vars

# ---------------------------------------------------------------------------
# FM projection-frequency (threshold) sweep -- inference only, NO retraining.
#
# Answers CZ-3: "projection percentage vs safety/cost" trade-off figure.
#
# The projector only fires when the (FM) integration time t_current satisfies
#   t_current >= 1 - diffusion_timestep_threshold
# so `threshold` is literally the fraction of the *late* integration window in
# which projection is active (threshold=1.0 -> project every step; 0.2 -> only
# the last 20% of steps). We encode the threshold in the projection-variant
# name via the `lateprojNN` suffix (eval.py parses NN/100 -> threshold), which
# ALSO becomes the npz filename, so nothing overwrites the K-sweep runs.
#
# Fixed FM checkpoint (K=20 weights), swept over:
#   K         in {20,15,10,5}                      (EVAL_FM_NTIMESTEPS override)
#   threshold in {1.0,0.7,0.5,0.3,0.2,0.1}         (lateproj{100,70,50,30,20,10})
# 4 x 6 = 24 configs x 3 seeds x 3 scenes = 216 jobs, one process per GPU.
#
# NOTE: at small K several thresholds collapse to the same #projections
# (e.g. K=5 with th0.2 and th0.1 both project once). This is expected; we plot
# against the threshold ratio and accept the overlap (user's choice).
#
# Usage:  GPUS=0,1,2,3,4,5,6,7 bash scripts/run_fm_thresholdsweep.sh
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."
source scripts/env_h800.sh >/dev/null

GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS_CSV"
NUM_GPUS="${#GPU_ARR[@]}"

KS_CSV="${KS:-20,15,10,5}"
# threshold percentages -> lateprojNN suffix (NN/100 = diffusion_timestep_threshold)
THS_CSV="${THS:-100,70,50,30,20,10}"
SEEDS_CSV="${SEEDS:-0,1,2}"
SCENES_CSV="${SCENES:-top-right-hard,top-left-hard,both-hard}"
BASE_VARIANT="${BASE_VARIANT:-dpcc-c-tightened}"
EXP="${EXP:-avoiding-synthetic-fm}"
NTRIALS="${NTRIALS:-100}"

IFS=',' read -ra KS <<< "$KS_CSV"
IFS=',' read -ra THS <<< "$THS_CSV"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra SCENES <<< "$SCENES_CSV"

LOG_DIR="${LOG_DIR:-run_logs/threshsweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

echo "GPUs: ${GPU_ARR[*]} | K: ${KS[*]} | thresholds%: ${THS[*]}"
echo "seeds: ${SEEDS[*]} | scenes: ${SCENES[*]}"
echo "exp=$EXP base_variant=$BASE_VARIANT n_trials=$NTRIALS | logs: $LOG_DIR"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$NUM_GPUS" ]; do
    wait -n
  done
}

job_idx=0
for K in "${KS[@]}"; do
  for TH in "${THS[@]}"; do
    variant="${BASE_VARIANT}-lateproj${TH}"
    for seed in "${SEEDS[@]}"; do
      for scene in "${SCENES[@]}"; do
        wait_for_slot
        gpu="${GPU_ARR[$((job_idx % NUM_GPUS))]}"
        log="$LOG_DIR/k${K}_th${TH}_seed${seed}_${scene}_gpu${gpu}.log"
        echo "GPU $gpu :: K=$K th=${TH}% seed=$seed scene=$scene variant=$variant"
        (
          export CUDA_VISIBLE_DEVICES="$gpu"
          export EVAL_EXPS="$EXP"
          export EVAL_SEEDS="$seed"
          export EVAL_HALFSPACE_VARIANTS="$scene"
          export EVAL_PROJECTION_VARIANTS="$variant"
          export EVAL_FM_NTIMESTEPS="$K"
          export EVAL_SAVE_TAG="thsweep_k${K}"
          export EVAL_N_TRIALS="$NTRIALS"
          python scripts/eval.py
        ) > "$log" 2>&1 &
        job_idx=$((job_idx + 1))
      done
    done
  done
done
wait
echo "FM threshold-sweep complete. npz: <seed>/results/halfspace_<scene>_thsweep_k<K>/${BASE_VARIANT}-lateproj<TH>.npz"
