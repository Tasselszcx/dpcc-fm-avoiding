#!/usr/bin/env bash
set -eo pipefail   # NOT -u: conda's MKL activation script references unbound vars

# ---------------------------------------------------------------------------
# DDPM projection-frequency (threshold) sweep -- inference only, NO retraining.
#
# Directly answers CZ-3 in its original DDPM framing: "based on the paper,
# applying projection only during the last half of the denoising timesteps
# achieves similar safety while reducing cost." We quantify that trade-off on
# the *DDPM* backbone here (the FM sweep lives in run_fm_thresholdsweep.sh).
#
# The projector fires only when the denoising timestep satisfies
#   t <= diffusion_timestep_threshold * n_timesteps          (diffusion.py:194)
# i.e. only in the *late* (near-data) portion of denoising. So `threshold` is
# the fraction of the late denoising window in which projection is active
# (threshold=1.0 -> project every step; 0.2 -> only the last ~20%). We encode
# the threshold in the projection-variant name via the `lateprojNN` suffix
# (eval.py:281 parses NN/100 -> threshold), which ALSO becomes the npz
# filename, so nothing overwrites the DDIM/main-eval runs.
#
# Fixed native DDPM checkpoint (K=20 weights). K is training-bound for DDPM, so
# unlike the FM sweep there is NO K override -- K is pinned at 20.
#   threshold in {1.0,0.7,0.5,0.3,0.2,0.1}   (lateproj{100,70,50,30,20,10})
# 6 thresholds x 3 seeds x 3 scenes = 54 jobs, one process per GPU.
#
# #projections for K=20 (t in 19..0, project when t <= thr*20):
#   thr 1.0 -> 20 | 0.7 -> 15 | 0.5 -> 11 | 0.3 -> 7 | 0.2 -> 5 | 0.1 -> 3
#
# Usage:  GPUS=0,1,2,3,4,5,6,7 bash scripts/run_ddpm_thresholdsweep.sh
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."
source scripts/env_h800.sh >/dev/null

GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS_CSV"
NUM_GPUS="${#GPU_ARR[@]}"

# threshold percentages -> lateprojNN suffix (NN/100 = diffusion_timestep_threshold)
THS_CSV="${THS:-100,70,50,30,20,10}"
SEEDS_CSV="${SEEDS:-0,1,2}"
SCENES_CSV="${SCENES:-top-right-hard,top-left-hard,both-hard}"
BASE_VARIANT="${BASE_VARIANT:-dpcc-c-tightened}"
EXP="${EXP:-avoiding-synthetic}"     # DDPM backbone (K=20, NO -fm suffix)
NTRIALS="${NTRIALS:-100}"

IFS=',' read -ra THS <<< "$THS_CSV"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra SCENES <<< "$SCENES_CSV"

LOG_DIR="${LOG_DIR:-run_logs/ddpm_threshsweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

echo "GPUs: ${GPU_ARR[*]} | K: 20 (native DDPM) | thresholds%: ${THS[*]}"
echo "seeds: ${SEEDS[*]} | scenes: ${SCENES[*]}"
echo "exp=$EXP base_variant=$BASE_VARIANT n_trials=$NTRIALS | logs: $LOG_DIR"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$NUM_GPUS" ]; do
    wait -n
  done
}

job_idx=0
for TH in "${THS[@]}"; do
  variant="${BASE_VARIANT}-lateproj${TH}"
  for seed in "${SEEDS[@]}"; do
    for scene in "${SCENES[@]}"; do
      wait_for_slot
      gpu="${GPU_ARR[$((job_idx % NUM_GPUS))]}"
      log="$LOG_DIR/ddpm_th${TH}_seed${seed}_${scene}_gpu${gpu}.log"
      echo "GPU $gpu :: DDPM K=20 th=${TH}% seed=$seed scene=$scene variant=$variant"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export EVAL_EXPS="$EXP"
        export EVAL_SEEDS="$seed"
        export EVAL_HALFSPACE_VARIANTS="$scene"
        export EVAL_PROJECTION_VARIANTS="$variant"
        export EVAL_SAVE_TAG="${SAVE_TAG:-ddpm_thsweep}"
        export EVAL_N_TRIALS="$NTRIALS"
        python scripts/eval.py
      ) > "$log" 2>&1 &
      job_idx=$((job_idx + 1))
    done
  done
done
wait
echo "DDPM threshold-sweep complete. npz: <seed>/results/halfspace_<scene>_ddpm_thsweep/${BASE_VARIANT}-lateproj<TH>.npz"
