#!/usr/bin/env bash
set -eo pipefail   # NOT -u: conda's MKL activation script references unbound vars

# ---------------------------------------------------------------------------
# Baseline re-evaluation at n_trials=100 (inference only, NO retraining).
#
# Brings the two NON-projection baselines up to the same n=100 workload as the
# main dpcc-c-tightened rows, so Experiment 1 (Diffuser / Post-processing /
# online-projection) is reported at a uniform n=100 for BOTH backbones:
#   diffuser         -> unconstrained sampling (no projection)
#   post_processing  -> single QP projection on the final trajectory
# for method in {fm (avoiding-synthetic-fm), ddpm (avoiding-synthetic)}.
#
# These variants don't depend on K (native K=20), so no K override is used.
# Results are written under an EVAL_SAVE_TAG="n100" suffix
#   halfspace_<scene>_n100/<variant>.npz
# so the existing n=50 npz (halfspace_<scene>/<variant>.npz) are NEVER touched.
#
# 2 variants x 2 backbones x 3 seeds x 3 scenes = 36 jobs.
# Both baselines are light (0 or 1 projection/episode), so they run fast even
# alongside a CPU-bound projection sweep.
#
# Usage:  GPUS=0,1,2,3,4,5,6,7 bash scripts/run_baselines_n100.sh
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."
source scripts/env_h800.sh >/dev/null

GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS_CSV"
NUM_GPUS="${#GPU_ARR[@]}"

VARIANTS_CSV="${VARIANTS:-diffuser,post_processing}"
METHODS_CSV="${METHODS:-fm,ddpm}"
SEEDS_CSV="${SEEDS:-0,1,2}"
SCENES_CSV="${SCENES:-top-right-hard,top-left-hard,both-hard}"
NTRIALS="${NTRIALS:-100}"

IFS=',' read -ra VARIANTS <<< "$VARIANTS_CSV"
IFS=',' read -ra METHODS <<< "$METHODS_CSV"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra SCENES <<< "$SCENES_CSV"

LOG_DIR="${LOG_DIR:-run_logs/baselines_n100_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

echo "GPUs: ${GPU_ARR[*]} | methods: ${METHODS[*]} | variants: ${VARIANTS[*]}"
echo "seeds: ${SEEDS[*]} | scenes: ${SCENES[*]} | n_trials=$NTRIALS | logs: $LOG_DIR"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$NUM_GPUS" ]; do
    wait -n
  done
}

job_idx=0
for method in "${METHODS[@]}"; do
  case "$method" in
    fm)   EXP="avoiding-synthetic-fm" ;;
    ddpm) EXP="avoiding-synthetic" ;;
    *)    echo "unknown method $method"; exit 1 ;;
  esac
  for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for scene in "${SCENES[@]}"; do
        wait_for_slot
        gpu="${GPU_ARR[$((job_idx % NUM_GPUS))]}"
        log="$LOG_DIR/${method}_${variant}_seed${seed}_${scene}_gpu${gpu}.log"
        echo "GPU $gpu :: $method($EXP) $variant seed=$seed scene=$scene"
        (
          export CUDA_VISIBLE_DEVICES="$gpu"
          export EVAL_EXPS="$EXP"
          export EVAL_SEEDS="$seed"
          export EVAL_HALFSPACE_VARIANTS="$scene"
          export EVAL_PROJECTION_VARIANTS="$variant"
          export EVAL_SAVE_TAG="n100"
          export EVAL_N_TRIALS="$NTRIALS"
          python scripts/eval.py
        ) > "$log" 2>&1 &
        job_idx=$((job_idx + 1))
      done
    done
  done
done
wait
echo "Baseline n100 eval complete. npz: <seed>/results/halfspace_<scene>_n100/{diffuser,post_processing}.npz"
