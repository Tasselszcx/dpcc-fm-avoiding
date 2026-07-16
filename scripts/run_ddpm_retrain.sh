#!/usr/bin/env bash
set -eo pipefail   # NOT -u: conda's MKL activation script references unbound vars

# ---------------------------------------------------------------------------
# Native low-step DDPM retraining + evaluation.
#
# The honest counterpart to "DDIM subsamples a K=20 model": here we TRAIN a
# fresh DDPM whose noise schedule is baked at n_diffusion_steps=K from the
# start (cosine_beta_schedule(K), betas clipped to [0,0.999] so K=2 is safe).
# Each K gets its own weights and its own H8_K<K>_Dmodels.GaussianDiffusion
# directory (train.py rebuilds exp_name/savepath from TRAIN_NDIFF), so nothing
# clobbers the existing K=20 model.
#
# This answers 学姐's todo: if you let the model KNOW it only has K steps at
# training time (unlike inference-time DDIM subsampling), does per-step DDPM
# recover the few-step quality that Flow Matching gets for free?
#
# MODE=train  : train K in {2,5,10,15} x seeds (K=20 reuses existing weights)
# MODE=eval   : eval those retrained weights via EVAL_NDIFF override
# MODE=all    : train then eval (default)
#
# Usage:  GPUS=0,1,2,3,4,5,6,7 MODE=all bash scripts/run_ddpm_retrain.sh
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."
source scripts/env_h800.sh >/dev/null

GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS_CSV"
NUM_GPUS="${#GPU_ARR[@]}"

MODE="${MODE:-all}"
KS_CSV="${KS:-2,5,10,15}"          # K=20 reuses existing trained model
SEEDS_CSV="${SEEDS:-0,1,2}"
SCENES_CSV="${SCENES:-top-right-hard,top-left-hard,both-hard}"
VARIANT="${VARIANT:-dpcc-c-tightened}"
EXP="${EXP:-avoiding-synthetic}"   # DDPM config
NTRIALS="${NTRIALS:-50}"

IFS=',' read -ra KS <<< "$KS_CSV"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=',' read -ra SCENES <<< "$SCENES_CSV"

LOG_DIR="${LOG_DIR:-run_logs/ddpmretrain_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

echo "GPUs: ${GPU_ARR[*]} | mode=$MODE | K: ${KS[*]} | seeds: ${SEEDS[*]}"
echo "exp=$EXP variant=$VARIANT n_trials=$NTRIALS | logs: $LOG_DIR"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$NUM_GPUS" ]; do
    wait -n
  done
}

job_idx=0

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
  echo "=== Training native low-step DDPMs ==="
  for K in "${KS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      wait_for_slot
      gpu="${GPU_ARR[$((job_idx % NUM_GPUS))]}"
      log="$LOG_DIR/train_K${K}_seed${seed}_gpu${gpu}.log"
      echo "GPU $gpu :: train K=$K seed=$seed"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export TRAIN_EXP="$EXP"
        export TRAIN_SEEDS="$seed"
        export TRAIN_NDIFF="$K"
        python scripts/train.py
      ) > "$log" 2>&1 &
      job_idx=$((job_idx + 1))
    done
  done
  wait
  echo "=== Training complete ==="
fi

job_idx=0

if [[ "$MODE" == "eval" || "$MODE" == "all" ]]; then
  echo "=== Evaluating retrained DDPMs (per-step projection) ==="
  for K in "${KS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for scene in "${SCENES[@]}"; do
        wait_for_slot
        gpu="${GPU_ARR[$((job_idx % NUM_GPUS))]}"
        log="$LOG_DIR/eval_K${K}_seed${seed}_${scene}_gpu${gpu}.log"
        echo "GPU $gpu :: eval K=$K seed=$seed scene=$scene"
        (
          export CUDA_VISIBLE_DEVICES="$gpu"
          export EVAL_EXPS="$EXP"
          export EVAL_SEEDS="$seed"
          export EVAL_HALFSPACE_VARIANTS="$scene"
          export EVAL_PROJECTION_VARIANTS="$VARIANT"
          export EVAL_NDIFF="$K"
          export EVAL_SAVE_TAG="retrain${K}"
          export EVAL_N_TRIALS="$NTRIALS"
          python scripts/eval.py
        ) > "$log" 2>&1 &
        job_idx=$((job_idx + 1))
      done
    done
  done
  wait
  echo "=== Evaluation complete. Results tagged retrain<K>. ==="
fi

echo "DDPM native-retrain pipeline done. Logs: $LOG_DIR"
