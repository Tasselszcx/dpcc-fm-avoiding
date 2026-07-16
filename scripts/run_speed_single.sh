#!/usr/bin/env bash
set -eo pipefail   # NOT -u: conda's MKL activation script references unbound vars

# ---------------------------------------------------------------------------
# Single-process t/step re-measurement (SPEED ONLY).
#
# The n100 sweep ran 8-way concurrent, so its avg_time is inflated: cvxpy /
# SLSQP projection solvers fight over CPU threads under concurrency (memory
# hard rule: t/step must be measured single-process on an idle machine).
#
# This launcher re-runs the SAME eval.py pipeline STRICTLY SERIALLY, one job
# at a time on a single GPU, with solver threads pinned to 1 so each projection
# solve is a clean single-core measurement not perturbed by other users' CPU
# load. Quality metrics are NOT the point here (single seed/scene, n_trials=20);
# we only read avg_time out of the resulting npz. Tags: <method><K>speed.
#
# Usage:  GPU=1 bash scripts/run_speed_single.sh
#         METHODS=fm KS=20 GPU=1 bash scripts/run_speed_single.sh   # calibrate
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."
source scripts/env_h800.sh >/dev/null

GPU="${GPU:-1}"
KS_CSV="${KS:-1,2,3,4,5,6,8,10,15,20}"
METHODS_CSV="${METHODS:-fm,ddim,ddpm}"
SEED="${SEED:-0}"
SCENE="${SCENE:-top-right-hard}"
VARIANT="${VARIANT:-dpcc-c-tightened}"
NTRIALS="${NTRIALS:-20}"

# Pin every CPU math/solver backend to 1 thread so a single projection solve
# does not fan out and collide with other users' load. Same setting for all
# methods => same yardstick.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

IFS=',' read -ra KS <<< "$KS_CSV"
IFS=',' read -ra METHODS <<< "$METHODS_CSV"

LOG_DIR="${LOG_DIR:-run_logs/speedsingle_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

echo "SERIAL single-process timing | GPU=$GPU seed=$SEED scene=$SCENE"
echo "methods=${METHODS[*]} K=${KS[*]} n_trials=$NTRIALS | threads pinned to 1"
echo "load at start: $(uptime | sed 's/.*load average/load/')"
echo "logs: $LOG_DIR"

for method in "${METHODS[@]}"; do
  for K in "${KS[@]}"; do
    log="$LOG_DIR/${method}_K${K}.log"
    echo ">>> $method K=$K  ($(date +%H:%M:%S))"
    (
      export CUDA_VISIBLE_DEVICES="$GPU"
      export EVAL_SEEDS="$SEED"
      export EVAL_HALFSPACE_VARIANTS="$SCENE"
      export EVAL_PROJECTION_VARIANTS="$VARIANT"
      export EVAL_N_TRIALS="$NTRIALS"
      case "$method" in
        fm)
          export EVAL_EXPS="avoiding-synthetic-fm"
          export EVAL_FM_NTIMESTEPS="$K"
          export EVAL_SAVE_TAG="fmk${K}speed"
          ;;
        ddim)
          export EVAL_EXPS="avoiding-synthetic"
          export EVAL_DDIM_STEPS="$K"
          export EVAL_DDIM_ETA="0.0"
          export EVAL_SAVE_TAG="ddim${K}speed"
          ;;
        ddpm)
          export EVAL_EXPS="avoiding-synthetic"
          export EVAL_NDIFF="$K"
          export EVAL_SAVE_TAG="retrain${K}speed"
          ;;
      esac
      python scripts/eval.py
    ) > "$log" 2>&1
    # pull the per-step time straight out of the log for a live readout
    tstep=$(grep -oE 'Average computation time per step: [0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$' || echo '?')
    echo "    t/step = ${tstep}s"
  done
done

echo "=== serial timing done. load now: $(uptime | sed 's/.*load average/load/') ==="
echo "Tags: <method><K>speed. Aggregate with scripts/aggregate_speed.py"
