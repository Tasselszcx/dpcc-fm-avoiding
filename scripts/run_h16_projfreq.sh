#!/usr/bin/env bash
# #41 DDPM K=16/H=16 投影次数 trade-off 扫参（学姐 todo③：只在后半段投影）
# 免训练，复用 H16_K16 GaussianDiffusion 模型。
# threshold 决定「只在最后 th×K 步投影」→ K=16 下投影次数 = th×16：
#   th0.5   -> 8 次投影 (=现成 dpcc-c-tightened 基准，不重跑)
#   th0.25  -> 4 次
#   th0.125 -> 2 次
#   th0.0625-> 1 次
# 3 variant × 3 场景 × 3 seed = 27 新 npz，与基准同目录 d3il_h16_ddpm_k16。
set -eo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="run_logs/projfreq"
mkdir -p "$LOG_DIR"
SCENES=(top-right-hard top-left-hard both-hard)
POOL=(0 1 2 3 4 5 6 7)
VARIANTS="dpcc-c-tightened-th0p25,dpcc-c-tightened-th0p125,dpcc-c-tightened-th0p0625"
i=0
for s in 0 1 2; do for sc in "${SCENES[@]}"; do
  g=${POOL[$(( i % 8 ))]}
  echo "[projfreq] seed=$s scene=$sc -> GPU$g"
  env CUDA_VISIBLE_DEVICES=$g EVAL_SEEDS=$s \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    EVAL_HALFSPACE_VARIANTS=$sc EVAL_PROJECTION_VARIANTS=$VARIANTS \
    EVAL_N_TRIALS=100 EVAL_DEVICE=cuda EVAL_HORIZON=16 \
    EVAL_EXPS=avoiding-d3il EVAL_NDIFF=16 EVAL_SAVE_TAG=d3il_h16_ddpm_k16 \
    bash -c 'source scripts/env_h800.sh >/dev/null 2>&1; python scripts/eval.py' \
    > "$LOG_DIR/pf_s${s}_${sc}_g${g}.log" 2>&1 &
  i=$(( i + 1 ))
done; done
wait
echo "[projfreq] $(date '+%F %T') ===PROJFREQ_DONE=== 共 $i 个 job"
