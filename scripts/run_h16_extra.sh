#!/usr/bin/env bash
# H=16 补充实验（门控在 #43 proj_cmp 全部结束后自动启动）：
#   1) DDPM K=1,2 H16 真实数据重训（6 job）
#   2) 并行评测：DDIM sweep K∈{1,2,4,8,16,32}（K32 基座子采样，免训练）
#                + 重训的 DDPM native K=1,2 评测
# 全部 dpcc-c-tightened, th=0.5, H=16, n=100, 3 场景 × 3 seed。
set -eo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="run_logs/h16_extra"
mkdir -p "$LOG_DIR"
SCENES=(top-right-hard top-left-hard both-hard)
POOL=(0 1 2 3 4 5 6 7)

# ---- 0) 等 #43 的 proj_cmp eval 全部结束 ----
echo "[extra] $(date '+%F %T') 等待 #43 eval 进程排空..."
while pgrep -f "python scripts/eval.py" >/dev/null 2>&1; do sleep 60; done
echo "[extra] $(date '+%F %T') GPU 已空闲，开始补充实验。"

# ---- 1) DDPM K=1,2 H16 重训 ----
i=0
for K in 1 2; do for s in 0 1 2; do
  g=${POOL[$(( i % 8 ))]}
  echo "[extra] train DDPM K=$K seed=$s -> GPU$g"
  env CUDA_VISIBLE_DEVICES=$g TRAIN_EXP=avoiding-d3il TRAIN_HORIZON=16 TRAIN_NDIFF=$K TRAIN_SEEDS=$s \
    bash -c 'source scripts/env_h800.sh >/dev/null 2>&1; python scripts/train.py' \
    > "$LOG_DIR/train_ddpm_k${K}_seed${s}_g${g}.log" 2>&1 &
  i=$(( i + 1 )); sleep 3
done; done
wait
echo "[extra] $(date '+%F %T') ===DDPM_K12_TRAIN_DONE==="

# ---- 2) 并行评测（slot 限制 24 并发 = 3/GPU，防显存超订）----
MAXJOBS=24
slot() { while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJOBS" ]; do wait -n; done; }
i=0
COMMON=(OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
        EVAL_PROJECTION_VARIANTS=dpcc-c-tightened EVAL_N_TRIALS=100 EVAL_DEVICE=cuda EVAL_HORIZON=16)

# 2a) DDIM sweep：K32 基座，子采样 DK 步
for DK in 1 2 4 8 16 32; do for s in 0 1 2; do for sc in "${SCENES[@]}"; do
  slot; g=${POOL[$(( i % 8 ))]}
  env CUDA_VISIBLE_DEVICES=$g EVAL_SEEDS=$s "${COMMON[@]}" \
    EVAL_HALFSPACE_VARIANTS=$sc EVAL_EXPS=avoiding-d3il EVAL_NDIFF=32 EVAL_DDIM_STEPS=$DK \
    EVAL_SAVE_TAG=d3il_h16_ddim_k${DK} \
    bash -c 'source scripts/env_h800.sh >/dev/null 2>&1; python scripts/eval.py' \
    > "$LOG_DIR/ddim_k${DK}_s${s}_${sc}_g${g}.log" 2>&1 &
  i=$(( i + 1 ))
done; done; done

# 2b) 重训 DDPM native K=1,2 评测（写入与 sweep 一致的 d3il_h16_ddpm_k{K} 目录）
for K in 1 2; do for s in 0 1 2; do for sc in "${SCENES[@]}"; do
  slot; g=${POOL[$(( i % 8 ))]}
  env CUDA_VISIBLE_DEVICES=$g EVAL_SEEDS=$s "${COMMON[@]}" \
    EVAL_HALFSPACE_VARIANTS=$sc EVAL_EXPS=avoiding-d3il EVAL_NDIFF=$K \
    EVAL_SAVE_TAG=d3il_h16_ddpm_k${K} \
    bash -c 'source scripts/env_h800.sh >/dev/null 2>&1; python scripts/eval.py' \
    > "$LOG_DIR/ddpm_k${K}_s${s}_${sc}_g${g}.log" 2>&1 &
  i=$(( i + 1 ))
done; done; done
wait
echo "[extra] $(date '+%F %T') ===ALL_EXTRA_EVAL_DONE=== 共 $i 个评测 job"
