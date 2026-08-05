#!/usr/bin/env bash
# H=8 DDPM native 训练矩阵（补齐与 H16 对称的 step-sweep）。
# FM 无需训练：单模型 H8_K20_Dmodels.FlowMatching 已存在，K 在推理期用 EVAL_FM_NTIMESTEPS 设定。
# 只训 DDPM K∈{1,2,4,8,16,32} × 3seed = 18 job；各 K 独立 H8_K<K>_Dmodels.GaussianDiffusion
# 目录（train.py 依 TRAIN_NDIFF 重建 exp_name/savepath），互不覆盖，也不动已有 H8_K20。
# 8×80G H800：run_h16_matrix 同款，全并发铺开（80G 卡叠得下多个 <2G 训练）。
set -eo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="run_logs/h8_matrix"
mkdir -p "$LOG_DIR"
POOL=(0 1 2 3 4 5 6 7)
JOBS=()
for K in 1 2 4 8 16 32; do
  for s in 0 1 2; do JOBS+=("ddpm_h8_k${K}|avoiding-d3il|$K|$s"); done
done
echo "[h8] 共 ${#JOBS[@]} 个训练 job，全并发铺到 ${#POOL[@]} 张卡"
i=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r TAG EXP NDIFF SEED <<< "$spec"
  g=${POOL[$(( i % ${#POOL[@]} ))]}
  log="$LOG_DIR/train_${TAG}_seed${SEED}_gpu${g}.log"
  echo "[h8] 派发 $TAG seed=$SEED -> GPU$g  (log: $log)"
  env CUDA_VISIBLE_DEVICES=$g TRAIN_EXP=$EXP TRAIN_HORIZON=8 TRAIN_NDIFF=$NDIFF TRAIN_SEEDS=$SEED \
    bash -c 'source scripts/env_h800.sh >/dev/null 2>&1; python scripts/train.py' \
    > "$log" 2>&1 &
  i=$(( i + 1 ))
  sleep 3   # 轻微错峰，避免同瞬间 18 个进程抢初始化
done
echo "[h8] 全部 ${#JOBS[@]} 个训练已并发派发，等待完成..."
wait
echo "[h8] $(date '+%F %T') ===ALL_H8_TRAINING_DONE==="
