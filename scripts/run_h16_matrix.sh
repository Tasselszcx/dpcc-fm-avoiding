#!/usr/bin/env bash
# H=16 训练矩阵：FM(3seed) + DDPM(K=4/8/16/32 各 3seed) = 15 次训练。
# 8×80G H800，显存管够，全部并发一次性铺开，不排队等空闲卡。
# round-robin 把 15 个 job 分到 8 卡；优先把前几个丢到当前空闲的 GPU6/7。
set -eo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="run_logs/h16"
mkdir -p "$LOG_DIR"

# 卡分配顺序：先用当前空闲的 6 7，再回到 0-5（这些卡上只跑着 <2G 的 eval/train，80G 卡完全叠得下）
POOL=(6 7 0 1 2 3 4 5)

# job 规格: "TAG|EXP|NDIFF|SEED"  (NDIFF 为空 = FM，训一个模型)
JOBS=()
for s in 0 1 2; do JOBS+=("fm_h16|avoiding-d3il-fm||$s"); done
for K in 4 8 16 32; do
  for s in 0 1 2; do JOBS+=("ddpm_h16_k${K}|avoiding-d3il|$K|$s"); done
done

echo "[h16] 共 ${#JOBS[@]} 个训练 job，全并发铺到 ${#POOL[@]} 张卡"
i=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r TAG EXP NDIFF SEED <<< "$spec"
  g=${POOL[$(( i % ${#POOL[@]} ))]}
  log="$LOG_DIR/train_${TAG}_seed${SEED}_gpu${g}.log"
  echo "[h16] 派发 $TAG seed=$SEED -> GPU$g  (log: $log)"
  NDIFF_ENV=""; [ -n "$NDIFF" ] && NDIFF_ENV="TRAIN_NDIFF=$NDIFF"
  env CUDA_VISIBLE_DEVICES=$g TRAIN_EXP=$EXP TRAIN_HORIZON=16 $NDIFF_ENV TRAIN_SEEDS=$SEED \
    bash -c 'source scripts/env_h800.sh >/dev/null 2>&1; python scripts/train.py' \
    > "$log" 2>&1 &
  i=$(( i + 1 ))
  sleep 3   # 轻微错峰，避免同一瞬间 15 个进程抢初始化
done

echo "[h16] 全部 15 个训练已并发派发，等待完成..."
wait
echo "[h16] ===ALL_H16_TRAINING_DONE==="
