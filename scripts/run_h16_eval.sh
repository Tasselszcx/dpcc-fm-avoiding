#!/usr/bin/env bash
# H=16 评测矩阵：FM(EVAL_FM_NTIMESTEPS 一模型扫K) + DDPM(EVAL_NDIFF 每K一模型)
# K∈{4,8,16,32} × seed∈{0,1,2} × 两主干 = 24 个评测 job，各跑三场景 n=100，th=0.5 默认。
set -eo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="run_logs/d3il_eval_h16"
mkdir -p "$LOG_DIR"
SCENES="top-right-hard,top-left-hard,both-hard"
i=0
launch() {  # $1=backbone(fm/ddpm) $2=K $3=seed
  local bb=$1 K=$2 s=$3 g=$(( i % 8 ))
  # 限制每进程单线程：SLSQP/numpy 默认抢占全部216核，24个job会线程爆炸(load~980)。
  # 并行度靠24个job本身，每job单线程 → 24线程<<216核，各自全速。
  local -a EA=(CUDA_VISIBLE_DEVICES=$g EVAL_SEEDS=$s
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    EVAL_HALFSPACE_VARIANTS=$SCENES EVAL_PROJECTION_VARIANTS=dpcc-c-tightened
    EVAL_N_TRIALS=100 EVAL_DEVICE=cuda EVAL_HORIZON=16)
  if [ "$bb" = "fm" ]; then
    EA+=(EVAL_EXPS=avoiding-d3il-fm EVAL_FM_NTIMESTEPS=$K EVAL_SAVE_TAG=d3il_h16_fm_k${K})
  else
    EA+=(EVAL_EXPS=avoiding-d3il EVAL_NDIFF=$K EVAL_SAVE_TAG=d3il_h16_ddpm_k${K})
  fi
  env "${EA[@]}" bash -c 'source scripts/env_h800.sh >/dev/null 2>&1; python scripts/eval.py' \
    > "$LOG_DIR/eval_${bb}_h16_k${K}_seed${s}_g${g}.log" 2>&1 &
  echo "[h16-eval] $bb K=$K seed=$s -> GPU$g"
  i=$(( i + 1 ))
}
for K in 4 8 16 32; do for s in 0 1 2; do launch fm   "$K" "$s"; done; done
for K in 4 8 16 32; do for s in 0 1 2; do launch ddpm "$K" "$s"; done; done
echo "[h16-eval] 共派发 $i 个评测 job"
