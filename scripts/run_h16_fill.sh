#!/usr/bin/env bash
# 按缺失清单单场景并行补齐 H16 评测。每行: "BB Kw Sw scene"
set +e
cd /home/hadoop-efficient-llm/projects/dpcc-fm-avoiding
mkdir -p run_logs/d3il_eval_h16
i=0
while read -r bb kw sw sc; do
  [ -z "$bb" ] && continue
  K=$(echo "$kw" | grep -oE '[0-9]+')
  s=$(echo "$sw" | grep -oE '[0-9]+')
  g=$(( i % 8 ))
  EA=(CUDA_VISIBLE_DEVICES=$g EVAL_SEEDS=$s
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
      EVAL_HALFSPACE_VARIANTS=$sc EVAL_PROJECTION_VARIANTS=dpcc-c-tightened
      EVAL_N_TRIALS=100 EVAL_DEVICE=cuda EVAL_HORIZON=16)
  if [ "$bb" = "FM" ]; then
    EA+=(EVAL_EXPS=avoiding-d3il-fm EVAL_FM_NTIMESTEPS=$K EVAL_SAVE_TAG=d3il_h16_fm_k${K})
  else
    EA+=(EVAL_EXPS=avoiding-d3il EVAL_NDIFF=$K EVAL_SAVE_TAG=d3il_h16_ddpm_k${K})
  fi
  env "${EA[@]}" bash -c 'source scripts/env_h800.sh >/dev/null 2>&1; python scripts/eval.py' \
    > "run_logs/d3il_eval_h16/fill_${bb}_k${K}_s${s}_${sc}_g${g}.log" 2>&1 &
  echo "[fill] $bb K=$K seed=$s $sc -> GPU$g"
  i=$(( i + 1 ))
done < /tmp/h16_missing.txt
echo "[fill] 共派发 $i 个单场景 job"
