#!/usr/bin/env bash
# 统一环境设置：source 本文件后即可直接跑 scripts/train.py / eval.py / run_h800_parallel.sh
# 用法: source scripts/env_h800.sh

PROJECT_ROOT="/home/hadoop-efficient-llm/projects/dpcc-fm-avoiding"

# 网络/路径：去掉 linuxbrew 与 spark 的 PYTHONPATH 干扰
unset no_proxy NO_PROXY PYTHONPATH
export PATH="/usr/bin:/usr/sbin:$HOME/miniconda3/bin:$PATH"

# 隔离 ~/.local 里的 verl editable 包（否则其顶层 scripts/ 会盖掉本项目 scripts/）
export PYTHONNOUSERSITE=1
# d3il 内部用 `from environments.d3il...`，需要 <root>/d3il 在 path 上
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/d3il"
# MuJoCo 无显示渲染
export MUJOCO_GL=egl
# 关闭 wandb 联网（实验本地跑）
export WANDB_MODE="${WANDB_MODE:-disabled}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dpcc-fm

echo "[env_h800] conda=$CONDA_DEFAULT_ENV python=$(which python)"
