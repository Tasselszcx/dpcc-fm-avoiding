#!/usr/bin/env bash
# =====================================================================
# DPCC — Mac Apple Silicon 开发环境搭建脚本 (M1/M2/M3)
#
# 无需 CUDA；使用 Apple MPS (Metal) 进行本地开发测试。
# 生产训练仍建议在 Linux + NVIDIA GPU 服务器上运行。
#
# 用法：chmod +x setup_local_mac.sh && ./setup_local_mac.sh
# =====================================================================
set -e

CONDA_BASE=/opt/homebrew/Caskroom/miniforge/base
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME=dpcc

# ── 0. 检查 Miniforge/conda ──────────────────────────────────────────
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "→ Miniforge not found. Installing via Homebrew..."
  brew install --cask miniforge
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── 1. 创建 / 重用 dpcc 环境 ─────────────────────────────────────────
if conda info --envs | grep -q "^$ENV_NAME "; then
  echo "✓ Conda env '$ENV_NAME' already exists, reusing."
else
  echo "→ Creating conda env '$ENV_NAME' with Python 3.10..."
  conda create -n $ENV_NAME python=3.10 -y
fi
conda activate $ENV_NAME

# ── 2. 安装 requirements（Mac 兼容版，去掉 CUDA/PyQt5/pybullet）────────
echo "→ Installing requirements (Mac-compatible)..."
pip install -r "$PROJ_DIR/requirements_mac.txt" --quiet

# ── 3. 修正 torchvision 版本（匹配 torch 2.4.x）────────────────────────
echo "→ Fixing torchvision to match torch 2.4.1..."
pip install "torchvision==0.19.1" --quiet

# ── 4. 安装 diffuser 包（项目本地包，editable 模式）──────────────────────
echo "→ Installing local diffuser package..."
pip install -e "$PROJ_DIR/diffuser" --quiet

# ── 5. 克隆 D3IL（如不存在）─────────────────────────────────────────────
if [ ! -d "$PROJ_DIR/d3il/agents" ]; then
  echo "→ Cloning D3IL..."
  git clone https://github.com/ALRhub/d3il.git "$PROJ_DIR/d3il"
else
  echo "✓ D3IL already cloned."
fi

# ── 6. 生成合成训练数据 ───────────────────────────────────────────────────
DATA_DIR="$PROJ_DIR/d3il/environments/dataset/data/avoiding_synthetic/data"
N_FILES=$(ls "$DATA_DIR" 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_FILES" -gt 10 ]; then
  echo "✓ Synthetic data already exists ($N_FILES files)."
else
  echo "→ Generating 100 synthetic training trajectories..."
  mkdir -p "$DATA_DIR"
  cd "$PROJ_DIR"
  python scripts/generate_synthetic_data.py
fi

# ── 7. 快速验证 ──────────────────────────────────────────────────────────
echo ""
echo "→ Running quick test (unit + 20-step MPS training)..."
cd "$PROJ_DIR"
python scripts/quick_test.py

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  常用命令："
echo ""
echo "    conda activate dpcc              # 激活环境"
echo "    python scripts/quick_test.py     # 快速验证"
echo "    python scripts/train.py          # 完整训练 (FlowMatching, 3 seeds)"
echo "    python scripts/eval.py           # 评测"
echo ""
echo "  ⚠ 完整训练建议在 Linux + CUDA 服务器上运行，此 Mac 可用于代码调试。"
echo "══════════════════════════════════════════════════════════════"
