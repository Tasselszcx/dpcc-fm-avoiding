#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-dpcc-fm}"

if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y libgl1 libglib2.0-0 libglfw3 libosmesa6 patchelf
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda/Miniforge first, then rerun this script." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env create -f environment_wsl_cuda.yml
fi

conda activate "$ENV_NAME"
pip install -e diffuser
pip install -e d3il/environments/d3il

python scripts/generate_synthetic_data.py
python scripts/audit_synthetic_data.py

python - <<'PY'
import torch
import pybullet
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("pybullet ok")
print("D3IL Avoiding import ok:", ObstacleAvoidanceEnv.__name__)
PY

echo "Setup complete. Activate with: conda activate $ENV_NAME"
