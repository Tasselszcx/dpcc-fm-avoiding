# DPCC Flow Matching Avoiding

This repo is a clean, runnable fork of the DPCC Avoiding experiment with a Flow Matching (FM) model added beside the DDPM baseline. The original D3IL Avoiding demonstrations were not available in this workspace, so this project uses a synthetic three-scenario Avoiding dataset that keeps train/test scenarios aligned.

## What Is Included

- DDPM baseline config: `config/avoiding-synthetic.py`
- FM config with the same `K=20`: `config/avoiding-synthetic-fm.py`
- Three-scenario synthetic data generator: `scripts/generate_synthetic_data.py`
- Data quality audit: `scripts/audit_synthetic_data.py`
- Training: `scripts/train.py`
- Evaluation: `scripts/eval.py`
- One-command full experiment: `scripts/run_d3il_fm_experiment.sh`
- 4090/WSL pilot script: `scripts/run_pilot_4090.sh`

The tracked synthetic dataset has 288 trajectories: 96 each for `top-right-hard`, `top-left-hard`, and `both-hard`. The latest audit report is in `reports/synthetic_data_quality_report.md`.

## Setup on WSL + RTX 4090

```bash
git clone <repo-url>
cd dpcc-fm-avoiding
bash setup_wsl_cuda.sh
conda activate dpcc-fm
```

The setup script creates a CUDA-enabled PyTorch environment, installs local `diffuser` and D3IL packages, regenerates/audits the dataset, and checks CUDA/D3IL imports.

## Quick Pilot

Run a short FM training/evaluation smoke test:

```bash
conda activate dpcc-fm
bash scripts/run_pilot_4090.sh
```

Useful overrides:

```bash
TRAIN_EXP=avoiding-synthetic TRAIN_STEPS=1000 bash scripts/run_pilot_4090.sh
TRAIN_EXP=avoiding-synthetic-fm TRAIN_STEPS=5000 EVAL_N_TRIALS=10 bash scripts/run_pilot_4090.sh
```

Pilot results are written under `logs/.../results/halfspace_*_pilot*` so they do not pollute full evaluation directories.

## Full Experiment

Default full run trains and evaluates DDPM + FM for seeds `0,1,2`:

```bash
conda activate dpcc-fm
bash scripts/run_d3il_fm_experiment.sh
```

To run at least one complete FM training plus evaluation first:

```bash
TRAIN_EXPS=avoiding-synthetic-fm \
TRAIN_SEEDS=0 \
EVAL_EXPS=avoiding-synthetic-fm \
EVAL_SEEDS=0 \
bash scripts/run_d3il_fm_experiment.sh
```

By default the configs train for `100000` steps and evaluate `50` trials per scene.

## Notes

- The FM and DDPM configs both use `n_diffusion_steps=20`; only the generative process is changed.
- Training uses CUDA automatically when available. Evaluation falls back to CPU only when CUDA is unavailable, because the projection code uses float64 tensors and MPS does not support them.
- Runtime outputs, logs, and checkpoints are ignored by git.
