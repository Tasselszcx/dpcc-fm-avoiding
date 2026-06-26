# Running on 8x H800 80GB

This project does not need distributed data parallel training. Each experiment is an independent model/seed pair:

- DDPM: `avoiding-synthetic`, seeds `0,1,2`
- FM: `avoiding-synthetic-fm`, seeds `0,1,2`

That gives 6 independent training jobs. On an 8x H800 machine, run them as 6 single-GPU processes in parallel, then run evaluation jobs in parallel by experiment, seed, and scene.

## Expected Runtime

These are planning estimates, not guaranteed numbers. The final runtime depends on CPU speed, disk speed, MuJoCo/headless setup, and enabled projection variants.

| Stage | Work | Expected wall time on 8x H800 |
| --- | --- | --- |
| Setup + data audit | install, regenerate 288 demos, audit | 5-20 min after conda packages are available |
| Training pilot | 1 FM seed, 1k-5k steps | a few minutes |
| Full training | 6 jobs = 2 models x 3 seeds x 100k steps | about 1-2 hours if all 6 jobs run concurrently |
| Quick eval | 2 models x 3 seeds x 3 scenes x 3 variants x 10 trials | about 0.5-2 hours |
| Full eval | 2 models x 3 seeds x 3 scenes x 3 variants x 50 trials | about 3-8 hours |
| Full train + full eval | default paper-style local config | about 4-10 hours total |

Training is mostly GPU-bound and parallelizes cleanly. Evaluation can be slower because DPCC projection uses SciPy/SLSQP and spends meaningful time on CPU-side optimization.

Current config uses the quick/core projection set:

- `diffuser`
- `dpcc-c-tightened`
- `post_processing`

If you uncomment all projection variants in `config/projection_eval.yaml`, full evaluation can become much longer.

## First Setup

```bash
git clone https://github.com/Tasselszcx/dpcc-fm-avoiding.git
cd dpcc-fm-avoiding
bash setup_wsl_cuda.sh
conda activate dpcc-fm
```

Check CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## Smoke Test

Run a short FM training/eval before launching the full job:

```bash
TRAIN_EXP=avoiding-synthetic-fm \
TRAIN_STEPS=5000 \
EVAL_N_TRIALS=10 \
bash scripts/run_pilot_4090.sh
```

The pilot writes evaluation outputs under a tagged directory such as `halfspace_top-right-hard_pilot5000`, so it will not overwrite full results.

## Full 8-GPU Run

Use the H800 parallel launcher:

```bash
conda activate dpcc-fm
bash scripts/run_h800_parallel.sh
```

Default behavior:

- uses GPUs `0,1,2,3,4,5,6,7`
- trains DDPM and FM
- trains seeds `0,1,2`
- evaluates all three scenes
- evaluates `diffuser,dpcc-c-tightened,post_processing`

Logs go to `run_logs/h800_YYYYMMDD_HHMMSS/`.

## Useful Overrides

Use specific GPUs:

```bash
GPUS=0,1,2,3,4,5 bash scripts/run_h800_parallel.sh
```

Train only FM:

```bash
MODE=train EXPS=avoiding-synthetic-fm SEEDS=0,1,2 bash scripts/run_h800_parallel.sh
```

Evaluate only FM after training:

```bash
MODE=eval EXPS=avoiding-synthetic-fm SEEDS=0,1,2 bash scripts/run_h800_parallel.sh
```

Quick eval with fewer trials:

```bash
MODE=eval \
EXPS=avoiding-synthetic-fm \
SEEDS=0 \
SCENES=top-right-hard \
PROJECTION_VARIANTS=diffuser,dpcc-c-tightened \
EVAL_N_TRIALS=10 \
bash scripts/run_h800_parallel.sh
```

Run the original one-process script instead:

```bash
bash scripts/run_d3il_fm_experiment.sh
```

That script is simpler but mostly serial; it will not use all 8 GPUs efficiently.

## Outputs

Training checkpoints:

```text
logs/avoiding-synthetic/diffusion_three_scenarios/H8_K20_Dmodels.GaussianDiffusion/<seed>/
logs/avoiding-synthetic/diffusion_three_scenarios/H8_K20_Dmodels.FlowMatching/<seed>/
```

Evaluation results:

```text
logs/avoiding-synthetic/plans_three_scenarios/H8_K20_Dmodels.GaussianDiffusion/<seed>/results/halfspace_<scene>/*.npz
logs/avoiding-synthetic/plans_three_scenarios/H8_K20_Dmodels.FlowMatching/<seed>/results/halfspace_<scene>/*.npz
```

Summaries:

```bash
RESULT_EXP=avoiding-synthetic python scripts/load_results.py
RESULT_EXP=avoiding-synthetic-fm python scripts/load_results.py
```

## Practical Notes

- Start with `TRAIN_STEPS=5000` or `10000` if you want to verify speed and logs before full 100k training.
- Do not put `logs/` into git. Checkpoints and result files are intentionally ignored.
- If evaluation is the bottleneck, reduce `EVAL_N_TRIALS` first, then reduce `PROJECTION_VARIANTS`.
- If multiple eval jobs write plots at the same time, the `.npz` metrics remain the important output. Plot files may be overwritten by parallel jobs.
