#!/usr/bin/env python3
"""
DPCC 快速端到端验证脚本
- 运行 FlowMatching 单元测试
- 生成合成数据（如未生成）
- 运行 20 步迷你训练，确认完整 pipeline 可运行
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

DEVICE = ('cuda' if torch.cuda.is_available()
          else 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
          else 'cpu')

print(f'=== DPCC Quick Test  |  device={DEVICE} ===\n')

# ──────────────────────────────────────────────────────────────────
# 1. FlowMatching 单元测试
# ──────────────────────────────────────────────────────────────────
print('[1/3] FlowMatching unit test ...')
from diffuser.models import FlowMatching, UNet1DTemporalCondModel

model = UNet1DTemporalCondModel(
    horizon=8, transition_dim=10, cond_dim=6,
    dim=32, dim_mults=(1, 2, 4, 8),
)
fm = FlowMatching(model=model, horizon=8, observation_dim=6, action_dim=4,
                  n_timesteps=5, loss_type='l2', ode_solver='euler')
fm = fm.to(DEVICE)

x    = torch.randn(4, 8, 10).to(DEVICE)
cond = {0: torch.randn(4, 6).to(DEVICE)}

loss, info = fm.loss(x, cond)
assert 'diffusion_loss' in info, 'missing diffusion_loss'

samples, infos = fm.conditional_sample(cond, horizon=8)
assert samples.shape == (4, 8, 10), f'bad shape: {samples.shape}'

print(f'  ✓ loss={loss.item():.4f}  samples={samples.shape}')

# ──────────────────────────────────────────────────────────────────
# 2. 合成数据生成（幂等）
# ──────────────────────────────────────────────────────────────────
print('[2/3] Synthetic data ...')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'd3il',
                        'environments', 'dataset', 'data', 'avoiding_synthetic', 'data')
DATA_DIR = os.path.normpath(DATA_DIR)
n_existing = len([f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]) if os.path.isdir(DATA_DIR) else 0

if n_existing >= 100:
    print(f'  ✓ Already have {n_existing} trajectories, skipping generation.')
else:
    print(f'  → Generating 100 trajectories …')
    import pickle
    # Import the generator function directly (no exec)
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_synthetic_data import generate_trajectory, N_TRAJS
    os.makedirs(DATA_DIR, exist_ok=True)
    for i in range(N_TRAJS):
        data = generate_trajectory(seed=i)
        with open(os.path.join(DATA_DIR, f'env_{i:03d}_00.pkl'), 'wb') as f:
            pickle.dump(data, f)
    n_existing = len(os.listdir(DATA_DIR))
    print(f'  ✓ Generated {n_existing} trajectories in {DATA_DIR}')

# ──────────────────────────────────────────────────────────────────
# 3. 迷你训练 (20 steps, FlowMatching)
# ──────────────────────────────────────────────────────────────────
print('[3/3] Mini training (20 steps) …')

import diffuser.utils as utils

class Parser(utils.Parser):
    dataset: str = 'avoiding-synthetic'
    config: str  = 'config.avoiding-synthetic-fm'

args = Parser().parse_args(experiment='diffusion', seed=0)
# Force CPU/MPS and tiny run
args.device    = DEVICE
args.n_train_steps     = 20
args.n_steps_per_epoch = 20
args.batch_size        = 4
args.train_test_split  = 1.0
# Use a temp save dir
import tempfile
tmp_dir = tempfile.mkdtemp(prefix='dpcc_quick_test_')
args.savepath = tmp_dir

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset, horizon=args.horizon,
    normalizer=args.normalizer, preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding, max_path_length=args.max_path_length,
    include_returns=args.include_returns, returns_scale=args.max_path_length,
    discount=args.discount,
)
dataset = dataset_config()

model_config = utils.Config(
    args.model, savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon, transition_dim=dataset.observation_dim + dataset.action_dim,
    cond_dim=dataset.observation_dim, dim_mults=args.dim_mults,
    returns_condition=args.returns_condition, dim=args.dim,
    condition_dropout=args.condition_dropout,
    device=args.device,
)
diffusion_config = utils.Config(
    args.diffusion, savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon, observation_dim=dataset.observation_dim,
    action_dim=dataset.action_dim, goal_dim=dataset.goal_dim,
    n_timesteps=args.n_diffusion_steps, loss_type=args.loss_type,
    clip_denoised=args.clip_denoised, predict_epsilon=args.predict_epsilon,
    action_weight=args.action_weight, loss_discount=args.loss_discount,
    returns_condition=args.returns_condition,
    condition_guidance_w=args.condition_guidance_w,
    device=args.device,
)
trainer_config = utils.Config(
    utils.Trainer, savepath=(args.savepath, 'trainer_config.pkl'),
    train_test_split=args.train_test_split,
    ema_decay=args.ema_decay,
    n_train_steps=args.n_train_steps,
    n_steps_per_epoch=args.n_steps_per_epoch,
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    results_folder=args.savepath,
    train_device=args.device,
)

model    = model_config()
diffusion = diffusion_config(model)
trainer  = trainer_config(diffusion, dataset)

t0 = time.time()
trainer.train()
elapsed = time.time() - t0

print(f'  ✓ Training 20 steps completed in {elapsed:.1f}s')

import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

# ──────────────────────────────────────────────────────────────────
print('\n' + '═'*50)
print('  ✅  All tests passed! Environment is ready.')
print(f'  Device: {DEVICE}')
print('═'*50)
