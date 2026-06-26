import torch
import time
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import diffuser.utils as utils

exp = os.environ.get('TRAIN_EXP', 'avoiding-synthetic-fm')
dataset_name = exp.split('-fm')[0]  # 'avoiding-synthetic-fm' -> 'avoiding-synthetic'

seeds = [int(seed) for seed in os.environ.get('TRAIN_SEEDS', '0,1,2').split(',')]
total_start = time.time()

# 设备自动检测
if torch.cuda.is_available():
    _AUTO_DEVICE = 'cuda'
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    _AUTO_DEVICE = 'mps'
    print(f'Device: Apple MPS (M-series GPU)')
else:
    _AUTO_DEVICE = 'cpu'
    print(f'Device: CPU (no GPU detected)')
print(f'Seeds: {seeds}')
print(f'Experiment: {exp}')
print('='*60)

class Parser(utils.Parser):
    dataset: str = dataset_name
    config: str = 'config.' + exp

for seed in seeds:
    seed_start = time.time()
    print(f'\n{"="*60}')
    print(f'开始训练 Seed {seed} ({seeds.index(seed)+1}/{len(seeds)})')
    print(f'{"="*60}')
    args = Parser().parse_args(experiment='diffusion', seed=seed)
    if os.environ.get('TRAIN_STEPS'):
        args.n_train_steps = int(float(os.environ['TRAIN_STEPS']))
    if os.environ.get('TRAIN_STEPS_PER_EPOCH'):
        args.n_steps_per_epoch = int(float(os.environ['TRAIN_STEPS_PER_EPOCH']))

    # 覆盖 device：如果 config 设置了 cuda 但本机没有 CUDA，自动降级
    if 'cuda' in getattr(args, 'device', '') and not torch.cuda.is_available():
        args.device = _AUTO_DEVICE
        print(f'[ train ] No CUDA found, falling back to device: {args.device}')

    # args.seed = seed
    torch.manual_seed(args.seed)

    # Get dataset
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=args.dataset,
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        include_returns=args.include_returns,
        # returns_scale=args.returns_scale,
        returns_scale=args.max_path_length,               # Because the reward is <= 1 in each timestep
        # returns_scale=args.n_diffusion_steps,
        discount=args.discount,
    )

    dataset = dataset_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    goal_dim = dataset.goal_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#

    if args.diffusion == 'models.GaussianInvDynDiffusion':
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            horizon=args.horizon,
            transition_dim=observation_dim,
            cond_dim=observation_dim,
            dim_mults=args.dim_mults,
            returns_condition=args.returns_condition,
            dim=args.dim,
            condition_dropout=args.condition_dropout,
            device=args.device,
        )

        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, 'diffusion_config.pkl'),
            horizon=args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            goal_dim=dataset.goal_dim,
            n_timesteps=args.n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            hidden_dim=args.hidden_dim,
            ## loss weighting
            action_weight=args.action_weight,
            loss_discount=args.loss_discount,
            returns_condition=args.returns_condition,
            condition_guidance_w=args.condition_guidance_w,
            device=args.device,
        )
    else:
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            horizon=args.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=args.dim_mults,
            returns_condition=args.returns_condition,
            dim=args.dim,
            condition_dropout=args.condition_dropout,
            device=args.device,
        )

        diffusion_kwargs = dict(
            savepath=(args.savepath, 'diffusion_config.pkl'),
            horizon=args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            goal_dim=dataset.goal_dim,
            n_timesteps=args.n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            ## loss weighting
            action_weight=args.action_weight,
            loss_discount=args.loss_discount,
            returns_condition=args.returns_condition,
            condition_guidance_w=args.condition_guidance_w,
            device=args.device,
        )
        if hasattr(args, 'ode_solver'):
            diffusion_kwargs['ode_solver'] = args.ode_solver
        diffusion_config = utils.Config(args.diffusion, **diffusion_kwargs)

    # Create trainer
    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
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

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset)

    # 自动断点续训：查找步数最大的 checkpoint
    import glob
    import os
    import re

    latest_step = 0
    checkpoints = glob.glob(os.path.join(args.savepath, 'state_*.pt'))
    if checkpoints:
        step_checkpoints = []
        for cp in checkpoints:
            match = re.search(r'state_(\d+)\.pt', cp)
            if match:
                step_checkpoints.append((int(match.group(1)), cp))

        if step_checkpoints:
            latest_step, latest_checkpoint = max(step_checkpoints, key=lambda x: x[0])
            print(f'检测到 Checkpoint: {latest_checkpoint}，正在从步数 {latest_step} 恢复训练...')
            trainer.load(latest_step)

    # 动态调整训练步数，确保总步数不超过 args.n_train_steps
    remaining_steps = max(0, int(args.n_train_steps - latest_step))
    if remaining_steps == 0:
        print(f'当前已完成 {latest_step} 步，已达到或超过目标总步数 {args.n_train_steps}。跳过训练。')
        continue

    print(f'剩余训练步数: {remaining_steps}，目标总步数: {args.n_train_steps}')
    trainer.n_train_steps = remaining_steps # 修正 Trainer 的剩余步数
    trainer.train()

    # 记录本 seed 训练时间和 GPU 占用
    seed_elapsed = time.time() - seed_start
    print(f'\n--- Seed {seed} 训练完成 ---')
    print(f'用时: {int(seed_elapsed//60)}m {int(seed_elapsed%60)}s')
    if torch.cuda.is_available():
        print(f'GPU 峰值显存: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')
        print(f'GPU 缓存显存: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB')
        torch.cuda.reset_peak_memory_stats()

total_elapsed = time.time() - total_start
print(f'\n{"="*60}')
print(f'全部训练完成! 总用时: {int(total_elapsed//3600)}h {int(total_elapsed%3600//60)}m {int(total_elapsed%60)}s')
print(f'{"="*60}')
