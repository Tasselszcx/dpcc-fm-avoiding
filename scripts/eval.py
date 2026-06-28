import time
import yaml
import os
import sys
import torch
from copy import copy
import minari
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import diffuser.utils as utils
from diffuser.sampling import Policy, Projector
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

# Load configuration
with open('config/projection_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)

# General
exps = config['exps']
seeds = config['seeds']
projection_variants = config['projection_variants']
halfspace_variants = config['avoiding_halfspace_variants'] if 'avoiding' in exps[0] else ['top-left']
n_trials = config['n_trials']
plot_how_many = config['plot_how_many']

if os.environ.get('EVAL_EXPS'):
    exps = [exp.strip() for exp in os.environ['EVAL_EXPS'].split(',') if exp.strip()]
if os.environ.get('EVAL_SEEDS'):
    seeds = [int(seed) for seed in os.environ['EVAL_SEEDS'].split(',') if seed.strip()]
if os.environ.get('EVAL_N_TRIALS'):
    n_trials = int(os.environ['EVAL_N_TRIALS'])
if os.environ.get('EVAL_HALFSPACE_VARIANTS'):
    halfspace_variants = [
        variant.strip()
        for variant in os.environ['EVAL_HALFSPACE_VARIANTS'].split(',')
        if variant.strip()
    ]
if os.environ.get('EVAL_PROJECTION_VARIANTS'):
    projection_variants = [
        variant.strip()
        for variant in os.environ['EVAL_PROJECTION_VARIANTS'].split(',')
        if variant.strip()
    ]

if torch.cuda.is_available():
    _AUTO_DEVICE = 'cuda'
else:
    # Projection uses float64 matrices; MPS does not support float64 tensors.
    _AUTO_DEVICE = 'cpu'


def _tag_save_path(path):
    tag = os.environ.get('EVAL_SAVE_TAG')
    return f'{path}_{tag}' if tag else path

# Constraint projection
constraint_types = config['constraint_types']

total_groups = len(exps) * len(halfspace_variants) * len(seeds) * len(projection_variants)
current_group = 0
eval_start_time = time.time()

for exp in exps:
    dataset_exp = exp.removesuffix('-fm')
    for halfspace_variant in halfspace_variants:
        robot_name = dataset_exp.split('-')[0]
        if halfspace_variant == 'top-left-hard':
            polytopic_constraints = [config['halfspace_constraints'][dataset_exp][0]]
            obstacle_constraints = [config['obstacle_constraints'][dataset_exp][3]]
        elif halfspace_variant == 'top-right-hard':
            polytopic_constraints = [config['halfspace_constraints'][dataset_exp][1]]
            obstacle_constraints = [config['obstacle_constraints'][dataset_exp][4]]
        elif halfspace_variant == 'both-hard':
            polytopic_constraints = [config['halfspace_constraints'][dataset_exp][2], config['halfspace_constraints'][dataset_exp][3]]
            obstacle_constraints = [config['obstacle_constraints'][dataset_exp][5]]

        bounds = config['bounds'][dataset_exp]
        ax_limits = config['ax_limits'][dataset_exp]
        enlarge_constraints = config['enlarge_constraints'][robot_name]
        dt = config['dt'][robot_name]

        class Parser(utils.Parser):
            dataset: str = dataset_exp
            config: str = 'config.' + exp

        figs_all_seeds, axes_all_seeds = zip(*[plt.subplots(1, 1, figsize=(9, 10)) for _ in range(len(projection_variants))])
        figs_all_seeds = list(figs_all_seeds)
        axes_all_seeds = list(axes_all_seeds)

        for seed in seeds:
            args = Parser().parse_args(experiment='plan', seed=seed)
            if 'cuda' in getattr(args, 'device', '') and not torch.cuda.is_available():
                args.device = _AUTO_DEVICE
                print(f'[ eval ] No CUDA found, falling back to device: {args.device}')
            if os.environ.get('EVAL_DEVICE'):
                args.device = os.environ['EVAL_DEVICE']
            if os.environ.get('EVAL_DIFFUSION_EPOCH'):
                epoch = os.environ['EVAL_DIFFUSION_EPOCH']
                args.diffusion_epoch = int(epoch) if epoch.isdigit() else epoch
            if os.environ.get('EVAL_MAX_EPISODE_LENGTH'):
                args.max_episode_length = int(os.environ['EVAL_MAX_EPISODE_LENGTH'])

            save_path = _tag_save_path(
                f'{args.savepath}/results/halfspace_{halfspace_variant}' if 'avoiding' in exp else f'{args.savepath}/results'
            )

            # Get model
            diffusion_experiment = utils.load_diffusion(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed), epoch=args.diffusion_epoch, device=args.device)
            diffusion = diffusion_experiment.diffusion
            dataset = diffusion_experiment.dataset

            if 'pointmaze' in dataset_exp or 'antmaze' in dataset_exp:
                minari_dataset = minari.load_dataset(dataset_exp, download=True)
                env = minari_dataset.recover_environment(eval_env=True) if 'pointmaze' in dataset_exp else minari_dataset.recover_environment()    # Set render_mode='human' to visualize the environment
            elif 'avoiding' in dataset_exp:
                env = ObstacleAvoidanceEnv()
                env.start()

            if robot_name == 'pointmaze': env.env.env.env.point_env.frame_skip = 2
            if robot_name == 'antmaze': env.env.env.env.ant_env.frame_skip = 5

            obs_indices = config['observation_indices'][robot_name]
            act_indices = config['action_indices'][robot_name]

            # Create projector
            if diffusion.action_dim > 0:
                trajectory_dim = diffusion.transition_dim - diffusion.goal_dim
                action_dim = diffusion.action_dim
                diffuser_variant = 'states_actions'
                obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
                act_obs_indices = {**act_indices, **obs_indices_updated}
            else:
                trajectory_dim = diffusion.observation_dim - diffusion.goal_dim
                action_dim = 0
                diffuser_variant = 'states'
                act_obs_indices = obs_indices

            # -------------------- Load constraints ------------------
            constraint_list = []
            constraint_list_tightened = []
            # Halfspace constraints
            constraint_list_polytopic_not_tightened = []
            if 'halfspace' in constraint_types:
                for constraint in polytopic_constraints:
                    constraint_list.append(('ineq', utils.formulate_halfspace_constraints(constraint, 0, trajectory_dim, act_obs_indices)))
                    constraint_list_tightened.append(('ineq', utils.formulate_halfspace_constraints(constraint, enlarge_constraints, trajectory_dim, act_obs_indices)))
                    constraint_list_polytopic_not_tightened.append(('ineq', utils.formulate_halfspace_constraints(constraint, 0, trajectory_dim, act_obs_indices)))

            # Bounds
            if 'bounds' in constraint_types:
                lower_bound, upper_bound = utils.formulate_bounds_constraints(constraint_types, bounds, trajectory_dim, act_obs_indices)
                constraint_list.extend([['lb', lower_bound], ['ub', upper_bound]])
                constraint_list_tightened.extend([['lb', lower_bound], ['ub', upper_bound]])

            # Obstacle constraints
            if 'obstacles' in constraint_types:
                for constr in obstacle_constraints:
                    constraint_list.append([constr['type'], [act_obs_indices[constr['dimensions'][0]], act_obs_indices[constr['dimensions'][1]]], constr['center'], constr['radius']])
                    constraint_list_tightened.append([constr['type'], [act_obs_indices[constr['dimensions'][0]], act_obs_indices[constr['dimensions'][1]]], constr['center'], constr['radius'] + enlarge_constraints])

            # Dynamics constraints
            constraint_list_without_prior = copy(constraint_list)
            constraint_list_without_prior_tightened = copy(constraint_list_tightened)
            dynamics_constraints = []
            if 'dynamics' in constraint_types: dynamics_constraints = utils.formulate_dynamics_constraints(dataset_exp, act_obs_indices, action_dim)

            for constraint in dynamics_constraints:
                constraint_list.append(constraint)
                constraint_list_tightened.append(constraint)

            # -------------------- Run experiments ------------------
            env_seeds = config['env_seeds'][dataset_exp] if 'pointmaze-umaze' in dataset_exp else np.arange(100)
            fig_all, ax_all = plt.subplots(
                min(n_trials, plot_how_many),
                len(projection_variants),
                figsize=(10 * len(projection_variants), 10 * min(n_trials, plot_how_many)),
                squeeze=False,
            )

            for variant_idx, variant in enumerate(projection_variants):
                result_file = f'{save_path}/{variant}.npz'
                if os.path.exists(result_file):
                    current_group += 1
                    print(f'Skipping existing result: {result_file}')
                    continue

                current_group += 1
                elapsed = time.time() - eval_start_time
                if current_group > 1:
                    eta = elapsed / (current_group - 1) * (total_groups - current_group + 1)
                    eta_str = f'{int(eta//60)}m{int(eta%60)}s'
                else:
                    eta_str = '估算中...'
                elapsed_str = f'{int(elapsed//60)}m{int(elapsed%60)}s'
                print(f'\n[{current_group}/{total_groups}] 已用时: {elapsed_str} | 预计剩余: {eta_str}')
                print(f'------------------------Running {exp} - {halfspace_variant} - {variant} ({seed})----------------------------')

                gradient = True if 'gradient' in variant else False

                if 'model_free' in variant and 'tightened' in variant:
                    constraints = constraint_list_without_prior_tightened
                elif 'model_free' in variant and not 'tightened' in variant:
                    constraints = constraint_list_without_prior
                elif not 'model_free' in variant and 'tightened' in variant:
                    constraints = constraint_list_tightened
                else:
                    constraints = constraint_list

                delta_t = dt
                if 'dt0p25' in variant:
                    delta_t = 0.25 * dt
                elif 'dt0p5' in variant:
                    delta_t = 0.5 * dt
                elif 'dt2p0' in variant:
                    delta_t = 2.0 * dt
                elif 'dt4p0' in variant:
                    delta_t = 4.0 * dt

                # --- FM projection-schedule sweep (additive; default behavior unchanged) ---
                # Optional variant-name suffixes to control *when/how often* projection runs,
                # mirroring the existing 'dt*' suffix pattern above. These only take effect when
                # the suffix is present in the variant name, so the original variants
                # (e.g. 'dpcc-c-tightened') keep their exact previous behavior.
                #   peN   -> project_every = N        (sparser projection during ODE integration)
                #   thX p Y -> diffusion_timestep_threshold = X.Y (e.g. 'th0p2' -> only project in
                #             the last 20% of integration, i.e. project later/closer to data)
                override_project_every = None
                for _pe in (2, 3, 4, 5):
                    if f'pe{_pe}' in variant:
                        override_project_every = _pe
                        break
                override_threshold = None
                import re as _re
                # Two equivalent spellings for the projection-window threshold:
                #   'lateprojNN' -> project only in the last NN% of integration
                #                   (more readable, e.g. 'lateproj20' = last 20%)
                #   'thXpY'      -> diffusion_timestep_threshold = X.Y (original spelling,
                #                   kept as a backward-compatible alias; 'th0p2' == 'lateproj20')
                _ml = _re.search(r'lateproj(\d+)', variant)
                if _ml:
                    override_threshold = int(_ml.group(1)) / 100.0
                else:
                    _m = _re.search(r'th(\d)p(\d+)', variant)
                    if _m:
                        override_threshold = float(f'{_m.group(1)}.{_m.group(2)}')

                # Optional alternative NLP solver for the projection step.
                #   'trustconstr' in the variant name -> scipy 'trust-constr'
                #   (interior-point / trust-region) instead of the default 'SLSQP'.
                scipy_method = 'trust-constr' if 'trustconstr' in variant else 'SLSQP'

                # Create projector
                project_every = config.get('flow_matching_project_every', 1) if diffusion.__class__.__name__ == 'FlowMatching' else 1
                if override_project_every is not None:
                    project_every = override_project_every
                projector_kwargs = {}
                if override_threshold is not None:
                    projector_kwargs['diffusion_timestep_threshold'] = override_threshold
                projector = Projector(horizon=args.horizon, transition_dim=trajectory_dim, action_dim=action_dim, goal_dim=diffusion.goal_dim, constraint_list=constraints, normalizer=dataset.normalizer,
                                        gradient=gradient, gradient_weights=[1, 0.5, 2], variant=diffuser_variant, dt=delta_t, cost_dims=None, device=args.device, solver='scipy',
                                        scipy_maxiter=config.get('projection_scipy_maxiter', 1000), project_every=project_every, scipy_method=scipy_method, **projector_kwargs)
                projector = None if variant == 'diffuser' else projector

                trajectory_selection = 'random'
                if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
                if 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'

                # Create policy
                policy = Policy(model=diffusion, normalizer=dataset.normalizer, preprocess_fns=args.preprocess_fns, 
                                test_ret=args.test_ret, projector=projector, trajectory_selection=trajectory_selection)    

                # Run policy
                fig, ax = plt.subplots(
                    min(n_trials, plot_how_many),
                    6,
                    figsize=(30, 5 * min(n_trials, plot_how_many)),
                    squeeze=False,
                )
                fig.suptitle(f'{exp} - {variant}')

                save_samples_every = args.horizon // 2

                # Store a few sampled trajectories
                sampled_trajectories_all = []        
                n_success = np.zeros(n_trials)
                n_success_and_constraints = np.zeros(n_trials)
                n_steps = np.zeros(n_trials)
                n_violations = np.zeros(n_trials)
                total_violations = np.zeros(n_trials)
                avg_time = np.zeros(n_trials)
                collision_free_completed = np.ones(n_trials)
                pos_tracking_errors = np.zeros((n_trials, args.max_episode_length - 1))
                for i in range(n_trials):
                    print(f'\r  Trial {i+1}/{n_trials}', end='', flush=True)
                    torch.manual_seed(i)
                    env_seed = env_seeds[i] if ('pointmaze-umaze' in dataset_exp) else i
                    
                    # Reset environment
                    if 'avoiding' in dataset_exp:
                        obs = env.reset()
                        action = env.robot_state()[:2]
                        fixed_z = env.robot_state()[2:]
                    else:
                        obs, _ = env.reset(seed=env_seed)
                    
                    if 'pointmaze' in dataset_exp:
                        obs = np.concatenate((obs['observation'], obs['desired_goal']))
                    elif 'antmaze' in dataset_exp:
                        obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
                    elif 'avoiding' in dataset_exp:
                        obs = np.concatenate((action[:2], obs))           
                        
                    obs_buffer = []
                    action_buffer = []

                    sampled_trajectories = []
                    disable_projection = False
                    for _ in range(args.max_episode_length):
                        # Check if a safety constraint is violated
                        violated_this_timestep = 0
                        if 'halfspace' in constraint_types:
                            for constraint in constraint_list_polytopic_not_tightened:
                                if constraint[0] == 'ineq':
                                    c, d = constraint[1]
                                    obs_to_check = obs[:-diffusion.goal_dim] if diffusion.goal_dim > 0 else obs
                                    if obs_to_check @ c[action_dim:] >= d:
                                        violated_this_timestep = 1
                                        total_violations[i] += obs_to_check @ c[action_dim:] - d
                                        collision_free_completed[i] = 0

                        if 'obstacles' in constraint_types:
                            for constraint in obstacle_constraints:
                                if np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center']) < constraint['radius']:
                                    violated_this_timestep = 1
                                    total_violations[i] += constraint['radius'] - np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center'])
                                    collision_free_completed[i] = 0
                        
                        if _ > 0 and 'bounds' in constraint_types:
                            act_obs = np.concatenate((action, obs)) if action_dim > 0 else obs
                            total_violations[i] += np.sum(np.maximum(0, act_obs - upper_bound)) + np.sum(np.maximum(0, lower_bound - act_obs))

                        n_violations[i] += violated_this_timestep
                        
                        # Calculate action
                        start = time.time()
                        action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, horizon=args.horizon, disable_projection=disable_projection)
                        avg_time[i] += time.time() - start

                        # Step environment
                        if 'avoiding' in dataset_exp:
                            next_pos_des = action + obs[:2] 
                            obs, rew, terminated, info = env.step(np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0))
                            success = info[1]
                        else:
                            obs, rew, terminated, truncated, info = env.step(action)
                            success = info['success']

                        if 'pointmaze' in dataset_exp:
                            obs = np.concatenate((obs['observation'], obs['desired_goal']))
                        elif 'antmaze' in dataset_exp:
                            obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
                        elif 'avoiding' in dataset_exp:
                            obs = np.concatenate((next_pos_des[:2], obs))

                        # Get tracking error
                        if _ >= 1:
                            pos_tracking_errors[i, _-1] = np.linalg.norm(obs[obs_indices['x']:obs_indices['y']+1] - desired_next_pos)
                        desired_next_pos = samples.observations[0, 1, [obs_indices['x'], obs_indices['y']]]

                        if _ % save_samples_every == 0:
                            sampled_trajectories.append(samples.observations[:, :, :])

                        obs_buffer.append(obs)
                        action_buffer.append(action)
                        if success: n_success[i] = 1
                        if (terminated or _ == args.max_episode_length - 1) and (not success): collision_free_completed[i] = 0

                        # if success or terminated or truncated or _ == n_timesteps - 1:
                        if success or terminated or _ == args.max_episode_length - 1:
                            n_steps[i] = _
                            avg_time[i] /= _
                            if success and collision_free_completed[i]: n_success_and_constraints[i] = 1
                            break

                    sampled_trajectories_all.append(sampled_trajectories)
                    if i >= plot_how_many:     # Plot only the first n trials
                        continue
                    obs_array = np.asarray(obs_buffer)
                    if obs_array.ndim != 2 or obs_array.shape[0] == 0:
                        continue
                    # plot_states = ['x', 'y', 'vx', 'vy'] if 'maze' in exp else ['x', 'y']
                    plot_states = ['x', 'y', 'x_des', 'y_des']

                    for j in range(len(plot_states)):
                        ax[i, j].plot(obs_array[:, obs_indices[plot_states[j]]])
                        ax[i, j].set_title(plot_states[j])
                    
                    axes = [ax[i, 4], ax_all[i, variant_idx]]
                    for curr_ax in axes:
                        curr_ax.plot(obs_array[:, obs_indices['x']], obs_array[:, obs_indices['y']], 'k')
                        curr_ax.plot(obs_array[0, obs_indices['x']], obs_array[0, obs_indices['y']], 'go', label='Start')            # Start
                        # if 'maze' in exp: curr_ax.plot(np.array(obs_buffer)[0, obs_indices['goal_x']], np.array(obs_buffer)[0, obs_indices['goal_y']], 'ro', label='Goal')   # Goal
                        curr_ax.set_xlim(ax_limits[0])
                        curr_ax.set_ylim(ax_limits[1])

                    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                    axes_all_seeds[variant_idx].plot(obs_array[:, obs_indices['x']], obs_array[:, obs_indices['y']], colors[seed % len(colors)], linewidth=2)
                    
                    axes = [ax[i, 5], ax_all[i, variant_idx]]
                    for __ in range(len(sampled_trajectories_all[i])):          # Iterate over timesteps of sampled trajectories
                        for ___ in range(min(args.batch_size, 4)):              # Iterate over batch
                            for curr_ax in axes:
                                curr_ax.plot(sampled_trajectories_all[i][__][___, :args.horizon, obs_indices['x']], sampled_trajectories_all[i][__][___, :args.horizon, obs_indices['y']], 'b')
                                curr_ax.plot(sampled_trajectories_all[i][__][___, 0, obs_indices['x']], sampled_trajectories_all[i][__][___, 0, obs_indices['y']], 'go', label='Start')    # Current state
                    # if 'maze' in exp: ax[i, 5].plot(np.array(obs_buffer)[0, obs_indices['goal_x']], np.array(obs_buffer)[0, obs_indices['goal_y']], 'ro', label='Goal')   # Goal
                    ax[i, 5].set_xlim(ax_limits[0])
                    ax[i, 5].set_ylim(ax_limits[1])

                    # Plot constraints
                    axes = [ax[i, 4], ax[i, 5], ax_all[i, variant_idx]]
                    for curr_ax in axes:
                        utils.plot_environment_constraints(dataset_exp, curr_ax)

                        if 'halfspace' in constraint_types: utils.plot_halfspace_constraints(dataset_exp, polytopic_constraints, curr_ax, ax_limits)

                        if 'obstacles' in constraint_types:
                            for constraint in obstacle_constraints:
                                curr_ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'], color='b', alpha=0.2))

                print(f'\n  Trial {n_trials}/{n_trials} 完成')
                print(f'Success rate: {np.mean(n_success)}')
                print(f'Constraints satisfied: {np.mean(collision_free_completed)}')
                print(f'Success rate (goal and constraints): {np.mean(n_success_and_constraints)}')
                print(f'Avg number of steps: {(np.mean(n_steps[n_success > 0]) if np.sum(n_success) > 0 else 0):.2f} +- {(np.std(n_steps[n_success > 0]) if np.sum(n_success) > 0 else 0):.2f}')
                print(f'Avg number of constraint violations: {np.mean(n_violations):.2f} +- {np.std(n_violations):.2f}')
                print(f'Avg total violation: {np.mean(total_violations):.3f} +- {np.std(total_violations):.3f}')
                print(f'Average computation time per step: {np.mean(avg_time):.3f}')
                if variant == 'diffuser': print(f'Tracking error: {np.max(pos_tracking_errors):.3f}')

                save_path = _tag_save_path(
                    f'{args.savepath}/results/halfspace_{halfspace_variant}' if 'avoiding' in exp else f'{args.savepath}/results'
                )
                os.makedirs(save_path, exist_ok=True)
                if config['write_to_file']:
                    np.savez(f'{save_path}/{variant}.npz', 
                            n_success=n_success, 
                            n_success_and_constraints=n_success_and_constraints,
                            n_steps=n_steps, 
                            n_violations=n_violations, 
                            total_violations=total_violations, 
                            avg_time=avg_time, 
                            collision_free_completed=collision_free_completed, 
                            args=args)

                fig.savefig(f'{save_path}/{variant}.png')   
                plt.close(fig)

                ax_all[0, variant_idx].set_title(variant)
            fig_all.savefig(f'{save_path}/all.png')
            plt.close(fig_all)
            env.close()
        
        variant_idx = 0
        path = _tag_save_path(f'{os.path.dirname(args.savepath)}/all_seeds/{halfspace_variant}')
        os.makedirs(path, exist_ok=True)
        for fig, ax in zip(figs_all_seeds, axes_all_seeds):
            ax.set_xlim(ax_limits[0])
            ax.set_ylim(ax_limits[1])
            ax.set_facecolor([1, 1, 0.9])
            utils.plot_environment_constraints(dataset_exp, ax)
            if 'halfspace' in constraint_types: utils.plot_halfspace_constraints(dataset_exp, polytopic_constraints, ax, ax_limits, enlarge_constraints=enlarge_constraints)
            if 'obstacles' in constraint_types:
                for constraint in obstacle_constraints:
                    ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'], color='b', alpha=0.2))
                    ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'] + enlarge_constraints, color='b', alpha=0.1, linestyle='--'))
            fig.savefig(f'{path}/{projection_variants[variant_idx]}.png', bbox_inches='tight')
            fig.savefig(f'{path}/{projection_variants[variant_idx]}.pdf', bbox_inches='tight', format='pdf')
            plt.close(fig)
            variant_idx += 1
