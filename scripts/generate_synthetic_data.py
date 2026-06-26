import os
import pickle
import numpy as np

# === 环境参数 ===
START = np.array([0.525, -0.280])
GOAL_Y = (0.365, 0.430)
Z_FIXED = 0.122  # z 轴固定值（从原始数据中观察到的）

# D3IL Avoiding 环境中的 6 个真实障碍物
OBSTACLES = [
    (np.array([0.50, -0.10]), 0.030),
    (np.array([0.425, 0.08]), 0.025),
    (np.array([0.575, 0.08]), 0.025),
    (np.array([0.35, 0.26]), 0.025),
    (np.array([0.50, 0.26]), 0.025),
    (np.array([0.65, 0.26]), 0.025),
]

# 与 config/projection_eval.yaml 中的三个测试场景对应。
SCENARIOS = {
    'top-right-hard': {
        'goal_x': (0.66, 0.76),
        'halfspaces': [
            [[0.2, -0.5], [0.6, 0.5], 'below'],
        ],
        'extra_obstacles': [
            (np.array([0.60, 0.08]), 0.08),
        ],
        'waypoints': [
            [0.525, -0.280],
            [0.60, -0.18],
            [0.72, -0.04],
            [0.75, 0.12],
            [0.73, 0.28],
            [0.70, 0.39],
        ],
    },
    'top-left-hard': {
        'goal_x': (0.28, 0.42),
        'halfspaces': [
            [[0.8, -0.5], [0.4, 0.5], 'below'],
        ],
        'extra_obstacles': [
            (np.array([0.40, 0.08]), 0.08),
        ],
        'waypoints': [
            [0.525, -0.280],
            [0.43, -0.20],
            [0.32, -0.06],
            [0.28, 0.12],
            [0.30, 0.28],
            [0.34, 0.39],
        ],
    },
    'both-hard': {
        'goal_x': (0.44, 0.56),
        'halfspaces': [
            [[0.8, -0.3], [0.575, 0.5], 'below'],
            [[0.2, -0.3], [0.425, 0.5], 'below'],
        ],
        'extra_obstacles': [
            (np.array([0.50, -0.09]), 0.08),
        ],
        'waypoints': [
            [0.525, -0.280],
            [0.39, -0.12],
            [0.40, 0.04],
            [0.51, 0.15],
            [0.43, 0.27],
            [0.50, 0.39],
        ],
    },
}

N_TRAJS_PER_SCENARIO = 96
N_TRAJS = N_TRAJS_PER_SCENARIO * len(SCENARIOS)
STEPS_RANGE = (70, 110)
SAFE_MARGIN = 0.030  # 额外安全距离
HALFSPACE_MARGIN = 0.015
MAX_ACTION_STEP = 0.0115


def _halfspace_line(constraint):
    p0, p1, side = constraint
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    slope = (p1[1] - p0[1]) / (p1[0] - p0[0])
    intercept = p0[1] - slope * p0[0]
    return slope, intercept, side


def _project_halfspaces(path, halfspaces, margin=HALFSPACE_MARGIN):
    """Project points back into the configured 2D halfspaces."""
    for constraint in halfspaces:
        slope, intercept, side = _halfspace_line(constraint)
        y_line = slope * path[:, 0] + intercept
        if side == 'below':
            violation = path[:, 1] - (y_line - margin)
            mask = violation > 0
            if np.any(mask):
                denom = slope ** 2 + 1
                path[mask, 0] += violation[mask] * slope / denom
                path[mask, 1] -= violation[mask] / denom
        elif side == 'above':
            violation = (y_line + margin) - path[:, 1]
            mask = violation > 0
            if np.any(mask):
                denom = slope ** 2 + 1
                path[mask, 0] -= violation[mask] * slope / denom
                path[mask, 1] += violation[mask] / denom
        else:
            raise ValueError(f'Unknown halfspace side: {side}')
    return path


def _push_outside_obstacles(path, obstacles, margin=SAFE_MARGIN):
    for center, radius in obstacles:
        diff = path - center
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        safe_radius = radius + margin
        mask = dist < safe_radius
        direction = diff / np.clip(dist, 1e-6, None)
        # If a point is exactly at the center, push it upward deterministically.
        direction = np.where(dist < 1e-6, np.array([[0.0, 1.0]]), direction)
        path = np.where(mask, center + direction * safe_radius, path)
    return path


def _sample_path_from_waypoints(rng, waypoints, goal):
    waypoints = np.asarray(waypoints, dtype=np.float64)
    waypoints[-1] = goal

    # Jitter intermediate waypoints to create demonstration diversity.
    jitter = rng.randn(*waypoints.shape) * np.array([0.020, 0.015])
    jitter[0] = 0
    jitter[-1] = 0
    waypoints = waypoints + jitter

    segment_lengths = np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1)
    proportions = segment_lengths / segment_lengths.sum()
    n_steps = rng.randint(*STEPS_RANGE)
    segment_steps = np.maximum(3, np.round(proportions * (n_steps - 1)).astype(int))
    segment_steps[-1] += (n_steps - 1) - segment_steps.sum()

    parts = []
    for i, n_seg in enumerate(segment_steps):
        t = np.linspace(0, 1, n_seg, endpoint=False)[:, None]
        parts.append(waypoints[i] + t * (waypoints[i + 1] - waypoints[i]))
    path = np.vstack(parts + [waypoints[-1:]])

    noise = rng.randn(len(path), 2) * np.array([0.004, 0.003])
    noise[0] = 0
    noise[-1] = 0
    path += np.cumsum(noise, axis=0) * 0.12
    path[0] = START
    path[-1] = goal
    return path


def _make_actions_bounds_friendly(path, max_step=MAX_ACTION_STEP):
    """Remove local backward-y wiggles and densify large action steps."""
    path = path.copy()
    goal = path[-1].copy()
    path[:, 1] = np.maximum.accumulate(path[:, 1])
    path[-1] = goal
    if path[-1, 1] < path[-2, 1]:
        path[-1, 1] = path[-2, 1]

    dense = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        delta = end - start
        n_insert = max(1, int(np.ceil(np.max(np.abs(delta)) / max_step)))
        for idx in range(1, n_insert + 1):
            dense.append(start + delta * idx / n_insert)
    return np.asarray(dense, dtype=np.float64)


def generate_trajectory(seed, scenario_name=None):
    rng = np.random.RandomState(seed)
    if scenario_name is None:
        scenario_names = list(SCENARIOS)
        scenario_name = scenario_names[seed % len(scenario_names)]
    scenario = SCENARIOS[scenario_name]

    goal = np.array([rng.uniform(*scenario['goal_x']), rng.uniform(*GOAL_Y)])
    path = _sample_path_from_waypoints(rng, scenario['waypoints'], goal)
    all_obstacles = OBSTACLES + scenario['extra_obstacles']

    # 交替平滑、半空间投影、障碍物排斥，确保三种测试场景都在训练分布内。
    for _ in range(30):
        path[1:-1] = 0.5 * path[1:-1] + 0.25 * (path[:-2] + path[2:])
        path = _project_halfspaces(path, scenario['halfspaces'])
        path = _push_outside_obstacles(path, all_obstacles)
        path[0] = START
        path[-1] = goal

    path = _make_actions_bounds_friendly(path)

    # 构建 pkl 数据（与原始格式完全一致）
    des_c_pos = np.zeros((len(path), 3))
    des_c_pos[:, :2] = path
    des_c_pos[:, 2] = Z_FIXED

    c_pos = des_c_pos + rng.randn(len(path), 3) * np.array([0.002, 0.002, 0.001])

    data = {'robot': {
        'time_stamp': np.arange(len(path), dtype=np.float64),
        'sim_step': np.arange(len(path), dtype=np.int64),
        'wall_clock': np.arange(len(path), dtype=np.int64),
        'j_pos': np.zeros((len(path), 7)),
        'j_vel': np.zeros((len(path), 7)),
        'c_pos': c_pos,
        'c_vel': np.zeros((len(path), 3)),
        'c_quat': np.tile([0, 1, 0, 0], (len(path), 1)).astype(np.float64),
        'des_c_pos': des_c_pos,
        'des_c_quat': np.tile([0, 1, 0, 0], (len(path), 1)).astype(np.float64),
        'des_j_pos': np.zeros((len(path), 7)),
    }, 'metadata': {
        'scenario': scenario_name,
    }}
    return data


def _check_trajectory(path, scenario_name):
    scenario = SCENARIOS[scenario_name]
    for constraint in scenario['halfspaces']:
        slope, intercept, side = _halfspace_line(constraint)
        y_line = slope * path[:, 0] + intercept
        if side == 'below' and np.any(path[:, 1] > y_line + 1e-6):
            return False
        if side == 'above' and np.any(path[:, 1] < y_line - 1e-6):
            return False
    for center, radius in OBSTACLES + scenario['extra_obstacles']:
        if np.any(np.linalg.norm(path - center, axis=1) < radius):
            return False
    return True


if __name__ == '__main__':
    # Use absolute path so the script works from any CWD
    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(_proj_root, 'd3il', 'environments', 'dataset', 'data',
                           'avoiding_synthetic', 'data')
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(out_dir):
        if file.endswith('.pkl'):
            os.remove(os.path.join(out_dir, file))

    idx = 0
    for scenario_name in SCENARIOS:
        for local_idx in range(N_TRAJS_PER_SCENARIO):
            seed = 1000 * list(SCENARIOS).index(scenario_name) + local_idx
            data = generate_trajectory(seed=seed, scenario_name=scenario_name)
            path = os.path.join(out_dir, f'env_{idx:03d}_{scenario_name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            idx += 1

    print(f'Generated {N_TRAJS} trajectories in {out_dir}/')

    # 验证
    first_file = sorted(file for file in os.listdir(out_dir) if file.endswith('.pkl'))[0]
    with open(os.path.join(out_dir, first_file), 'rb') as f:
        d = pickle.load(f)
    r = d['robot']
    print(f"Keys: {list(r.keys())}")
    print(f"des_c_pos shape: {r['des_c_pos'].shape}")
    print(f"Start: ({r['des_c_pos'][0,0]:.3f}, {r['des_c_pos'][0,1]:.3f})")
    print(f"End:   ({r['des_c_pos'][-1,0]:.3f}, {r['des_c_pos'][-1,1]:.3f})")

    # 检查每个场景的半空间和障碍物安全距离
    violations = {scenario_name: 0 for scenario_name in SCENARIOS}
    counts = {scenario_name: 0 for scenario_name in SCENARIOS}
    for file in os.listdir(out_dir):
        if not file.endswith('.pkl'):
            continue
        with open(os.path.join(out_dir, file), 'rb') as f:
            d = pickle.load(f)
        scenario_name = d['metadata']['scenario']
        counts[scenario_name] += 1
        pos = d['robot']['des_c_pos'][:, :2]
        if not _check_trajectory(pos, scenario_name):
            violations[scenario_name] += 1
    for scenario_name in SCENARIOS:
        print(f"{scenario_name}: {counts[scenario_name]} trajectories, violations={violations[scenario_name]}")
