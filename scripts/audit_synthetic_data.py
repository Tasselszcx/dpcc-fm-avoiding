import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_synthetic_data import (
    GOAL_Y,
    OBSTACLES,
    SCENARIOS,
    START,
    _halfspace_line,
)


DATA_DIR = os.path.join(
    PROJECT_ROOT,
    'd3il',
    'environments',
    'dataset',
    'data',
    'avoiding_synthetic',
    'data',
)
REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')

ACTION_BOUNDS = {
    'tight': {
        'lower': np.array([-0.01, 0.0]),
        'upper': np.array([0.01, 0.01]),
    },
    'loose': {
        'lower': np.array([-0.012, 0.0]),
        'upper': np.array([0.012, 0.012]),
    },
}


def _empty_scenario_stats():
    return {
        'count': 0,
        'lengths': [],
        'goal_x': [],
        'goal_y': [],
        'min_obstacle_margin': [],
        'min_extra_obstacle_margin': [],
        'max_halfspace_violation': [],
        'max_abs_action': [],
        'max_action_bound_violation_tight': [],
        'max_action_bound_violation_loose': [],
        'nan_or_inf_trajectories': 0,
        'start_error': [],
    }


def _max_bound_violation(actions, lower, upper):
    lower_violation = np.maximum(0.0, lower - actions)
    upper_violation = np.maximum(0.0, actions - upper)
    return float(np.max(lower_violation + upper_violation))


def _max_halfspace_violation(path, halfspaces):
    max_violation = 0.0
    for constraint in halfspaces:
        slope, intercept, side = _halfspace_line(constraint)
        y_line = slope * path[:, 0] + intercept
        if side == 'below':
            violation = path[:, 1] - y_line
        elif side == 'above':
            violation = y_line - path[:, 1]
        else:
            raise ValueError(f'Unknown halfspace side: {side}')
        max_violation = max(max_violation, float(np.max(violation)))
    return max(0.0, max_violation)


def _min_margin(path, obstacles):
    if not obstacles:
        return float('inf')
    margins = []
    for center, radius in obstacles:
        margins.append(np.min(np.linalg.norm(path - center, axis=1) - radius))
    return float(np.min(margins))


def _summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {'min': None, 'mean': None, 'max': None, 'std': None}
    return {
        'min': float(np.min(arr)),
        'mean': float(np.mean(arr)),
        'max': float(np.max(arr)),
        'std': float(np.std(arr)),
    }


def audit_dataset():
    files = sorted(file for file in os.listdir(DATA_DIR) if file.endswith('.pkl'))
    if not files:
        raise FileNotFoundError(f'No .pkl files found in {DATA_DIR}')

    scenario_stats = defaultdict(_empty_scenario_stats)
    unknown_scenarios = Counter()

    for file in files:
        with open(os.path.join(DATA_DIR, file), 'rb') as f:
            data = pickle.load(f)

        scenario_name = data.get('metadata', {}).get('scenario', 'missing')
        if scenario_name not in SCENARIOS:
            unknown_scenarios[scenario_name] += 1
            continue

        robot = data['robot']
        path = np.asarray(robot['des_c_pos'])[:, :2]
        actions = np.diff(path, axis=0)
        scenario = SCENARIOS[scenario_name]
        stats = scenario_stats[scenario_name]

        stats['count'] += 1
        stats['lengths'].append(len(path))
        stats['goal_x'].append(float(path[-1, 0]))
        stats['goal_y'].append(float(path[-1, 1]))
        stats['min_obstacle_margin'].append(_min_margin(path, OBSTACLES))
        stats['min_extra_obstacle_margin'].append(_min_margin(path, scenario['extra_obstacles']))
        stats['max_halfspace_violation'].append(_max_halfspace_violation(path, scenario['halfspaces']))
        stats['max_abs_action'].append(float(np.max(np.abs(actions))))
        stats['max_action_bound_violation_tight'].append(
            _max_bound_violation(actions, ACTION_BOUNDS['tight']['lower'], ACTION_BOUNDS['tight']['upper'])
        )
        stats['max_action_bound_violation_loose'].append(
            _max_bound_violation(actions, ACTION_BOUNDS['loose']['lower'], ACTION_BOUNDS['loose']['upper'])
        )
        stats['start_error'].append(float(np.linalg.norm(path[0] - START)))

        arrays = [np.asarray(value) for value in robot.values() if isinstance(value, np.ndarray)]
        if any(not np.isfinite(arr).all() for arr in arrays):
            stats['nan_or_inf_trajectories'] += 1

    summary = {
        'data_dir': DATA_DIR,
        'total_files': len(files),
        'expected_total_files': 96 * len(SCENARIOS),
        'unknown_scenarios': dict(unknown_scenarios),
        'scenarios': {},
        'pass': True,
    }

    for scenario_name in SCENARIOS:
        stats = scenario_stats[scenario_name]
        goal_x = np.asarray(stats['goal_x'], dtype=np.float64)
        goal_y = np.asarray(stats['goal_y'], dtype=np.float64)
        goal_x_range = SCENARIOS[scenario_name]['goal_x']

        scenario_summary = {
            'count': stats['count'],
            'expected_count': 96,
            'length': _summarize(stats['lengths']),
            'goal_x': _summarize(stats['goal_x']),
            'goal_y': _summarize(stats['goal_y']),
            'start_error': _summarize(stats['start_error']),
            'min_obstacle_margin': _summarize(stats['min_obstacle_margin']),
            'min_extra_obstacle_margin': _summarize(stats['min_extra_obstacle_margin']),
            'max_halfspace_violation': _summarize(stats['max_halfspace_violation']),
            'max_abs_action': _summarize(stats['max_abs_action']),
            'max_action_bound_violation_tight': _summarize(stats['max_action_bound_violation_tight']),
            'max_action_bound_violation_loose': _summarize(stats['max_action_bound_violation_loose']),
            'nan_or_inf_trajectories': stats['nan_or_inf_trajectories'],
        }

        failures = []
        if stats['count'] != 96:
            failures.append('wrong_count')
        if goal_x.size and (np.min(goal_x) < goal_x_range[0] or np.max(goal_x) > goal_x_range[1]):
            failures.append('goal_x_out_of_range')
        if goal_y.size and (np.min(goal_y) < GOAL_Y[0] or np.max(goal_y) > GOAL_Y[1]):
            failures.append('goal_y_out_of_range')
        if stats['min_obstacle_margin'] and min(stats['min_obstacle_margin']) < -1e-9:
            failures.append('base_obstacle_collision')
        if stats['min_extra_obstacle_margin'] and min(stats['min_extra_obstacle_margin']) < -1e-9:
            failures.append('scenario_obstacle_collision')
        if stats['max_halfspace_violation'] and max(stats['max_halfspace_violation']) > 1e-9:
            failures.append('halfspace_violation')
        if stats['max_action_bound_violation_loose'] and max(stats['max_action_bound_violation_loose']) > 1e-9:
            failures.append('loose_action_bound_violation')
        if stats['nan_or_inf_trajectories'] > 0:
            failures.append('nan_or_inf')

        scenario_summary['failures'] = failures
        summary['scenarios'][scenario_name] = scenario_summary
        summary['pass'] = summary['pass'] and not failures

    if unknown_scenarios:
        summary['pass'] = False
    if len(files) != summary['expected_total_files']:
        summary['pass'] = False

    return summary


def write_reports(summary):
    os.makedirs(REPORT_DIR, exist_ok=True)
    json_path = os.path.join(REPORT_DIR, 'synthetic_data_quality_report.json')
    md_path = os.path.join(REPORT_DIR, 'synthetic_data_quality_report.md')

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    lines = [
        '# Synthetic Avoiding Data Quality Report',
        '',
        f"Data directory: `{summary['data_dir']}`",
        f"Total files: {summary['total_files']} / {summary['expected_total_files']}",
        f"Overall pass: {summary['pass']}",
        '',
        '| Scenario | Count | Len mean | Goal x | Min base margin | Min scenario margin | Max halfspace viol. | Max loose action viol. |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |',
    ]

    for scenario_name, stats in summary['scenarios'].items():
        lines.append(
            '| {name} | {count}/{expected} | {len_mean:.1f} | '
            '[{goal_min:.3f}, {goal_max:.3f}] | {base_margin:.4f} | {extra_margin:.4f} | '
            '{halfspace:.2e} | {action:.2e} |'.format(
                name=scenario_name,
                count=stats['count'],
                expected=stats['expected_count'],
                len_mean=stats['length']['mean'] or 0,
                goal_min=stats['goal_x']['min'] or 0,
                goal_max=stats['goal_x']['max'] or 0,
                base_margin=stats['min_obstacle_margin']['min'] or 0,
                extra_margin=stats['min_extra_obstacle_margin']['min'] or 0,
                halfspace=stats['max_halfspace_violation']['max'] or 0,
                action=stats['max_action_bound_violation_loose']['max'] or 0,
            )
        )

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return json_path, md_path


if __name__ == '__main__':
    report = audit_dataset()
    json_path, md_path = write_reports(report)
    print(json.dumps(report, indent=2))
    print(f'Wrote {json_path}')
    print(f'Wrote {md_path}')
    if not report['pass']:
        raise SystemExit(1)
