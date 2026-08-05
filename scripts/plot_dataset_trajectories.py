import os, pickle, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agents.utils.sim_path import sim_framework_path
import diffuser.utils as utils

with open('config/projection_eval.yaml') as f:
    config = yaml.safe_load(f)

exp = 'avoiding-d3il'
ax_limits = config['ax_limits'][exp]

# Load 96 d3il human demos (same loader as diffuser.datasets.d4rl)
data_dir = sim_framework_path('environments/dataset/data/avoiding/data')
trajs = []
for fn in sorted(os.listdir(data_dir)):
    with open(os.path.join(data_dir, fn), 'rb') as f:
        env_state = pickle.load(f)
        trajs.append(env_state['robot']['des_c_pos'][:, :2])
print(f'Loaded {len(trajs)} trajectories')

fig, ax = plt.subplots(1, 1, figsize=(8, 9))
ax.set_facecolor([1, 1, 0.9])
utils.plot_environment_constraints(exp, ax)  # draws 6 obstacle circles + goal line
for t in trajs:
    ax.plot(t[:, 0], t[:, 1], color='b', alpha=0.45, linewidth=0.6)
ax.set_xlim(ax_limits[0]); ax.set_ylim(ax_limits[1])
ax.set_aspect('equal', adjustable='box')  # keep geometry undistorted (matches env_layout)
ax.set_title(f'{exp}: {len(trajs)} human demonstrations', fontsize=16)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/dataset_trajs.pdf', bbox_inches='tight')
fig.savefig('figures/dataset_trajs.png', dpi=180, bbox_inches='tight')
print('SAVED figures/dataset_trajs.{pdf,png}')
