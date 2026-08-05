import os
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import diffuser.utils as utils

with open('config/projection_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)

exp = 'avoiding-d3il'
halfspace_variants = config['avoiding_halfspace_variants']  # ['top-right-hard','top-left-hard','both-hard']
ax_limits = config['ax_limits'][exp]

# Authoritative mapping copied verbatim from scripts/eval.py (lines 74-82):
#   top-left-hard  -> halfspace[0], obstacle[3]
#   top-right-hard -> halfspace[1], obstacle[4]
#   both-hard      -> halfspace[2]+[3], obstacle[5]
def constraints_for(v):
    hs = config['halfspace_constraints'][exp]
    ob = config['obstacle_constraints'][exp]
    if v == 'top-left-hard':
        return [hs[0]], [ob[3]]
    elif v == 'top-right-hard':
        return [hs[1]], [ob[4]]
    elif v == 'both-hard':
        return [hs[2], hs[3]], [ob[5]]
    else:
        raise ValueError(v)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, v in zip(axes, halfspace_variants):
    polytopic, obstacle = constraints_for(v)
    utils.plot_environment_constraints(exp, ax)
    utils.plot_halfspace_constraints(exp, polytopic, ax, ax_limits)
    for c in obstacle:
        ax.add_patch(matplotlib.patches.Circle(c['center'], c['radius'], color='b', alpha=0.2))
    ax.plot([ax_limits[0][0], ax_limits[0][1]], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)
    ax.set_xlim(ax_limits[0])
    ax.set_ylim(ax_limits[1])
    ax.set_facecolor([1, 1, 0.9])
    ax.set_title(v, fontsize=18)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/env_layout_3scenes.pdf', bbox_inches='tight')
fig.savefig('figures/env_layout_3scenes.png', dpi=150, bbox_inches='tight')
print('SAVED figures/env_layout_3scenes.{pdf,png}  order:', list(halfspace_variants))
