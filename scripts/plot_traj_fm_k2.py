import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base = 'logs/avoiding-d3il/plans/H8_K20_Dmodels.FlowMatching/all_seeds'
scenes = [
    ('top-left-hard',  base + '/top-left-hard_d3il_h8_fm_k2/dpcc-c-tightened.png'),
    ('top-right-hard', base + '/top-right-hard_d3il_h8_fm_k2/dpcc-c-tightened.png'),
    ('both-hard',      base + '/both-hard_d3il_h8_fm_k2/dpcc-c-tightened.png'),
]

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, (title, path) in zip(axes, scenes):
    ax.imshow(mpimg.imread(path))
    ax.set_title(title, fontsize=20)
    ax.axis('off')
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/traj_fm_k2_3scenes.pdf', bbox_inches='tight')
fig.savefig('figures/traj_fm_k2_3scenes.png', dpi=140, bbox_inches='tight')
print('SAVED figures/traj_fm_k2_3scenes.{pdf,png}')
