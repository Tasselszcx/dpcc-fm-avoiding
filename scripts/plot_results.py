"""
Generate comparison figures and a full per-cell metric table (incl. time/step)
for the DDPM-vs-FM constrained-DPC experiment.

Reads the per-trial result arrays from logs/.../results/halfspace_<scene>/<variant>.npz
and writes PNG figures into figures/. Purely additive; does not touch eval/training.

Usage:
    source scripts/env_h800.sh
    python scripts/plot_results.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import diffuser.utils as utils

SEEDS = [0, 1, 2]
SCENES = ['top-left-hard', 'top-right-hard', 'both-hard']
EXPS = {'DDPM': 'avoiding-synthetic', 'FM': 'avoiding-synthetic-fm'}
# variant key -> on-disk npz name (FM lateproj20 was stored under its th0p2 alias)
VARIANTS = {
    'diffuser': 'diffuser',
    'dpcc-c-tightened': 'dpcc-c-tightened',
    'post_processing': 'post_processing',
    'dpcc-c-tightened-lateproj20': 'dpcc-c-tightened-lateproj20',
}
ALIAS = {'dpcc-c-tightened-lateproj20': 'dpcc-c-tightened-th0p2'}

FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)


def _parser(exp):
    ds = exp.removesuffix('-fm')
    class P(utils.Parser):
        dataset: str = ds
        config: str = 'config.' + exp
    return P


def load_cell(exp, scene, npz_name):
    Pr = _parser(exp)
    g, gc, vio, tt = [], [], [], []
    for s in SEEDS:
        args = Pr().parse_args(experiment='plan', seed=s)
        p = f'{args.savepath}/results/halfspace_{scene}/{npz_name}.npz'
        if not os.path.exists(p):
            return None
        d = np.load(p, allow_pickle=True)
        g += list(np.atleast_1d(d['n_success']).ravel())
        gc += list(np.atleast_1d(d['n_success_and_constraints']).ravel())
        vio += list(np.atleast_1d(d['n_violations']).ravel())
        tt += list(np.atleast_1d(d['avg_time']).ravel())
    return dict(goal=100 * np.mean(g), goalcons=100 * np.mean(gc),
                viol=float(np.mean(vio)), tstep=float(np.mean(tt)))


def gather():
    data = {}  # (model, scene, variant) -> metrics
    for mname, exp in EXPS.items():
        for scene in SCENES:
            for vkey, vname in VARIANTS.items():
                cell = load_cell(exp, scene, vname)
                if cell is None and vkey in ALIAS:
                    cell = load_cell(exp, scene, ALIAS[vkey])
                if cell is not None:
                    data[(mname, scene, vkey)] = cell
    return data


def fig_headline(data):
    """Grouped bars: goal+cons% per scene for the 3 headline methods."""
    methods = [
        ('DDPM', 'dpcc-c-tightened', 'DDPM dpcc-c-tightened', '#3b6fb6'),
        ('FM', 'dpcc-c-tightened', 'FM dpcc-c-tightened', '#c45a5a'),
        ('FM', 'dpcc-c-tightened-lateproj20', 'FM lateproj20 (ours)', '#4a9e5c'),
    ]
    x = np.arange(len(SCENES)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (m, v, label, c) in enumerate(methods):
        vals = [data.get((m, sc, v), {}).get('goalcons', np.nan) for sc in SCENES]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=c)
        for b, val in zip(bars, vals):
            if not np.isnan(val):
                ax.annotate(f'{val:.1f}', (b.get_x() + b.get_width() / 2, val),
                            ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(SCENES, fontsize=11)
    ax.set_ylabel('goal + constraints  (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title('Primary metric by scene: late projection lifts FM to DDPM level', fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{FIG_DIR}/fig_goalcons_by_scene.png', dpi=140)
    plt.close(fig)


def fig_fm_variants(data):
    """FM: all variants per scene (shows projection vs no-projection vs late)."""
    order = [('diffuser', 'diffuser (no proj)', '#999999'),
             ('post_processing', 'post_processing', '#d6a64a'),
             ('dpcc-c-tightened', 'dpcc-c-tightened', '#c45a5a'),
             ('dpcc-c-tightened-lateproj20', 'lateproj20 (ours)', '#4a9e5c')]
    x = np.arange(len(SCENES)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (v, label, c) in enumerate(order):
        vals = [data.get(('FM', sc, v), {}).get('goalcons', np.nan) for sc in SCENES]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=c)
        for b, val in zip(bars, vals):
            if not np.isnan(val):
                ax.annotate(f'{val:.0f}', (b.get_x() + b.get_width() / 2, val),
                            ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(SCENES, fontsize=11)
    ax.set_ylabel('goal + constraints  (%)', fontsize=12); ax.set_ylim(0, 100)
    ax.set_title('FM: projection variants compared (per scene)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{FIG_DIR}/fig_fm_variants.png', dpi=140)
    plt.close(fig)


def fig_quality_vs_cost(data):
    """Scatter: goal+cons% (avg over scenes) vs time/step for headline methods."""
    methods = [
        ('DDPM', 'dpcc-c-tightened', 'DDPM dpcc-c', '#3b6fb6', 'o'),
        ('FM', 'dpcc-c-tightened', 'FM dpcc-c', '#c45a5a', 's'),
        ('FM', 'dpcc-c-tightened-lateproj20', 'FM lateproj20 (ours)', '#4a9e5c', '*'),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for m, v, label, c, mk in methods:
        gc = np.nanmean([data.get((m, sc, v), {}).get('goalcons', np.nan) for sc in SCENES])
        ts = np.nanmean([data.get((m, sc, v), {}).get('tstep', np.nan) for sc in SCENES])
        ax.scatter(ts, gc, s=260, c=c, marker=mk, label=label, edgecolors='k', zorder=3)
        ax.annotate(f'  {label}\n  ({ts:.3f}s, {gc:.1f}%)', (ts, gc), fontsize=9, va='center')
    ax.set_xlabel('time / projection step  (s, lower = faster)', fontsize=12)
    ax.set_ylabel('goal + constraints  (%, higher = better)', fontsize=12)
    ax.set_title('Quality vs. projection cost (top-left = best)', fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc='center right')
    ax.margins(x=0.18, y=0.12)
    fig.tight_layout(); fig.savefig(f'{FIG_DIR}/fig_quality_vs_cost.png', dpi=140)
    plt.close(fig)


def print_table(data):
    print(f"\n{'model':5s} {'scene':15s} {'variant':28s} {'goal%':>6s} {'g+c%':>6s} {'viol':>6s} {'t/step':>8s}")
    for mname in EXPS:
        for scene in SCENES:
            for vkey in VARIANTS:
                c = data.get((mname, scene, vkey))
                if c is None:
                    print(f"{mname:5s} {scene:15s} {vkey:28s}   [missing]"); continue
                print(f"{mname:5s} {scene:15s} {vkey:28s} {c['goal']:6.1f} {c['goalcons']:6.1f} {c['viol']:6.1f} {c['tstep']:8.3f}")


if __name__ == '__main__':
    data = gather()
    print_table(data)
    fig_headline(data)
    fig_fm_variants(data)
    fig_quality_vs_cost(data)
    print(f"\nFigures written to {FIG_DIR}/:")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {FIG_DIR}/{f}")
