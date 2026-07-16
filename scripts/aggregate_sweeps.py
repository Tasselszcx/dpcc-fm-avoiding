#!/usr/bin/env python
"""
Aggregate K-sweep npz results into per-method CSV tables + comparison charts.

Three methods share the same eval metrics but live under different paths/tags:
  FM          : Dmodels.FlowMatching/<seed>/results/halfspace_<scene>_k<K>/
                (K=20 = the UNTAGGED halfspace_<scene>/ dir)
  DDIM        : Dmodels.GaussianDiffusion (H8_K20)/<seed>/results/
                halfspace_<scene>_ddim<K>/
  native DDPM : H8_K<K>_Dmodels.GaussianDiffusion/<seed>/results/
                halfspace_<scene>_retrain<K>/   (K=20 = untagged K=20 per-step)

Each npz stores per-trial arrays (len = n_trials): n_success,
n_success_and_constraints (=JSR), n_violations, avg_time. We average across all
seed x scene npz for each K.

Usage: python scripts/aggregate_sweeps.py
Outputs into run_logs/sweep_analysis/.
"""
import os, glob, re, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'logs')
OUT = os.path.join(ROOT, 'run_logs', 'sweep_analysis')
os.makedirs(OUT, exist_ok=True)
VARIANT = 'dpcc-c-tightened'


def load_metrics(npz_paths):
    """Mean metrics over a list of npz (each holds per-trial arrays)."""
    goal, jsr, vio, t = [], [], [], []
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        goal.append(np.mean(d['n_success']))
        jsr.append(np.mean(d['n_success_and_constraints']))
        vio.append(np.mean(d['n_violations']))
        t.append(np.mean(d['avg_time']))
    return (100 * np.mean(goal), 100 * np.mean(jsr),
            float(np.mean(vio)), float(np.mean(t)), len(npz_paths))


def collect(dir_glob):
    """glob -> list of variant npz."""
    out = []
    for d in glob.glob(dir_glob, recursive=True):
        f = os.path.join(d, f'{VARIANT}.npz')
        if os.path.isfile(f):
            out.append(f)
    return sorted(out)


def sweep_fm():
    rows = {}
    # K=20 = untagged (dir ends in '-hard', no _k<K> / _smoke / _diag tag)
    rows[20] = collect(f'{LOGS}/**/*Dmodels.FlowMatching/*/results/halfspace_*[!_]hard')
    # tagged k<K>
    for d in glob.glob(f'{LOGS}/**/*Dmodels.FlowMatching/*/results/halfspace_*_k*', recursive=True):
        m = re.search(r'_k(\d+)$', d)
        if m:
            rows.setdefault(int(m.group(1)), []).append(os.path.join(d, f'{VARIANT}.npz'))
    return rows


def sweep_ddim():
    rows = {}
    for d in glob.glob(f'{LOGS}/**/H8_K20_Dmodels.GaussianDiffusion/*/results/halfspace_*_ddim*', recursive=True):
        m = re.search(r'_ddim(\d+)$', d)
        if m:
            f = os.path.join(d, f'{VARIANT}.npz')
            if os.path.isfile(f):
                rows.setdefault(int(m.group(1)), []).append(f)
    return rows


def sweep_ddpm_retrain():
    rows = {}
    # native retrained K in own H8_K<K> dir, tagged retrain<K>
    for d in glob.glob(f'{LOGS}/**/H8_K*_Dmodels.GaussianDiffusion/*/results/halfspace_*_retrain*', recursive=True):
        m = re.search(r'_retrain(\d+)$', d)
        if m:
            f = os.path.join(d, f'{VARIANT}.npz')
            if os.path.isfile(f):
                rows.setdefault(int(m.group(1)), []).append(f)
    # K=20 native = untagged per-step DDPM baseline
    base = collect(f'{LOGS}/**/H8_K20_Dmodels.GaussianDiffusion/*/results/halfspace_*[!_]hard')
    if base:
        rows[20] = base
    return rows


def write_csv(name, rows):
    path = os.path.join(OUT, f'{name}.csv')
    ks = sorted(rows)
    data = {}
    t20 = None
    for k in ks:
        files = rows[k] if isinstance(rows[k], list) else rows[k]
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            continue
        g, j, v, t, n = load_metrics(files)
        data[k] = (g, j, v, t, n)
        if k == 20:
            t20 = t
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['K', 'goal_pct', 'JSR_pct', 'vbar', 't_per_step', 'speedup_vs_K20', 'n_npz'])
        for k in sorted(data):
            g, j, v, t, n = data[k]
            sp = (t20 / t) if t20 else float('nan')
            w.writerow([k, f'{g:.1f}', f'{j:.1f}', f'{v:.3f}', f'{t:.4f}', f'{sp:.2f}', n])
    print(f'[{name}] wrote {path}')
    return data


def main():
    fm = write_csv('fm_sweep', sweep_fm())
    ddim = write_csv('ddim_sweep', sweep_ddim())
    ddpm = write_csv('ddpm_retrain_sweep', sweep_ddpm_retrain())

    # 3-way comparison charts (JSR & speed vs K)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    series = [('FM (flow, no retrain)', fm, 'o-', 'C0'),
              ('DDIM (determ, no retrain)', ddim, 's--', 'C1'),
              ('native DDPM (retrained)', ddpm, '^-', 'C2')]
    for label, dat, style, c in series:
        if not dat:
            continue
        ks = sorted(dat)
        axes[0].plot(ks, [dat[k][1] for k in ks], style, color=c, label=label)
        axes[1].plot(ks, [dat[k][2] for k in ks], style, color=c, label=label)
        axes[2].plot(ks, [dat[k][3] for k in ks], style, color=c, label=label)
    axes[0].set(title='JSR (goal+constraints) vs K', xlabel='sampling steps K', ylabel='JSR %')
    axes[1].set(title='mean violations vs K', xlabel='sampling steps K', ylabel='v-bar')
    axes[2].set(title='time / plan-step vs K', xlabel='sampling steps K', ylabel='s / step')
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, 'three_way_comparison.png')
    fig.savefig(p, dpi=130)
    print(f'[chart] wrote {p}')


if __name__ == '__main__':
    main()
