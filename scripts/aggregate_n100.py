#!/usr/bin/env python
"""
Aggregate the unified n_trials=100 three-way sweep into two tables + a chart.

All three families were re-evaluated at the SAME 100 test scenarios so every
cell is a paired comparison (see scripts/run_eval_n100.sh). Tags:
  FM          : Dmodels.FlowMatching/<seed>/results/halfspace_<scene>_fmk<K>n100/
  DDIM        : H8_K20_Dmodels.GaussianDiffusion/<seed>/results/
                halfspace_<scene>_ddim<K>n100/
  native DDPM : H8_K<K>_Dmodels.GaussianDiffusion/<seed>/results/
                halfspace_<scene>_retrain<K>n100/

Each npz stores per-trial arrays (len=100): n_success, n_success_and_constraints
(=JSR), n_violations, avg_time. We average across all seed x scene npz per K.

Table 1 = FM at K in {1,2,3,4,5,6,8,10,15,20}
Table 2 = DDIM and native DDPM at the same K.

Usage: python scripts/aggregate_n100.py
Outputs into run_logs/sweep_analysis_n100/.
"""
import os, glob, re, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'logs')
OUT = os.path.join(ROOT, 'run_logs', 'sweep_analysis_n100')
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


def sweep(model_glob, tag_re):
    """Generic: glob dirs under model_glob, key by K from tag_re, drop _smoke."""
    rows = {}
    for d in glob.glob(model_glob, recursive=True):
        if '_smoke' in d:
            continue
        m = re.search(tag_re, os.path.basename(d))
        if not m:
            continue
        f = os.path.join(d, f'{VARIANT}.npz')
        if os.path.isfile(f):
            rows.setdefault(int(m.group(1)), []).append(f)
    return rows


def sweep_fm():
    return sweep(
        f'{LOGS}/**/*Dmodels.FlowMatching/*/results/halfspace_*_fmk*n100',
        r'_fmk(\d+)n100$')


def sweep_ddim():
    return sweep(
        f'{LOGS}/**/H8_K20_Dmodels.GaussianDiffusion/*/results/halfspace_*_ddim*n100',
        r'_ddim(\d+)n100$')


def sweep_ddpm_retrain():
    return sweep(
        f'{LOGS}/**/H8_K*_Dmodels.GaussianDiffusion/*/results/halfspace_*_retrain*n100',
        r'_retrain(\d+)n100$')


def _avg_t(paths):
    ts = [np.mean(np.load(p, allow_pickle=True)['avg_time']) for p in paths]
    return float(np.mean(ts)) if ts else None


def serial_tstep(model_glob, tag_re):
    """Single-process re-measured t/step keyed by K (tag suffix 'speed')."""
    rows = {}
    for d in glob.glob(model_glob, recursive=True):
        if '_smoke' in d:
            continue
        m = re.search(tag_re, os.path.basename(d))
        if not m:
            continue
        f = os.path.join(d, f'{VARIANT}.npz')
        if os.path.isfile(f):
            rows.setdefault(int(m.group(1)), []).append(f)
    return {k: _avg_t(v) for k, v in rows.items()}


SERIAL = {
    'fm_n100': (f'{LOGS}/**/*Dmodels.FlowMatching/*/results/halfspace_*_fmk*speed',
                r'_fmk(\d+)speed$'),
    'ddim_n100': (f'{LOGS}/**/H8_K20_Dmodels.GaussianDiffusion/*/results/halfspace_*_ddim*speed',
                  r'_ddim(\d+)speed$'),
    'ddpm_retrain_n100': (f'{LOGS}/**/H8_K*_Dmodels.GaussianDiffusion/*/results/halfspace_*_retrain*speed',
                          r'_retrain(\d+)speed$'),
}


def compute(rows):
    """K -> (goal, jsr, vbar, t, n)."""
    data = {}
    for k in sorted(rows):
        files = [f for f in rows[k] if os.path.isfile(f)]
        if files:
            data[k] = load_metrics(files)
    return data


def write_csv(name, data):
    path = os.path.join(OUT, f'{name}.csv')
    t20 = data.get(20, (None,) * 4)[3]
    # single-process re-measured t/step (authoritative for speed claims)
    ser = {}
    if name in SERIAL:
        ser = serial_tstep(*SERIAL[name])
    s20 = ser.get(20)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['K', 'goal_pct', 'JSR_pct', 'vbar',
                    't_per_step_conc', 'speedup_conc',
                    't_per_step_serial', 'speedup_serial', 'n_npz'])
        for k in sorted(data):
            g, j, v, t, n = data[k]
            sp = (t20 / t) if t20 else float('nan')
            ts = ser.get(k)
            sps = (s20 / ts) if (s20 and ts) else float('nan')
            ts_s = f'{ts:.4f}' if ts else 'NA'
            sps_s = f'{sps:.2f}' if ts else 'NA'
            w.writerow([k, f'{g:.1f}', f'{j:.1f}', f'{v:.3f}',
                        f'{t:.4f}', f'{sp:.2f}', ts_s, sps_s, n])
    print(f'[{name}] wrote {path}')


def print_table(title, data):
    print(f'\n=== {title} (n_trials=100) ===')
    print(f'{"K":>4} {"goal%":>7} {"JSR%":>7} {"v_bar":>7} '
          f'{"t/step":>8} {"speedup":>8} {"n":>3}')
    t20 = data.get(20, (None,) * 4)[3]
    for k in sorted(data):
        g, j, v, t, n = data[k]
        sp = (t20 / t) if t20 else float('nan')
        print(f'{k:>4} {g:>7.1f} {j:>7.1f} {v:>7.3f} {t:>8.4f} '
              f'{sp:>7.2f}x {n:>3}')


def main():
    fm = compute(sweep_fm())
    ddim = compute(sweep_ddim())
    ddpm = compute(sweep_ddpm_retrain())

    write_csv('fm_n100', fm)
    write_csv('ddim_n100', ddim)
    write_csv('ddpm_retrain_n100', ddpm)

    print_table('Table 1: FM K-sweep', fm)
    print_table('Table 2a: DDIM (subsample K=20, no retrain)', ddim)
    print_table('Table 2b: native DDPM (retrained per K)', ddpm)

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
    axes[0].set(title='JSR (goal+constraints) vs K (n=100)',
                xlabel='sampling steps K', ylabel='JSR %')
    axes[1].set(title='mean violations vs K (n=100)',
                xlabel='sampling steps K', ylabel='v-bar')
    axes[2].set(title='time / plan-step vs K (n=100)',
                xlabel='sampling steps K', ylabel='s / step')
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, 'three_way_comparison_n100.png')
    fig.savefig(p, dpi=130)
    print(f'\n[chart] wrote {p}')


if __name__ == '__main__':
    main()
