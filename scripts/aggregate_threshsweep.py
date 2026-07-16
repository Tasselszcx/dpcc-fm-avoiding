#!/usr/bin/env python
"""
Aggregate the FM projection-frequency (threshold) sweep into a CSV + CZ-3 figure.

Fixed FM backbone, swept over K in {20,15,10,5} x threshold in {1.0,0.7,0.5,0.3,0.2,0.1}.
The projector fires only when the FM integration time t_current >= 1 - threshold, so
`threshold` is the fraction of the (late) integration window in which projection is
active. Each (K, threshold) cell is averaged over 3 seeds x 3 scenes x 100 trials.

Number of projections for a given (K, threshold): count of Euler steps
    step in {0..K-1}  with  step/K >= 1 - threshold
(exactly how flow_matching.py:267 gates the projector).

Outputs into run_logs/threshsweep_analysis/:
  - fm_threshsweep.csv           (K, threshold, n_proj, goal%, JSR%, vbar, t/step, n_npz)
  - fig_proj_frequency.pdf/.png  (left: JSR vs threshold; right: t/step vs threshold)

Usage: python scripts/aggregate_threshsweep.py
"""
import os, glob, re, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'logs')
OUT = os.path.join(ROOT, 'run_logs', 'threshsweep_analysis')
os.makedirs(OUT, exist_ok=True)

KS = [20, 15, 10, 5]
THS = [100, 70, 50, 30, 20, 10]   # percent -> threshold = TH/100


def n_projections(K, th_frac):
    """#Euler steps that fall inside the projection window (FM gating)."""
    return sum(1 for step in range(K) if step / K >= (1.0 - th_frac))


def collect(K, TH):
    """Mean metrics over all seed x scene npz for one (K, TH) cell."""
    # only seed-numbered dirs (skip any 'all_seeds' aggregate artefacts)
    pat = (f'{LOGS}/**/H8_K20_Dmodels.FlowMatching/*/results/'
           f'halfspace_*_thsweep_k{K}/dpcc-c-tightened-lateproj{TH}.npz')
    goal, jsr, vio, t, npz = [], [], [], [], 0
    for p in glob.glob(pat, recursive=True):
        # guard: the seed segment must be an integer dir
        seg = p.split('/H8_K20_Dmodels.FlowMatching/')[1].split('/')[0]
        if not seg.isdigit():
            continue
        d = np.load(p, allow_pickle=True)
        goal.append(np.mean(d['n_success']))
        jsr.append(np.mean(d['n_success_and_constraints']))
        vio.append(np.mean(d['n_violations']))
        t.append(np.mean(d['avg_time']))
        npz += 1
    if npz == 0:
        return None
    return (100 * np.mean(goal), 100 * np.mean(jsr),
            float(np.mean(vio)), float(np.mean(t)), npz)


def main():
    rows = []
    data = {}   # (K, TH) -> metrics
    for K in KS:
        for TH in THS:
            m = collect(K, TH)
            if m is None:
                print(f'[warn] no npz for K={K} th={TH}')
                continue
            g, j, v, t, n = m
            npj = n_projections(K, TH / 100.0)
            data[(K, TH)] = (g, j, v, t, n, npj)
            rows.append([K, f'{TH/100:.2f}', npj, f'{g:.1f}', f'{j:.1f}',
                         f'{v:.3f}', f'{t:.4f}', n])

    csv_path = os.path.join(OUT, 'fm_threshsweep.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['K', 'threshold', 'n_proj', 'goal_pct', 'JSR_pct',
                    'vbar', 't_per_step', 'n_npz'])
        w.writerows(rows)
    print(f'[csv] wrote {csv_path}  ({len(rows)} cells)')

    # console table
    print(f'\n=== FM threshold sweep (n=100/cell) ===')
    print(f'{"K":>3} {"thr":>5} {"nproj":>5} {"goal%":>6} {"JSR%":>6} '
          f'{"vbar":>6} {"t/step":>7} {"n":>3}')
    for K in KS:
        for TH in THS:
            if (K, TH) not in data:
                continue
            g, j, v, t, n, npj = data[(K, TH)]
            print(f'{K:>3} {TH/100:>5.2f} {npj:>5} {g:>6.1f} {j:>6.1f} '
                  f'{v:>6.3f} {t:>7.4f} {n:>3}')

    # figure: JSR vs threshold, t/step vs threshold, one line per K
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {20: 'C0', 15: 'C1', 10: 'C2', 5: 'C3'}
    markers = {20: 'o', 15: 's', 10: '^', 5: 'D'}
    for K in KS:
        ths = [TH / 100 for TH in THS if (K, TH) in data]
        js = [data[(K, TH)][1] for TH in THS if (K, TH) in data]
        ts = [data[(K, TH)][3] for TH in THS if (K, TH) in data]
        ax[0].plot(ths, js, marker=markers[K], color=colors[K],
                   label=f'FM $K$={K}')
        ax[1].plot(ths, ts, marker=markers[K], color=colors[K],
                   label=f'FM $K$={K}')
    ax[0].axhline(71.3, ls='--', color='gray', lw=1,
                  label='DDPM($K$=20) 71.3')
    ax[0].set(xlabel='projection window (threshold fraction)',
              ylabel='JSR (goal+cons \\%)',
              title='Safety vs projection frequency')
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set(xlabel='projection window (threshold fraction)',
              ylabel='per-step cost (s)',
              title='Cost vs projection frequency')
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        p = os.path.join(OUT, f'fig_proj_frequency.{ext}')
        fig.savefig(p, dpi=140)
        print(f'[fig] wrote {p}')


if __name__ == '__main__':
    main()
