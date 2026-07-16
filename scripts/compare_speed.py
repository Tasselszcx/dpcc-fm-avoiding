#!/usr/bin/env python
"""
Compare concurrent (8-way) vs single-process t/step, reading avg_time straight
from npz. Concurrent tags = <m><K>n100 (n_trials=100, ran 8 concurrent).
Serial  tags = <m><K>speed (n_trials=20, ran strictly serial, threads pinned).

Prints one table per method: K, t_concurrent, t_serial, ratio(serial/conc).
"""
import os, glob, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'logs')
VARIANT = 'dpcc-c-tightened'


def avg_t(paths):
    ts = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        ts.append(np.mean(d['avg_time']))
    return float(np.mean(ts)) if ts else float('nan')


def sweep(model_glob, tag_re):
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
    return {k: avg_t(v) for k, v in rows.items()}


METHODS = {
    'FM': (
        f'{LOGS}/**/*Dmodels.FlowMatching/*/results/halfspace_*_fmk*',
        r'_fmk(\d+)n100$', r'_fmk(\d+)speed$'),
    'DDIM': (
        f'{LOGS}/**/H8_K20_Dmodels.GaussianDiffusion/*/results/halfspace_*_ddim*',
        r'_ddim(\d+)n100$', r'_ddim(\d+)speed$'),
    'DDPM': (
        f'{LOGS}/**/H8_K*_Dmodels.GaussianDiffusion/*/results/halfspace_*_retrain*',
        r'_retrain(\d+)n100$', r'_retrain(\d+)speed$'),
}

for name, (g, conc_re, ser_re) in METHODS.items():
    conc = sweep(g, conc_re)
    ser = sweep(g, ser_re)
    print(f'\n=== {name}: concurrent(n=100,8-way) vs serial(n=20,1-proc) ===')
    print(f'{"K":>4} {"t_conc":>9} {"t_serial":>9} {"ser/conc":>9}')
    for k in sorted(set(conc) | set(ser)):
        c = conc.get(k, float("nan"))
        s = ser.get(k, float("nan"))
        r = (s / c) if c and not np.isnan(c) and not np.isnan(s) else float('nan')
        print(f'{k:>4} {c:>9.4f} {s:>9.4f} {r:>8.2f}x')
