#!/usr/bin/env python
"""聚合 H8 对称评测结果(FM + DDPM × K∈{1,2,4,8,16,32})。
每格 = 3 seed × 3 scene = 9 npz × 100 trial = 900 trial 宏平均。
指标: goal%=mean(n_success)*100, JSR%=mean(n_success_and_constraints)*100,
       v̄=mean(total_violations), t/step(ms)=mean(avg_time)*1000。"""
import glob, numpy as np

BASE = "logs/avoiding-d3il/plans"
KS = [1, 2, 4, 8, 16, 32]


def cell(pattern):
    files = sorted(glob.glob(pattern))
    ns, nsc, tv, at = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        ns.append(d["n_success"]); nsc.append(d["n_success_and_constraints"])
        tv.append(d["total_violations"]); at.append(d["avg_time"])
    ns = np.concatenate(ns); nsc = np.concatenate(nsc)
    tv = np.concatenate(tv); at = np.concatenate(at)
    return len(files), len(ns), ns.mean()*100, nsc.mean()*100, tv.mean(), at.mean()*1000


def table(name, patt):
    print(f"\n=== {name} ===")
    print(f"{'K':>3} {'nfile':>5} {'ntrial':>6} {'goal%':>7} {'JSR%':>7} {'v̄':>10} {'t/step(ms)':>11}")
    for K in KS:
        nf, nt, g, j, v, t = cell(patt.format(K=K))
        print(f"{K:>3} {nf:>5} {nt:>6} {g:>7.1f} {j:>7.1f} {v:>10.3g} {t:>11.1f}")


table("FM  H8", BASE + "/H8_K20_Dmodels.FlowMatching/*/results/halfspace_*_d3il_h8_fm_k{K}/dpcc-c-tightened.npz")
table("DDPM H8", BASE + "/H8_K{K}_Dmodels.GaussianDiffusion/*/results/halfspace_*_d3il_h8_ddpm_k{K}/dpcc-c-tightened.npz")
