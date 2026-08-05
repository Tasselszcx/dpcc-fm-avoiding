#!/usr/bin/env python
"""Recompute the violation metric as mean(n_violations / n_steps)*100 per cell,
for every table in Chapter 3. Prints goal%/JSR%/viol%/t(ms) so tables can be updated."""
import glob, numpy as np

BASE = "logs/avoiding-d3il/plans"


def cell(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    ns, nsc, nv, nstep, at = [], [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        ns.append(d["n_success"]); nsc.append(d["n_success_and_constraints"])
        nv.append(d["n_violations"]); nstep.append(d["n_steps"]); at.append(d["avg_time"])
    ns = np.concatenate(ns); nsc = np.concatenate(nsc)
    nv = np.concatenate(nv); nstep = np.concatenate(nstep); at = np.concatenate(at)
    frac = np.where(nstep > 0, nv / nstep, 0.0)
    return dict(nf=len(files), nt=len(ns), goal=ns.mean()*100, jsr=nsc.mean()*100,
                violpct=frac.mean()*100, t=at.mean()*1000)


def show(name, patt, ks):
    print("\n=== %s ===" % name)
    print("%4s %5s %6s %7s %7s %8s %9s" % ("K", "nf", "nt", "goal%", "JSR%", "viol%", "t(ms)"))
    for K in ks:
        c = cell(patt.format(K=K))
        if c is None:
            print("%4s  (missing)" % K); continue
        print("%4s %5d %6d %7.1f %7.1f %8.3g %9.1f" % (K, c["nf"], c["nt"], c["goal"], c["jsr"], c["violpct"], c["t"]))


KS = [1, 2, 4, 8, 16, 32]
show("FM  H16", BASE + "/H16_K20_Dmodels.FlowMatching/*/results/halfspace_*_d3il_h16_fm_k{K}/dpcc-c-tightened.npz", KS)
show("DDPM H16", BASE + "/H16_K{K}_Dmodels.GaussianDiffusion/*/results/halfspace_*_d3il_h16_ddpm_k{K}/dpcc-c-tightened.npz", KS)
show("FM  H8", BASE + "/H8_K20_Dmodels.FlowMatching/*/results/halfspace_*_d3il_h8_fm_k{K}/dpcc-c-tightened.npz", KS)
show("DDPM H8", BASE + "/H8_K{K}_Dmodels.GaussianDiffusion/*/results/halfspace_*_d3il_h8_ddpm_k{K}/dpcc-c-tightened.npz", KS)
show("DDIM H16", BASE + "/H16_K32_Dmodels.GaussianDiffusion/*/results/halfspace_*_d3il_h16_ddim_k{K}/dpcc-c-tightened.npz", KS)
