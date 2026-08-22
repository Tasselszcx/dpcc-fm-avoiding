#!/usr/bin/env python
"""Generate thesis-ready experiment figures from the authoritative NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from thesis_plot_style import (
    COLORS,
    apply_paper_style,
    method_legend_handles,
    panel_label,
    save_figure,
    style_axis,
)


KS = np.asarray([1, 2, 4, 8, 16, 32])
SCENES = ("top-right-hard", "top-left-hard", "both-hard")
SEEDS = (0, 1, 2)


def _main_path(base: Path, horizon: int, method: str, seed: int, scene: str, k: int) -> Path:
    if method == "fm":
        train_dir = f"H{horizon}_K20_Dmodels.FlowMatching"
        result_dir = f"halfspace_{scene}_d3il_h{horizon}_fm_k{k}"
    elif method == "ddpm":
        train_dir = f"H{horizon}_K{k}_Dmodels.GaussianDiffusion"
        result_dir = f"halfspace_{scene}_d3il_h{horizon}_ddpm_k{k}"
    else:
        raise ValueError(method)
    return base / train_dir / str(seed) / "results" / result_dir / "dpcc-c-tightened.npz"


def _aggregate_seed(path_fn):
    jsr_by_seed, time_by_seed = [], []
    for seed in SEEDS:
        jsr, timing = [], []
        for scene in SCENES:
            path = path_fn(seed, scene)
            if not path.exists():
                raise FileNotFoundError(path)
            data = np.load(path, allow_pickle=True)
            jsr.append(np.asarray(data["n_success_and_constraints"], dtype=float))
            timing.append(np.asarray(data["avg_time"], dtype=float) * 1000.0)
        jsr_by_seed.append(np.concatenate(jsr).mean() * 100.0)
        time_by_seed.append(np.concatenate(timing).mean())
    return np.asarray(jsr_by_seed), np.asarray(time_by_seed)


def main_sweep(base: Path):
    result = {}
    for horizon in (8, 16):
        for method in ("fm", "ddpm"):
            jsr_seed, time_seed = [], []
            for k in KS:
                jsr, timing = _aggregate_seed(
                    lambda seed, scene, h=horizon, m=method, kk=int(k): _main_path(
                        base, h, m, seed, scene, kk
                    )
                )
                jsr_seed.append(jsr)
                time_seed.append(timing)
            result[(horizon, method)] = {
                "jsr_seed": np.stack(jsr_seed, axis=1),
                "time_seed": np.stack(time_seed, axis=1),
            }
    return result


def plot_main_results(base: Path, output_dir: Path) -> None:
    values = main_sweep(base)
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.2), sharex="col")
    x = np.arange(len(KS))
    method_style = {
        "fm": dict(color=COLORS["fm"], fill=COLORS["fm_fill"], marker="o", linestyle="-"),
        "ddpm": dict(color=COLORS["ddpm"], fill=COLORS["ddpm_fill"], marker="s", linestyle="--"),
    }

    for col, horizon in enumerate((8, 16)):
        ax_jsr = axes[0, col]
        ax_time = axes[1, col]
        for method in ("fm", "ddpm"):
            style = method_style[method]
            jsr_seed = values[(horizon, method)]["jsr_seed"]
            time_seed = values[(horizon, method)]["time_seed"]
            jsr_mean, jsr_sd = jsr_seed.mean(0), jsr_seed.std(0, ddof=1)
            time_mean, time_sd = time_seed.mean(0), time_seed.std(0, ddof=1)

            ax_jsr.fill_between(
                x,
                np.clip(jsr_mean - jsr_sd, 0, 100),
                np.clip(jsr_mean + jsr_sd, 0, 100),
                color=style["fill"],
                alpha=0.48,
                linewidth=0,
            )
            ax_jsr.plot(
                x,
                jsr_mean,
                color=style["color"],
                marker=style["marker"],
                markerfacecolor=COLORS["canvas"],
                markeredgecolor=style["color"],
                markeredgewidth=1.0,
                linestyle=style["linestyle"],
                zorder=3,
            )
            ax_time.fill_between(
                x,
                np.maximum(time_mean - time_sd, 0.5),
                time_mean + time_sd,
                color=style["fill"],
                alpha=0.38,
                linewidth=0,
            )
            ax_time.plot(
                x,
                time_mean,
                color=style["color"],
                marker=style["marker"],
                markerfacecolor=COLORS["canvas"],
                markeredgecolor=style["color"],
                markeredgewidth=1.0,
                linestyle=style["linestyle"],
                zorder=3,
            )

        for method, selected_k in (("fm", 2), ("ddpm", 4)):
            idx = int(np.where(KS == selected_k)[0][0])
            jsr_y = values[(horizon, method)]["jsr_seed"].mean(0)[idx]
            time_y = values[(horizon, method)]["time_seed"].mean(0)[idx]
            ax_jsr.scatter(
                [idx],
                [jsr_y],
                s=72,
                facecolors="none",
                edgecolors=COLORS["accent"],
                linewidths=1.15,
                zorder=5,
            )
            ax_time.scatter(
                [idx],
                [time_y],
                s=72,
                facecolors="none",
                edgecolors=COLORS["accent"],
                linewidths=1.15,
                zorder=5,
            )

        ax_jsr.set_title(f"Planning horizon $H={horizon}$", pad=5)
        ax_jsr.set_ylim(0, 103)
        ax_jsr.set_yticks([0, 25, 50, 75, 100])
        style_axis(ax_jsr)
        ax_time.set_yscale("log")
        style_axis(ax_time)
        ax_time.set_xticks(x, [str(k) for k in KS])
        ax_time.set_xlabel("Generative evaluations $K$")

    axes[0, 0].set_ylabel("Joint success rate (\%)")
    axes[1, 0].set_ylabel("Parallel-sweep planning time (ms)")
    panel_label(axes[0, 0], "(a)")
    panel_label(axes[0, 1], "(b)")
    panel_label(axes[1, 0], "(c)")
    panel_label(axes[1, 1], "(d)")
    fig.legend(
        handles=method_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        borderpad=0.35,
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.text(
        0.985,
        0.505,
        "Shading: $\pm$1 sample SD across three training seeds",
        rotation=90,
        va="center",
        ha="right",
        fontsize=6.7,
        color=COLORS["secondary_text"],
    )
    fig.subplots_adjust(left=0.105, right=0.95, bottom=0.105, top=0.9, wspace=0.25, hspace=0.16)
    save_figure(fig, output_dir, "main_results_k_sweep")
    plt.close(fig)


def _projection_path(base: Path, seed: int, scene: str, threshold: float) -> Path:
    root = base / "H16_K16_Dmodels.GaussianDiffusion" / str(seed) / "results"
    result_dir = root / f"halfspace_{scene}_d3il_h16_ddpm_k16"
    if threshold == 0.5:
        name = "dpcc-c-tightened.npz"
    else:
        token = {0.25: "0p25", 0.125: "0p125", 0.0625: "0p0625"}[threshold]
        name = f"dpcc-c-tightened-th{token}.npz"
    return result_dir / name


def plot_projection_window(base: Path, output_dir: Path) -> None:
    thresholds = np.asarray([0.0625, 0.125, 0.25, 0.5])
    calls = np.asarray([2, 3, 5, 9])
    jsr_seed, time_seed = [], []
    for threshold in thresholds:
        jsr, timing = _aggregate_seed(
            lambda seed, scene, th=float(threshold): _projection_path(base, seed, scene, th)
        )
        jsr_seed.append(jsr)
        time_seed.append(timing)
    jsr_seed = np.stack(jsr_seed, axis=1)
    time_seed = np.stack(time_seed, axis=1)

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.65))
    x = np.arange(len(calls))
    labels = [f"{c}\n$q={q:g}$" for c, q in zip(calls, thresholds)]

    for ax, seed_values, ylabel, ylim in (
        (axes[0], jsr_seed, "Joint success rate (\%)", (35, 101)),
        (axes[1], time_seed, "Parallel-sweep planning time (ms)", None),
    ):
        ax.axvspan(2.72, 3.28, color=COLORS["accent_fill"], alpha=0.25, linewidth=0, zorder=0)
        for seed, row in zip(SEEDS, seed_values):
            ax.plot(
                x,
                row,
                color=COLORS["light_axis"],
                marker="o",
                markersize=3.4,
                markerfacecolor=COLORS["canvas"],
                markeredgewidth=0.75,
                linewidth=0.75,
                alpha=0.8,
                zorder=1,
            )
        mean = seed_values.mean(0)
        sd = seed_values.std(0, ddof=1)
        ax.fill_between(x, mean - sd, mean + sd, color=COLORS["ddpm_fill"], alpha=0.65, linewidth=0)
        ax.plot(
            x,
            mean,
            color=COLORS["ddpm"],
            marker="s",
            markerfacecolor=COLORS["canvas"],
            markeredgewidth=1.0,
            linewidth=1.65,
            zorder=3,
        )
        ax.set_xticks(x, labels)
        ax.set_xlabel("Projector calls per planning decision")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        style_axis(ax)

    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    axes[0].text(
        3,
        97.5,
        "default window",
        ha="center",
        va="top",
        fontsize=6.8,
        color=COLORS["accent"],
    )
    handles = [
        Line2D([0], [0], color=COLORS["ddpm"], marker="s", markerfacecolor="white", label="Mean"),
        Line2D([0], [0], color=COLORS["light_axis"], marker="o", markerfacecolor="white", label="Training seed"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        borderpad=0.35,
        handlelength=2.2,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.23, top=0.82, wspace=0.27)
    save_figure(fig, output_dir, "projection_window_ablation")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plans-dir",
        type=Path,
        default=Path("logs/avoiding-d3il/plans"),
        help="Root containing trained-plan result directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_main_results(args.plans_dir, args.output_dir)
    plot_projection_window(args.plans_dir, args.output_dir)
    print(f"Saved thesis figures to {args.output_dir.resolve()}")
