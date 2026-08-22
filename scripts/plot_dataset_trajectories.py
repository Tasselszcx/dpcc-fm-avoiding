#!/usr/bin/env python
"""Plot the 96 D3IL demonstrations using the shared thesis visual style."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from thesis_plot_style import COLORS, apply_paper_style, draw_task_geometry, save_figure, task_legend_handles


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open("config/projection_eval.yaml") as stream:
        config = yaml.safe_load(stream)

    exp = "avoiding-d3il"
    ax_limits = config["ax_limits"][exp]
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "d3il/environments/dataset/data/avoiding/data"
    trajectories = []
    for filename in sorted(os.listdir(data_dir)):
        with open(os.path.join(data_dir, filename), "rb") as stream:
            env_state = pickle.load(stream)
        trajectories.append(env_state["robot"]["des_c_pos"][:, :2])

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.45, 5.0))
    draw_task_geometry(ax, ax_limits=ax_limits, show_axis_labels=True)
    for trajectory in trajectories:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=COLORS["ddpm"],
            alpha=0.22,
            linewidth=0.55,
            zorder=2,
        )
    ax.scatter(
        [trajectories[0][0, 0]],
        [trajectories[0][0, 1]],
        s=20,
        facecolor=COLORS["canvas"],
        edgecolor=COLORS["text"],
        linewidth=0.75,
        zorder=7,
    )
    ax.text(
        0.515,
        -0.265,
        "start",
        fontsize=7,
        ha="left",
        va="bottom",
        color=COLORS["secondary_text"],
    )
    handles = task_legend_handles(include_demo=True)
    # The dataset panel contains no analytic evaluation constraints.
    handles = [handles[0], handles[1], handles[4]]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        borderpad=0.3,
        handlelength=1.5,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.1, top=0.88)
    save_figure(fig, args.output_dir, "dataset_trajs")
    plt.close(fig)
    print(f"Saved {len(trajectories)} demonstrations to {args.output_dir}")
