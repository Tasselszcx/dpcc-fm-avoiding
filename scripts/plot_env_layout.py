#!/usr/bin/env python
"""Plot the three evaluation scenarios using the shared thesis visual style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from thesis_plot_style import apply_paper_style, draw_task_geometry, save_figure, task_legend_handles


def constraints_for(config, variant):
    halfspaces = config["halfspace_constraints"]["avoiding-d3il"]
    obstacles = config["obstacle_constraints"]["avoiding-d3il"]
    mapping = {
        "top-right-hard": ([halfspaces[1]], obstacles[4]),
        "top-left-hard": ([halfspaces[0]], obstacles[3]),
        "both-hard": ([halfspaces[2], halfspaces[3]], obstacles[5]),
    }
    return mapping[variant]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open("config/projection_eval.yaml") as stream:
        config = yaml.safe_load(stream)

    variants = ("top-right-hard", "top-left-hard", "both-hard")
    titles = ("Top-right", "Top-left", "Corridor")
    ax_limits = config["ax_limits"]["avoiding-d3il"]
    clearance = config["enlarge_constraints"]["avoiding"]

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), sharex=True, sharey=True)
    for index, (ax, variant, title) in enumerate(zip(axes, variants, titles)):
        halfspaces, circle = constraints_for(config, variant)
        draw_task_geometry(
            ax,
            ax_limits=ax_limits,
            halfspaces=halfspaces,
            analytic_circle=circle,
            clearance=clearance,
            show_tightening=True,
            show_axis_labels=False,
        )
        ax.set_title(f"({chr(97 + index)}) {title}", pad=5)
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    handles = task_legend_handles(include_tightening=True)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=5,
        borderpad=0.3,
        handlelength=1.65,
        columnspacing=0.85,
    )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.14, top=0.79, wspace=0.08)
    save_figure(fig, args.output_dir, "env_layout_3scenes")
    plt.close(fig)
    print(f"Saved scenario layout to {args.output_dir}")
