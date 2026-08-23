#!/usr/bin/env python
"""Plot the three evaluation-only constraint scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from avoiding_plot_style import (
    CORAL,
    SCENES,
    draw_scene,
    scene_legend_handles,
    style_environment_axis,
)
from thesis_plot_style import apply_paper_style, save_figure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "config/projection_eval.yaml").open() as stream:
        config = yaml.safe_load(stream)
    ax_limits = config["ax_limits"]["avoiding-d3il"]

    apply_paper_style()
    matplotlib.rcParams["hatch.color"] = CORAL
    matplotlib.rcParams["hatch.linewidth"] = 0.55
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.15))
    for index, (ax, (name, title, halfspace_indices, circle_index)) in enumerate(
        zip(axes, SCENES)
    ):
        style_environment_axis(ax, ax_limits)
        draw_scene(ax, config, halfspace_indices, circle_index, ax_limits)
        panel = chr(ord("a") + index)
        ax.set_title(f"({panel}) {title}", loc="left", pad=5, color="#163A70")
        ax.text(
            0.99,
            1.025,
            name,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="#555B66",
        )
    fig.legend(
        handles=scene_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        borderpad=0.4,
        handlelength=2.0,
        columnspacing=1.15,
        labelspacing=0.5,
    )
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.23, top=0.94, wspace=0.09)
    save_figure(fig, args.output_dir, "env_layout_3scenes")
    plt.close(fig)
    print(f"Saved {len(SCENES)} evaluation-only constraint scenes to {args.output_dir}")


if __name__ == "__main__":
    main()
