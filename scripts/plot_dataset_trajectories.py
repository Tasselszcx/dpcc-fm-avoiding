#!/usr/bin/env python
"""Plot the 96 D3IL demonstrations as four data-driven route families."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

from avoiding_plot_style import (
    GOAL_Y,
    NAVY,
    draw_common_geometry,
    style_environment_axis,
)
from thesis_plot_style import COLORS, apply_paper_style, save_figure


ROUTE_NAMES = (
    "Far-left crossing",
    "Left-centre crossing",
    "Right-centre crossing",
    "Far-right crossing",
)
ROUTE_COLORS = ("#4E79A7", "#4F948E", "#C28E3D", "#8874B2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def _trim_at_goal(trajectory: np.ndarray) -> np.ndarray:
    """Stop a path at its first goal crossing and interpolate that crossing."""
    crossing = np.flatnonzero(trajectory[:, 1] >= GOAL_Y)
    if not crossing.size:
        return trajectory
    index = int(crossing[0])
    if index == 0:
        return trajectory[:1]
    before, after = trajectory[index - 1], trajectory[index]
    denominator = after[1] - before[1]
    fraction = 0.0 if abs(denominator) < 1e-12 else (GOAL_Y - before[1]) / denominator
    endpoint = before + np.clip(fraction, 0.0, 1.0) * (after - before)
    return np.vstack([trajectory[:index], endpoint])


def _load_trajectories(data_dir: Path) -> list[np.ndarray]:
    trajectories = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            env_state = pickle.load(stream)
        trajectory = np.asarray(env_state["robot"]["des_c_pos"][:, :2], dtype=float)
        trajectories.append(_trim_at_goal(trajectory))
    if len(trajectories) != 96:
        raise RuntimeError(f"Expected 96 demonstrations, found {len(trajectories)}")
    return trajectories


def _resample_by_arclength(trajectory: np.ndarray, count: int = 120) -> np.ndarray:
    segment_length = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segment_length)])
    keep = np.r_[True, np.diff(arclength) > 1e-12]
    trajectory, arclength = trajectory[keep], arclength[keep]
    if arclength[-1] <= 1e-12:
        return np.repeat(trajectory[:1], count, axis=0)
    samples = np.linspace(0.0, arclength[-1], count)
    return np.column_stack(
        [np.interp(samples, arclength, trajectory[:, dimension]) for dimension in range(2)]
    )


def _route_families(trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Split trajectories at the three largest natural gaps in goal-crossing x."""
    goal_x = np.asarray([trajectory[-1, 0] for trajectory in trajectories])
    order = np.argsort(goal_x)
    split_points = np.sort(np.argsort(np.diff(goal_x[order]))[-3:] + 1)
    labels = np.empty(len(trajectories), dtype=int)
    for label, indices in enumerate(np.split(order, split_points)):
        labels[indices] = label
    return labels, goal_x


def _representative_index(trajectories: list[np.ndarray], indices: np.ndarray) -> int:
    paths = np.stack([_resample_by_arclength(trajectories[index]) for index in indices])
    features = paths.reshape(len(indices), -1)
    pairwise = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    return int(indices[np.argmin(pairwise.mean(axis=1))])


def _legend_handles():
    return [
        Line2D(
            [0],
            [0],
            color=NAVY,
            alpha=0.35,
            linewidth=0.8,
            label="All demonstrations in family",
        ),
        Line2D([0], [0], color=NAVY, linewidth=1.8, label="Representative demonstration"),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=5.5,
            linestyle="none",
            markerfacecolor=COLORS["obstacle"],
            markeredgecolor="#984B46",
            label="Simulator obstacles",
        ),
        Line2D([0], [0], color=COLORS["goal"], linewidth=1.7, label="Goal line"),
    ]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "config/projection_eval.yaml").open() as stream:
        config = yaml.safe_load(stream)
    ax_limits = config["ax_limits"]["avoiding-d3il"]
    trajectories = _load_trajectories(
        repo_root / "d3il/environments/dataset/data/avoiding/data"
    )
    labels, goal_x = _route_families(trajectories)

    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.35))
    counts, centres = [], []
    for label, (ax, name, color) in enumerate(zip(axes.flat, ROUTE_NAMES, ROUTE_COLORS)):
        indices = np.flatnonzero(labels == label)
        representative = _representative_index(trajectories, indices)
        counts.append(len(indices))
        centres.append(float(goal_x[indices].mean()))
        style_environment_axis(ax, ax_limits)
        for index in indices:
            trajectory = trajectories[index]
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                alpha=0.24,
                linewidth=0.52,
                solid_capstyle="round",
                zorder=3,
            )
        representative_path = trajectories[representative]
        ax.plot(
            representative_path[:, 0],
            representative_path[:, 1],
            color=color,
            alpha=0.98,
            linewidth=1.75,
            solid_capstyle="round",
            zorder=6,
        )
        ax.scatter(
            goal_x[indices],
            np.full(len(indices), GOAL_Y),
            s=7,
            facecolor=COLORS["canvas"],
            edgecolor=color,
            linewidth=0.45,
            alpha=0.75,
            zorder=10,
        )
        draw_common_geometry(
            ax,
            ax_limits,
            label_start=True,
            label_goal=True,
            emphasise_terminals=True,
        )
        panel = chr(ord("a") + label)
        ax.set_title(f"({panel}) {name}", loc="left", pad=6, color=color)
        ax.text(
            0.99,
            1.025,
            f"$n={len(indices)}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color=color,
        )

    if counts != [24, 24, 24, 24]:
        raise RuntimeError(f"Unexpected route-family counts: {counts}")
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=2,
        borderpad=0.4,
        handlelength=2.2,
        columnspacing=1.35,
        labelspacing=0.55,
    )
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.105, top=0.975, wspace=0.14, hspace=0.17)
    save_figure(fig, args.output_dir, "dataset_trajs")
    plt.close(fig)
    centre_text = ", ".join(f"{value:.3f}" for value in centres)
    print(f"Saved route families {counts}; mean goal-crossing x = [{centre_text}]")


if __name__ == "__main__":
    main()
