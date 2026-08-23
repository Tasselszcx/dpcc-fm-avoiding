#!/usr/bin/env python
"""Restyle the retained FM K=2 rollout traces for the three test scenes.

The evaluator keeps a compact plot of up to ten executed rollouts per scene,
but does not store the underlying trajectory arrays. This script extracts only
the blue rollout layer from those authoritative evaluator PNGs and overlays it
on the shared thesis geometry. It changes the visual presentation without
inventing or interpolating new trajectories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import yaml

from avoiding_plot_style import SCENES, draw_scene, scene_legend_handles, style_environment_axis
from thesis_plot_style import COLORS, apply_paper_style, save_figure


SOURCE_AXES_PIXELS = (53, 15, 751, 785)  # left, top, right, bottom
SOURCE_IMAGE_SIZE = (771, 819)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("logs/avoiding-d3il/plans/H8_K20_Dmodels.FlowMatching/all_seeds"),
        help="Directory containing the evaluator's aggregate FM K=2 scene plots.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def _source_path(source_root: Path, scene: str) -> Path:
    return source_root / f"{scene}_d3il_h8_fm_k2" / "dpcc-c-tightened.png"


def _rollout_layer(path: Path) -> np.ndarray:
    """Return a transparent FM-coloured raster containing only rollout pixels."""
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    if image.size != SOURCE_IMAGE_SIZE:
        raise RuntimeError(
            f"Unexpected evaluator image size {image.size} for {path}; "
            f"expected {SOURCE_IMAGE_SIZE}. Recalibrate SOURCE_AXES_PIXELS."
        )
    rgb = np.asarray(image, dtype=float)
    left, top, right, bottom = SOURCE_AXES_PIXELS
    crop = rgb[top : bottom + 1, left : right + 1]

    # Evaluator rollouts are saturated blue. Constraints are pale blue or
    # lavender, so the additional red/green cutoff isolates the path layer.
    red, green, blue = np.moveaxis(crop, -1, 0)
    strength = np.clip((blue - np.maximum(red, green)) / 155.0, 0.0, 1.0)
    strength[(blue < 180.0) | (red > 145.0) | (green > 145.0)] = 0.0
    if np.count_nonzero(strength) < 1000:
        raise RuntimeError(f"Failed to isolate a credible rollout layer from {path}")

    fm_rgb = np.asarray(matplotlib.colors.to_rgb(COLORS["fm"]))
    layer = np.empty((*strength.shape, 4), dtype=float)
    layer[..., :3] = fm_rgb
    layer[..., 3] = 0.78 * strength
    return layer


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_root = args.source_root
    if not source_root.is_absolute():
        source_root = repo_root / source_root
    with (repo_root / "config/projection_eval.yaml").open() as stream:
        config = yaml.safe_load(stream)
    ax_limits = config["ax_limits"]["avoiding-d3il"]

    apply_paper_style()
    matplotlib.rcParams["hatch.color"] = "#C95850"
    matplotlib.rcParams["hatch.linewidth"] = 0.55
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.25))
    for index, (ax, (scene, title, halfspace_indices, circle_index)) in enumerate(
        zip(axes, SCENES)
    ):
        style_environment_axis(ax, ax_limits)
        draw_scene(ax, config, halfspace_indices, circle_index, ax_limits)
        layer = _rollout_layer(_source_path(source_root, scene))
        ax.imshow(
            layer,
            extent=(*ax_limits[0], *ax_limits[1]),
            origin="upper",
            interpolation="bilinear",
            aspect="auto",
            zorder=6,
        )
        ax.set_aspect("equal", adjustable="box")
        panel = chr(ord("a") + index)
        ax.set_title(f"({panel}) {title}", loc="left", pad=5, color="#163A70")
        ax.text(
            0.99,
            1.025,
            scene,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=COLORS["secondary_text"],
        )

    rollout_handle = Line2D(
        [0],
        [0],
        color=COLORS["fm"],
        linewidth=1.6,
        label=r"Executed FM rollouts ($H=8$, $K=2$)",
    )
    geometry_handles = scene_legend_handles()
    handles = [
        rollout_handle,
        geometry_handles[0],
        geometry_handles[1],
        geometry_handles[2],
        geometry_handles[5],
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        borderpad=0.4,
        handlelength=2.0,
        columnspacing=1.1,
        labelspacing=0.45,
    )
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.205, top=0.94, wspace=0.09)
    save_figure(fig, args.output_dir, "traj_fm_k2_3scenes")
    plt.close(fig)
    print(f"Restyled retained FM K=2 rollout traces from {source_root}")


if __name__ == "__main__":
    main()
