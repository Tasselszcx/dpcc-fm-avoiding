"""Shared visual language for thesis figures.

The palette and typography follow the local figure-style reference: white canvas,
deep-grey structure, low-saturation semantic colours, redundant line encodings,
and light statistical uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np


COLORS = {
    "text": "#202124",
    "secondary_text": "#555B66",
    "axis": "#727984",
    "light_axis": "#B8BEC7",
    "grid": "#E4E7EB",
    "canvas": "#FFFFFF",
    "panel": "#F7F8FA",
    "fm": "#5C9B70",
    "fm_fill": "#BCE2C3",
    "ddpm": "#567FB5",
    "ddpm_fill": "#D9E7F4",
    "accent": "#D97973",
    "accent_fill": "#FFC7CE",
    "cream": "#FFF9D9",
    "lavender": "#B798C8",
    "lavender_fill": "#F2CFEE",
    "obstacle": "#C96B65",
    "goal": "#65966F",
    "forbidden": "#D9E7F4",
    "boundary": "#6F8FB5",
}


PHYSICAL_OBSTACLES = (
    ((0.500, -0.100), 0.030),
    ((0.425, 0.080), 0.025),
    ((0.575, 0.080), 0.025),
    ((0.350, 0.260), 0.025),
    ((0.500, 0.260), 0.025),
    ((0.650, 0.260), 0.025),
)


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": COLORS["canvas"],
            "savefig.facecolor": COLORS["canvas"],
            "axes.facecolor": COLORS["canvas"],
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["secondary_text"],
            "ytick.color": COLORS["secondary_text"],
            "font.family": "STIXGeneral",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.3,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.55,
            "lines.markersize": 4.8,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.75,
            "legend.frameon": True,
            "legend.facecolor": COLORS["canvas"],
            "legend.edgecolor": COLORS["light_axis"],
            "legend.framealpha": 0.96,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, which="major")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["axis"])
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(length=2.8, width=0.7, color=COLORS["axis"], pad=2.5)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["text"],
    )


def method_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS["fm"],
            marker="o",
            markerfacecolor=COLORS["canvas"],
            markeredgewidth=1.0,
            linestyle="-",
            label="Flow Matching",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["ddpm"],
            marker="s",
            markerfacecolor=COLORS["canvas"],
            markeredgewidth=1.0,
            linestyle="--",
            label="Native DDPM",
        ),
    ]


def _halfspace_line(constraint, clearance: float = 0.0) -> tuple[float, float]:
    """Return y = m*x + d using the same physical tightening as evaluation."""
    p0 = np.asarray(constraint[0], dtype=float)
    p1 = np.asarray(constraint[1], dtype=float)
    m = (p1[1] - p0[1]) / (p1[0] - p0[0])
    normal = np.asarray([-1.0, 1.0 / m], dtype=float)
    normal /= np.linalg.norm(normal)
    if (m > 0 and constraint[2] == "below") or (m < 0 and constraint[2] == "above"):
        normal *= -1.0
    shifted = p0 + clearance * normal
    d = shifted[1] - m * shifted[0]
    return float(m), float(d)


def draw_task_geometry(
    ax: plt.Axes,
    *,
    ax_limits=((0.2, 0.8), (-0.3, 0.4)),
    halfspaces=(),
    analytic_circle=None,
    clearance: float = 0.025,
    show_tightening: bool = False,
    show_axis_labels: bool = True,
) -> None:
    """Draw the avoiding task with consistent, print-friendly semantics."""
    xlim, ylim = ax_limits
    xx = np.linspace(xlim[0], xlim[1], 480)

    if halfspaces:
        gx, gy = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 420),
            np.linspace(ylim[0], ylim[1], 420),
        )
        forbidden = np.zeros_like(gx, dtype=bool)
        for constraint in halfspaces:
            m, d = _halfspace_line(constraint, 0.0)
            if constraint[2] == "below":
                forbidden |= gy > m * gx + d
            else:
                forbidden |= gy < m * gx + d
        ax.contourf(
            gx,
            gy,
            forbidden.astype(float),
            levels=[0.5, 1.5],
            colors=[COLORS["forbidden"]],
            alpha=0.88,
            zorder=0,
        )
        for constraint in halfspaces:
            m, d = _halfspace_line(constraint, 0.0)
            ax.plot(xx, m * xx + d, color=COLORS["boundary"], linewidth=0.95, zorder=2)
            if show_tightening:
                mt, dt = _halfspace_line(constraint, clearance)
                ax.plot(
                    xx,
                    mt * xx + dt,
                    color=COLORS["boundary"],
                    linewidth=0.9,
                    linestyle=(0, (3, 2)),
                    alpha=0.9,
                    zorder=2,
                )

    for center, radius in PHYSICAL_OBSTACLES:
        ax.add_patch(
            Circle(
                center,
                radius,
                facecolor=COLORS["obstacle"],
                edgecolor="#9E504B",
                linewidth=0.65,
                zorder=5,
            )
        )

    if analytic_circle is not None:
        center = analytic_circle["center"]
        radius = analytic_circle["radius"]
        ax.add_patch(
            Circle(
                center,
                radius,
                facecolor=COLORS["lavender_fill"],
                edgecolor=COLORS["lavender"],
                linewidth=0.9,
                alpha=0.72,
                zorder=3,
            )
        )
        if show_tightening:
            ax.add_patch(
                Circle(
                    center,
                    radius + clearance,
                    facecolor="none",
                    edgecolor=COLORS["lavender"],
                    linewidth=0.9,
                    linestyle=(0, (3, 2)),
                    zorder=3,
                )
            )

    ax.axhline(0.35, color=COLORS["goal"], linewidth=1.35, zorder=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    if show_axis_labels:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(COLORS["light_axis"])
        spine.set_linewidth(0.65)
    ax.tick_params(length=2.5, width=0.6, color=COLORS["axis"], pad=2)


def task_legend_handles(*, include_demo: bool = False, include_tightening: bool = False):
    handles = []
    if include_demo:
        handles.append(Line2D([0], [0], color=COLORS["ddpm"], alpha=0.55, label="Human demonstration"))
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=6,
                linestyle="none",
                markerfacecolor=COLORS["obstacle"],
                markeredgecolor="#9E504B",
                label="Simulator obstacle",
            ),
            Line2D([0], [0], color=COLORS["boundary"], linewidth=1.1, label="Halfspace boundary"),
            Line2D([0], [0], color=COLORS["lavender"], linewidth=2.4, alpha=0.7, label="Circular exclusion"),
            Line2D([0], [0], color=COLORS["goal"], linewidth=1.4, label="Goal line"),
        ]
    )
    if include_tightening:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["axis"],
                linewidth=0.9,
                linestyle=(0, (3, 2)),
                label="Tightened boundary",
            )
        )
    return handles


def save_figure(fig: plt.Figure, output_dir, stem: str, *, dpi: int = 320) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.03)
