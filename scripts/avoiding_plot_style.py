"""Shared geometry and visual semantics for the D3IL avoiding task."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np

from thesis_plot_style import COLORS, PHYSICAL_OBSTACLES


GOAL_Y = 0.35
START = np.asarray([0.525, -0.280])
CLEARANCE = 0.025

NAVY = "#163A70"
CORAL = "#C95850"
CORAL_FILL = "#FBE7E5"
LAVENDER = "#8F72C7"
LAVENDER_FILL = "#EEE8FA"
PANEL = "#FBFCFF"
PANEL_EDGE = "#CBD6E8"
TIGHTENED = "#4F65A4"
START_FILL = "#39A8A0"


SCENES = (
    ("top-right-hard", "Right-side crossing", (1,), 4),
    ("top-left-hard", "Left-side crossing", (0,), 3),
    ("both-hard", "Central corridor", (2, 3), 5),
)


def halfspace_line(constraint, clearance: float = 0.0) -> tuple[float, float]:
    p0 = np.asarray(constraint[0], dtype=float)
    p1 = np.asarray(constraint[1], dtype=float)
    slope = (p1[1] - p0[1]) / (p1[0] - p0[0])
    normal = np.asarray([-1.0, 1.0 / slope], dtype=float)
    normal /= np.linalg.norm(normal)
    if (slope > 0 and constraint[2] == "below") or (
        slope < 0 and constraint[2] == "above"
    ):
        normal *= -1.0
    shifted = p0 + clearance * normal
    return float(slope), float(shifted[1] - slope * shifted[0])


def style_environment_axis(ax: plt.Axes, ax_limits) -> None:
    ax.set_facecolor(PANEL)
    ax.set_xlim(ax_limits[0])
    ax.set_ylim(ax_limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.9)


def draw_common_geometry(
    ax: plt.Axes,
    ax_limits,
    *,
    label_start: bool = False,
    label_goal: bool = False,
    emphasise_terminals: bool = False,
) -> None:
    for center, radius in PHYSICAL_OBSTACLES:
        ax.add_patch(
            Circle(
                center,
                radius,
                facecolor=COLORS["obstacle"],
                edgecolor="#984B46",
                linewidth=0.8,
                zorder=7,
            )
        )
    if emphasise_terminals:
        ax.axhspan(
            GOAL_Y - 0.007,
            GOAL_Y + 0.007,
            color=COLORS["goal"],
            alpha=0.11,
            zorder=7,
        )
    ax.axhline(
        GOAL_Y,
        color=COLORS["goal"],
        linewidth=2.25 if emphasise_terminals else 1.8,
        zorder=8,
    )
    ax.scatter(
        [START[0]],
        [START[1]],
        s=68 if emphasise_terminals else 42,
        facecolor=COLORS["canvas"],
        edgecolor=NAVY,
        linewidth=1.5 if emphasise_terminals else 1.25,
        zorder=10,
    )
    ax.scatter(
        [START[0]],
        [START[1]],
        s=18 if emphasise_terminals else 10,
        color=START_FILL,
        zorder=11,
    )
    if label_start:
        ax.annotate(
            "START",
            xy=START,
            xytext=(9, 6),
            textcoords="offset points",
            fontsize=7.0,
            fontweight="bold",
            color=NAVY,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor=COLORS["canvas"],
                edgecolor="none",
                alpha=0.86,
            ),
            zorder=12,
        )
    if label_goal:
        ax.text(
            ax_limits[0][1] - 0.008,
            GOAL_Y + 0.012,
            "GOAL  $y=0.35$",
            ha="right",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color=COLORS["goal"],
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor=COLORS["canvas"],
                edgecolor="none",
                alpha=0.86,
            ),
            zorder=12,
        )


def draw_forbidden_halfspaces(ax: plt.Axes, constraints, ax_limits) -> None:
    x_grid, y_grid = np.meshgrid(
        np.linspace(*ax_limits[0], 500),
        np.linspace(*ax_limits[1], 500),
    )
    forbidden = np.zeros_like(x_grid, dtype=bool)
    xx = np.linspace(*ax_limits[0], 500)
    for constraint in constraints:
        slope, intercept = halfspace_line(constraint)
        if constraint[2] == "below":
            forbidden |= y_grid > slope * x_grid + intercept
        else:
            forbidden |= y_grid < slope * x_grid + intercept
    ax.contourf(
        x_grid,
        y_grid,
        forbidden.astype(float),
        levels=[0.5, 1.5],
        colors=[CORAL_FILL],
        alpha=0.72,
        zorder=1,
    )
    ax.contourf(
        x_grid,
        y_grid,
        forbidden.astype(float),
        levels=[0.5, 1.5],
        colors=["none"],
        hatches=["////"],
        zorder=2,
    )
    for constraint in constraints:
        slope, intercept = halfspace_line(constraint)
        ax.plot(xx, slope * xx + intercept, color=NAVY, linewidth=1.45, zorder=5)
        tightened_slope, tightened_intercept = halfspace_line(constraint, CLEARANCE)
        ax.plot(
            xx,
            tightened_slope * xx + tightened_intercept,
            color=TIGHTENED,
            linewidth=1.15,
            linestyle=(0, (3, 2)),
            zorder=5,
        )


def draw_scene(ax: plt.Axes, config, halfspace_indices, circle_index, ax_limits) -> None:
    constraints = [
        config["halfspace_constraints"]["avoiding-d3il"][index]
        for index in halfspace_indices
    ]
    circle = config["obstacle_constraints"]["avoiding-d3il"][circle_index]
    draw_forbidden_halfspaces(ax, constraints, ax_limits)
    centre = circle["center"]
    radius = float(circle["radius"])
    ax.add_patch(
        Circle(
            centre,
            radius,
            facecolor=LAVENDER_FILL,
            edgecolor=LAVENDER,
            linewidth=1.15,
            alpha=0.92,
            zorder=4,
        )
    )
    ax.add_patch(
        Circle(
            centre,
            radius + CLEARANCE,
            facecolor="none",
            edgecolor=LAVENDER,
            linewidth=1.1,
            linestyle=(0, (3, 2)),
            zorder=4,
        )
    )
    draw_common_geometry(ax, ax_limits)
    ax.text(
        0.5 * (ax_limits[0][0] + ax_limits[0][1]),
        ax_limits[1][1] - 0.018,
        "TEST-TIME CONSTRAINTS",
        ha="center",
        va="top",
        fontsize=6.3,
        fontweight="bold",
        color=CORAL,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor=COLORS["canvas"],
            edgecolor="none",
            alpha=0.9,
        ),
        zorder=12,
    )


def scene_legend_handles() -> list:
    return [
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
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=5.5,
            linestyle="none",
            markerfacecolor=COLORS["canvas"],
            markeredgecolor=NAVY,
            markeredgewidth=1.1,
            label="Fixed start",
        ),
        Line2D([0], [0], color=COLORS["goal"], linewidth=1.7, label="Goal line"),
        Patch(facecolor=CORAL_FILL, edgecolor=CORAL, hatch="////", label="Forbidden halfspace"),
        Patch(facecolor=LAVENDER_FILL, edgecolor=LAVENDER, label="Circular exclusion"),
        Line2D(
            [0],
            [0],
            color=TIGHTENED,
            linewidth=1.1,
            linestyle=(0, (3, 2)),
            label=r"Tightened boundary ($\delta=0.025$)",
        ),
    ]
