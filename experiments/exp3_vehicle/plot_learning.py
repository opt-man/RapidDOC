from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "figures" / "generated"
RESULTS = HERE / "data"
LEARNING = RESULTS / "training" / "ours" / "learning.csv"
LEARNING_ACADOS = RESULTS / "training" / "acados" / "learning.csv"

COLORS = {
    "reference": "#555555",
    "before": "#F40506",
    "after": "#0503F2",
    "start": "#21C95B",
    "ours": "#0503F2",
    "acados": "#F40506",
}


def add_direction_arrow(axis, x, y, index, color, linewidth):
    """Add a compact tangent arrow without altering the trajectory."""
    start = max(0, min(int(index), len(x) - 16))
    stop = start + 15
    axis.annotate(
        "", xy=(x[stop], y[stop]), xytext=(x[start], y[start]),
        arrowprops={
            "arrowstyle": "->", "color": color, "lw": linewidth,
            "mutation_scale": 10, "shrinkA": 0, "shrinkB": 0,
        },
        zorder=5,
    )


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No data rows in {path}")
    return {
        key: np.asarray([row[key] for row in rows])
        for key in rows[0]
    }


def complete_runs(stage: str) -> list[dict[str, object]]:
    runs = []
    for directory in sorted(
        (RESULTS / "physical_trajectories").glob(f"car_ours_{stage}_*")
    ):
        path = directory / "trajectory.csv"
        if not path.exists():
            continue
        try:
            data = read_csv(path)
        except RuntimeError:
            continue
        if data["status"][-1] != "reference_period_complete":
            continue
        x = data["x"].astype(float)
        y = data["y"].astype(float)
        ref_x = data["ref_x"].astype(float)
        ref_y = data["ref_y"].astype(float)
        error = np.hypot(x - ref_x, y - ref_y)
        runs.append({
            "name": directory.name,
            "path": path,
            "data": data,
            "rmse_m": float(np.sqrt(np.mean(error**2))),
        })
    if not runs:
        raise RuntimeError(
            f"No complete {stage} trajectories found in "
            f"{RESULTS / 'physical_trajectories'}"
        )
    return runs


def representative(runs: list[dict[str, object]]) -> dict[str, object]:
    values = np.asarray([run["rmse_m"] for run in runs], dtype=float)
    median = float(np.median(values))
    return min(runs, key=lambda run: abs(float(run["rmse_m"]) - median))


def main() -> None:
    before_runs = complete_runs("before")
    after_runs = complete_runs("after")
    before = representative(before_runs)
    after = representative(after_runs)
    learning = read_csv(LEARNING)
    learning_acados = read_csv(LEARNING_ACADOS)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.8,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Native IEEE double-column width keeps the text at its specified size.
    fig = plt.figure(figsize=(7.14, 2.55), constrained_layout=False)
    grid = fig.add_gridspec(
        1, 2, width_ratios=(0.78, 1.32),
        left=0.085, right=0.985, bottom=0.25, top=0.79,
        wspace=0.30,
    )
    ax_loss = fig.add_subplot(grid[0, 0])
    ax_track = fig.add_subplot(grid[0, 1])

    b = before["data"]
    a = after["data"]
    ref_x = a["ref_x"].astype(float)
    ref_y = a["ref_y"].astype(float)
    before_x = b["x"].astype(float)
    before_y = b["y"].astype(float)
    after_x = a["x"].astype(float)
    after_y = a["y"].astype(float)
    ax_track.plot(
        ref_x, ref_y, ls="--", color=COLORS["reference"], lw=1.3,
        label="Reference", zorder=1,
    )
    ax_track.plot(
        before_x, before_y, color=COLORS["before"], lw=1.85,
        label="Before", solid_capstyle="round", zorder=2,
    )
    ax_track.plot(
        after_x, after_y, color=COLORS["after"], lw=2.0,
        label="After", solid_capstyle="round", zorder=3,
    )
    add_direction_arrow(ax_track, before_x, before_y, 175,
                        COLORS["before"], 1.85)
    add_direction_arrow(ax_track, before_x, before_y, 470,
                        COLORS["before"], 1.85)
    add_direction_arrow(ax_track, after_x, after_y, 175,
                        COLORS["after"], 2.0)
    add_direction_arrow(ax_track, after_x, after_y, 470,
                        COLORS["after"], 2.0)

    # The figure-eight is closed, so retain both measured endpoints and use
    # nested markers to distinguish their nearly coincident locations.
    ax_track.scatter(
        [after_x[0]], [after_y[0]], s=75, facecolor="white",
        edgecolor=COLORS["start"], linewidth=2.2, zorder=6,
    )
    ax_track.scatter(
        [before_x[-1]], [before_y[-1]], s=52, marker="x",
        color=COLORS["before"], linewidth=2.0, zorder=7,
    )
    ax_track.scatter(
        [after_x[-1]], [after_y[-1]], s=52, marker="x",
        color=COLORS["after"], linewidth=2.0, zorder=8,
    )
    ax_track.set_xlabel("Position $x$ (m)")
    ax_track.set_ylabel("Position $y$ (m)")
    ax_track.set_aspect("equal", adjustable="box")
    ax_track.legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
        ncol=3, handlelength=2.0, columnspacing=0.8,
    )

    epoch = learning["id"].astype(float)
    loss = learning["loss"].astype(float)
    epoch_acados = learning_acados["id"].astype(float)
    loss_acados = learning_acados["loss"].astype(float)
    ax_loss.semilogy(
        epoch, loss, color=COLORS["ours"], lw=1.8,
        marker="o", markevery=50, ms=4.0, mfc="white", mew=1.0,
        label="RapidDOC",
    )
    ax_loss.semilogy(
        epoch_acados, loss_acados, color=COLORS["acados"], lw=1.7,
        ls=(0, (5, 2.5)), marker="s", markevery=50, ms=3.8,
        mfc="white", mew=1.0, label="acados",
    )
    ax_loss.set_xlabel("Learning epoch")
    ax_loss.set_ylabel("Outer loss")
    ax_loss.legend(frameon=False, loc="upper right", handlelength=2.5)

    for axis in (ax_track, ax_loss):
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        axis.grid(color="#E3E3E3", lw=0.55, ls="--")
        axis.tick_params(direction="out", length=3, width=0.7)

    loss_bounds = ax_loss.get_position()
    track_bounds = ax_track.get_position()
    fig.text(0.5 * (loss_bounds.x0 + loss_bounds.x1), 0.055,
             "(a) Learning convergence",
             ha="center", va="center", weight="bold")
    fig.text(0.5 * (track_bounds.x0 + track_bounds.x1), 0.055,
             "(b) Figure-eight tracking",
             ha="center", va="center", weight="bold")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "fig5_figure8_learning.pdf", dpi=300)
    fig.savefig(OUTPUT / "fig5_figure8_learning.svg", dpi=300)
    fig.savefig(OUTPUT / "fig5_figure8_learning.png", dpi=300)
    plt.close(fig)

    metadata = {
        "selection_rule": "complete run with RMSE closest to the stage median",
        "before_complete_runs": [
            {"name": run["name"], "rmse_cm": 100 * float(run["rmse_m"])}
            for run in before_runs
        ],
        "after_complete_runs": [
            {"name": run["name"], "rmse_cm": 100 * float(run["rmse_m"])}
            for run in after_runs
        ],
        "selected_before": before["name"],
        "selected_after": after["name"],
        "learning_csv": LEARNING.relative_to(HERE).as_posix(),
        "learning_acados_csv": LEARNING_ACADOS.relative_to(HERE).as_posix(),
    }
    (OUTPUT / "figure8_learning_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
