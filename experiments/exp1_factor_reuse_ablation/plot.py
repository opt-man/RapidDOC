from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "gmres_real_ocp_ablation_raw.csv"
OUTPUTS = (ROOT / "regenerated" / "fig2_factor_reuse_ablation",)
PROBLEMS = ("cartpole", "quadrotor", "robotarm")
LABELS = ("CartPole", "Quadrotor", "Robot Arm")
METHODS = ("Ours", "Target refactor", "No-reuse GMRES")
DISPLAY = ("RapidDOC", "Refactor", "No reuse")
COLORS = ("#2457FF", "#21C95B", "#F03A3A")
HATCHES = (None, "\\\\", "///")


def load_statistics() -> tuple[np.ndarray, np.ndarray, int]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    with DATA.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"no rows found in {DATA}")
    for row in rows:
        if int(row["converged"]) != 1:
            raise RuntimeError("nonconverged rows cannot enter the figure")
        grouped[(row["problem"], row["method"])].append(
            float(row["time_median_ms"])
        )
    counts = {len(values) for values in grouped.values()}
    if counts != {32}:
        raise RuntimeError(f"expected 32 instances per bar, found {sorted(counts)}")
    means = np.asarray([
        [np.mean(grouped[(problem, method)]) for problem in PROBLEMS]
        for method in METHODS
    ])
    standard_deviations = np.asarray([
        [np.std(grouped[(problem, method)], ddof=1) for problem in PROBLEMS]
        for method in METHODS
    ])
    return means, standard_deviations, counts.pop()


def value_label(value: float) -> str:
    if value < 0.1:
        return f"{value:.3f}"
    if value < 10.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def main() -> None:
    means, deviations, count = load_statistics()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.8,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    # Native single-column width keeps the rendered text close to the
    # IEEE body size when included with width=\columnwidth.
    fig, axis = plt.subplots(figsize=(3.49, 2.55))
    positions = np.arange(len(PROBLEMS))
    width = 0.23
    offsets = (-width, 0.0, width)
    groups = []
    for method_index, offset in enumerate(offsets):
        values = means[method_index]
        errors = deviations[method_index]
        yerr = np.vstack((np.minimum(errors, 0.92 * values), errors))
        bars = axis.bar(
            positions + offset, values, width,
            yerr=yerr, capsize=2.7,
            error_kw={
                "ecolor": COLORS[method_index],
                "lw": 0.85,
                "capthick": 0.85,
            },
            label=DISPLAY[method_index],
            color=mcolors.to_rgba(COLORS[method_index], 0.16),
            edgecolor=COLORS[method_index], linewidth=1.0,
            hatch=HATCHES[method_index], zorder=3,
        )
        groups.append(bars)

    axis.set_yscale("log")
    axis.set_ylim(0.015, 20000.0)
    axis.set_ylabel("Backward time (ms)")
    axis.set_xticks(positions, LABELS)
    axis.yaxis.set_major_locator(LogLocator(base=10.0))
    axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 5)))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(
        axis="y", which="major", color="#E1E1E1",
        linestyle="--", linewidth=0.55,
    )
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False,
        ncol=3, columnspacing=1.5, handletextpad=0.45,
    )
    # Nudge the middle label just enough to keep it off the taller bar.
    label_x_offsets = (-2.0, -1.0, 0.0)
    label_y_offsets = (3.0, 3.0, 3.0)
    for method_index, bars in enumerate(groups):
        for problem_index, bar in enumerate(bars):
            value = float(bar.get_height())
            error_top = value + float(deviations[method_index, problem_index])
            axis.annotate(
                value_label(value),
                (bar.get_x() + bar.get_width() / 2.0, error_top),
                xytext=(
                    label_x_offsets[method_index],
                    label_y_offsets[method_index],
                ),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7.6,
            )

    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.75)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.18, top=0.84)
    for stem in OUTPUTS:
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(
            stem.with_suffix(".png"), dpi=600,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
