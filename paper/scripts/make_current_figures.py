#!/usr/bin/env python3
"""Generate current AdaProx/AdaDitto paper figures from experiment artifacts."""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pfllib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "Figures" / "current"


PALETTE = {
    "ditto": "#2f6f9f",
    "ada": "#c44e52",
    "oracle": "#555555",
    "muted": "#777777",
    "grid": "#dddddd",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 7.8,
            "axes.titlesize": 8.3,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(value: str) -> float:
    return float(value)


def coefficient_from_run_id(run_id: str) -> float | None:
    for prefix in ("lam_", "mu_", "ditto_jr0.1_lam", "ditto_jr0.5_lam"):
        if run_id.startswith(prefix):
            raw = run_id[len(prefix) :]
            return float(raw)
    return None


def score_for(row: dict[str, str]) -> float:
    if row["algorithm"] in {"Ditto", "AdaProxDitto"}:
        return f(row["personalized_accuracy"])
    return f(row["global_accuracy"])


def log_position(x: float) -> float:
    if x == 0:
        return -3.5
    return math.log10(x)


def lambda_points(rows: list[dict[str, str]]) -> tuple[list[float], list[float], int]:
    points = []
    for row in rows:
        lam = coefficient_from_run_id(row["run_id"])
        if lam is None:
            lam = f(row["coefficient_mean"])
        points.append((lam, score_for(row)))
    points = sorted(points, key=lambda p: (p[0] != 0, p[0]))
    xs = [log_position(p[0]) for p in points]
    ys = [100 * p[1] for p in points]
    best_idx = int(np.argmax(ys)) if ys else 0
    return xs, ys, best_idx


def style_lambda_axis(ax: mpl.axes.Axes) -> None:
    ax.grid(axis="y")
    ax.set_xlabel(r"fixed $\lambda$")
    ax.set_xticks([-3.5, -3, -2, -1, 0])
    ax.set_xticklabels(["0", ".001", ".01", ".1", "1"])
    ax.margins(x=0.08, y=0.13)


def plot_coefficient_brittleness() -> None:
    phase1 = read_csv(ROOT / "experiments" / "phase1_selection.csv")
    shake01 = read_csv(ROOT / "experiments" / "shakespeare_sweep_selection.csv")
    shake05 = read_csv(ROOT / "experiments" / "shakespeare_shift_selection.csv")

    panels: list[tuple[str, list[tuple[str, list[dict[str, str]]]]]] = []
    for split, title in [
        ("Cifar100_practical", "CIFAR practical\n(full participation)"),
        ("Cifar100_pathological", "CIFAR pathological\n(full participation)"),
    ]:
        rows = [
            r
            for r in phase1
            if r["dataset"] == split and r["algorithm"] == "Ditto" and r["run_id"].startswith("lam_")
        ]
        panels.append((title, [("Ditto", rows)]))

    shake_rows_01 = [r for r in shake01 if r["algorithm"] == "Ditto"]
    shake_rows_05 = [r for r in shake05 if r["algorithm"] == "Ditto"]
    panels.append(("Shakespeare\n(participation stable)", [("jr=0.1", shake_rows_01), ("jr=0.5", shake_rows_05)]))

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), sharey=False)
    for ax, (title, series) in zip(axes, panels):
        for idx, (label, rows) in enumerate(series):
            xs, ys, best_idx = lambda_points(rows)
            color = PALETTE["ditto"] if idx == 0 else PALETTE["muted"]
            marker = "o" if idx == 0 else "s"
            ax.plot(xs, ys, marker=marker, linewidth=1.35, markersize=3.4, color=color, label=label)
            ax.scatter([xs[best_idx]], [ys[best_idx]], s=34, facecolor="white", edgecolor=color, linewidth=1.2, zorder=5)
        ax.set_title(title)
        style_lambda_axis(ax)
        if len(series) > 1:
            ax.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("personalized accuracy (%)")
    axes[0].text(-0.11, 1.07, "A", transform=axes[0].transAxes, weight="bold")
    axes[1].text(-0.11, 1.07, "B", transform=axes[1].transAxes, weight="bold")
    axes[2].text(-0.11, 1.07, "C", transform=axes[2].transAxes, weight="bold")
    save(fig, "coefficient_brittleness")


def plot_lambda_stress_grid() -> None:
    phase1 = read_csv(ROOT / "experiments" / "phase1_selection.csv")
    shake01 = read_csv(ROOT / "experiments" / "shakespeare_sweep_selection.csv")
    shake05 = read_csv(ROOT / "experiments" / "shakespeare_shift_selection.csv")

    fig, axes = plt.subplots(2, 3, figsize=(7.05, 3.85), sharey=False)
    flat = axes.flatten()

    top_panels = [
        (
            "CIFAR practical\nfull participation",
            [
                (
                    "Full",
                    [
                        r
                        for r in phase1
                        if r["dataset"] == "Cifar100_practical"
                        and r["algorithm"] == "Ditto"
                        and r["run_id"].startswith("lam_")
                    ],
                    PALETTE["ditto"],
                    "o",
                    "-",
                )
            ],
        ),
        (
            "CIFAR pathological\nfull participation",
            [
                (
                    "Full",
                    [
                        r
                        for r in phase1
                        if r["dataset"] == "Cifar100_pathological"
                        and r["algorithm"] == "Ditto"
                        and r["run_id"].startswith("lam_")
                    ],
                    PALETTE["ditto"],
                    "o",
                    "-",
                )
            ],
        ),
        (
            "Shakespeare\nparticipation stable",
            [
                ("10%", [r for r in shake01 if r["algorithm"] == "Ditto"], PALETTE["ditto"], "o", "-"),
                ("50%", [r for r in shake05 if r["algorithm"] == "Ditto"], PALETTE["muted"], "s", "--"),
            ],
        ),
    ]
    bottom_panels = [
        (
            "CIFAR practical\nparticipation shift",
            [
                ("Full", rows_for_lambda_landscape("Cifar100_practical", 1.0), PALETTE["ditto"], "o", "-"),
                ("50%", rows_for_lambda_landscape("Cifar100_practical", 0.5), PALETTE["muted"], "s", "--"),
                ("20%", rows_for_lambda_landscape("Cifar100_practical", 0.2), PALETTE["ada"], "^", "-."),
            ],
        ),
        (
            "CIFAR pathological\nparticipation shift",
            [
                ("Full", rows_for_lambda_landscape("Cifar100_pathological", 1.0), PALETTE["ditto"], "o", "-"),
                ("50%", rows_for_lambda_landscape("Cifar100_pathological", 0.5), PALETTE["muted"], "s", "--"),
                ("20%", rows_for_lambda_landscape("Cifar100_pathological", 0.2), PALETTE["ada"], "^", "-."),
            ],
        ),
    ]

    for ax, (title, series), panel in zip(flat[:3], top_panels, ["A", "B", "C"]):
        for label, rows, color, marker, linestyle in series:
            xs, ys, best_idx = lambda_points(rows)
            ax.plot(xs, ys, marker=marker, linestyle=linestyle, linewidth=1.15, markersize=3.1, color=color, label=label)
            ax.scatter([xs[best_idx]], [ys[best_idx]], s=58, marker="*", facecolor="white", edgecolor=color, linewidth=1.0, zorder=5)
        ax.set_title(title)
        style_lambda_axis(ax)
        ax.text(-0.12, 1.07, panel, transform=ax.transAxes, weight="bold")
        if len(series) > 1:
            ax.legend(frameon=False, loc="lower right")

    for ax, (title, series), panel in zip(flat[3:5], bottom_panels, ["D", "E"]):
        for label, rows, color, marker, linestyle in series:
            xs, ys, best_idx = lambda_points(rows)
            ax.plot(xs, ys, marker=marker, linestyle=linestyle, linewidth=1.1, markersize=3.0, color=color, label=label)
            ax.scatter([xs[best_idx]], [ys[best_idx]], s=56, marker="*", facecolor="white", edgecolor=color, linewidth=1.0, zorder=5)
        ax.set_title(title)
        style_lambda_axis(ax)
        ax.text(-0.12, 1.07, panel, transform=ax.transAxes, weight="bold")

    axes[0, 0].set_ylabel("personalized accuracy (%)")
    axes[1, 0].set_ylabel("personalized accuracy (%)")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    axes[1, 2].axis("off")
    axes[1, 2].legend(handles, labels, frameon=False, title="participation", loc="center left", title_fontsize=7.0)
    save(fig, "lambda_stress_grid")


def parse_jr(run_id: str) -> float:
    marker = "_jr"
    start = run_id.index(marker) + len(marker)
    end = run_id.find("_", start)
    if end == -1:
        end = len(run_id)
    return float(run_id[start:end])


def summarize_dropout() -> dict[tuple[str, float], tuple[float, float]]:
    rows = read_csv(ROOT / "experiments" / "phase_dropout_summary.csv")
    values: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        if row["dataset"] != "Cifar100_pathological":
            continue
        if "cdr0.0" not in row["run_id"]:
            continue
        if row["algorithm"] not in {"Ditto", "AdaProxDitto"}:
            continue
        values[(row["algorithm"], parse_jr(row["run_id"]))].append(score_for(row))
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in values.items()}


def summarize_shakespeare_comparison() -> dict[tuple[str, float], tuple[float, float]]:
    values: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in read_csv(ROOT / "experiments" / "shakespeare_core_summary.csv"):
        if row["algorithm"] == "Ditto":
            values[("Tuned Ditto", 0.1)].append(score_for(row))
        elif row["algorithm"] == "AdaProxDitto":
            values[("AdaDitto", 0.1)].append(score_for(row))

    shift_rows = read_csv(ROOT / "experiments" / "shakespeare_shift_selection.csv")
    ditto_shift = [row for row in shift_rows if row["algorithm"] == "Ditto"]
    best_ditto = max(ditto_shift, key=score_for)
    values[("Tuned Ditto", 0.5)].append(score_for(best_ditto))
    for row in shift_rows:
        if row["algorithm"] == "AdaProxDitto":
            values[("AdaDitto", 0.5)].append(score_for(row))

    return {
        k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        for k, v in values.items()
    }


def plot_participation_shift() -> None:
    summary = summarize_dropout()
    x_values = [1.0, 0.5, 0.2]
    x_pos = np.arange(len(x_values))
    labels = ["Full", "50%", "20%"]

    ditto = np.array([summary[("Ditto", jr)][0] for jr in x_values]) * 100
    ditto_sd = np.array([summary[("Ditto", jr)][1] for jr in x_values]) * 100
    ada = np.array([summary[("AdaProxDitto", jr)][0] for jr in x_values]) * 100
    ada_sd = np.array([summary[("AdaProxDitto", jr)][1] for jr in x_values]) * 100

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.errorbar(
        x_pos,
        ditto,
        yerr=ditto_sd,
        marker="o",
        color=PALETTE["ditto"],
        linewidth=1.55,
        markersize=4,
        capsize=2.2,
        label=r"Ditto, frozen $\lambda=0$",
    )
    ax.errorbar(
        x_pos,
        ada,
        yerr=ada_sd,
        marker="s",
        color=PALETTE["ada"],
        linewidth=1.55,
        markersize=4,
        capsize=2.2,
        label="AdaDitto",
    )
    ax.axvspan(1.55, 2.45, color=PALETTE["ada"], alpha=0.06, linewidth=0)
    ax.annotate(
        "+1.8 pts",
        xy=(2, ada[-1]),
        xytext=(1.55, ada[-1] + 2.5),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": PALETTE["ada"]},
        color=PALETTE["ada"],
        fontsize=8,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("participating clients per round")
    ax.set_ylabel("personalized accuracy (%)", labelpad=6)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower left")
    ax.margins(x=0.08, y=0.15)
    save(fig, "participation_shift")


def plot_shift_response() -> None:
    summary = summarize_dropout()
    x_values = [1.0, 0.5, 0.2]
    x_pos = np.arange(len(x_values))
    labels = ["Full", "50%", "20%"]

    ditto = np.array([summary[("Ditto", jr)][0] for jr in x_values]) * 100
    ditto_sd = np.array([summary[("Ditto", jr)][1] for jr in x_values]) * 100
    ada = np.array([summary[("AdaProxDitto", jr)][0] for jr in x_values]) * 100
    ada_sd = np.array([summary[("AdaProxDitto", jr)][1] for jr in x_values]) * 100

    cifar_dirs = run_dirs_from_summary(
        ROOT / "experiments" / "phase_dropout_summary.csv",
        "Cifar100_pathological",
        "AdaProxDitto",
        "cdr0.0_jr0.2",
    )
    rounds, mean, lo, hi = trajectory_stats(controller_rows(cifar_dirs))
    shake = summarize_shakespeare_comparison()

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.30), gridspec_kw={"width_ratios": [1.05, 1.0, 0.82]})
    ax = axes[0]
    ax.errorbar(
        x_pos,
        ditto,
        yerr=ditto_sd,
        marker="o",
        color=PALETTE["ditto"],
        linewidth=1.55,
        markersize=4,
        capsize=2.2,
        label=r"Ditto, frozen $\lambda=0$",
    )
    ax.errorbar(
        x_pos,
        ada,
        yerr=ada_sd,
        marker="s",
        color=PALETTE["ada"],
        linewidth=1.55,
        markersize=4,
        capsize=2.2,
        label="AdaDitto",
    )
    ax.axvspan(1.55, 2.45, color=PALETTE["ada"], alpha=0.06, linewidth=0)
    ax.annotate(
        "+1.8 pts",
        xy=(2, ada[-1]),
        xytext=(1.47, ada[-1] + 2.6),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": PALETTE["ada"]},
        color=PALETTE["ada"],
        fontsize=8,
    )
    ax.set_title("CIFAR pathological\nshift outcome")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("participating clients per round")
    ax.set_ylabel("personalized accuracy (%)", labelpad=6)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower left")
    ax.margins(x=0.08, y=0.15)
    ax.text(-0.12, 1.07, "A", transform=ax.transAxes, weight="bold")

    ax = axes[1]
    ax.plot(rounds, mean, color=PALETTE["ada"], linewidth=1.5, label="mean")
    ax.fill_between(rounds, lo, hi, color=PALETTE["ada"], alpha=0.16, linewidth=0, label="10-90%")
    ax.set_title("AdaDitto at 20% participation\nadaptive response")
    ax.set_xlabel("round")
    ax.set_ylabel(r"adaptive $\lambda$")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower right")
    ax.margins(x=0.02, y=0.16)
    ax.text(-0.12, 1.07, "B", transform=ax.transAxes, weight="bold")

    ax = axes[2]
    x = np.arange(2)
    width = 0.36
    labels = ["10%", "50%"]
    tuned = np.array([shake[("Tuned Ditto", 0.1)][0], shake[("Tuned Ditto", 0.5)][0]]) * 100
    tuned_sd = np.array([shake[("Tuned Ditto", 0.1)][1], shake[("Tuned Ditto", 0.5)][1]]) * 100
    ada_shake = np.array([shake[("AdaDitto", 0.1)][0], shake[("AdaDitto", 0.5)][0]]) * 100
    ada_shake_sd = np.array([shake[("AdaDitto", 0.1)][1], shake[("AdaDitto", 0.5)][1]]) * 100
    ax.bar(x - width / 2, tuned, width, yerr=tuned_sd, color=PALETTE["ditto"], alpha=0.85, capsize=2, label="Tuned Ditto")
    ax.bar(x + width / 2, ada_shake, width, yerr=ada_shake_sd, color=PALETTE["ada"], alpha=0.85, capsize=2, label="AdaDitto")
    ax.set_title("Shakespeare\ntext result")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("participation")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(47.0, 52.2)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", fontsize=6.3, handlelength=0.9, labelspacing=0.25)
    ax.margins(x=0.08)
    ax.text(-0.18, 1.07, "C", transform=ax.transAxes, weight="bold")
    save(fig, "shift_response")


def rows_for_lambda_landscape(split: str, participation: float) -> list[dict[str, str]]:
    if participation == 1.0:
        rows = read_csv(ROOT / "experiments" / "phase1_selection.csv")
        return [
            r
            for r in rows
            if r["dataset"] == split and r["algorithm"] == "Ditto" and r["run_id"].startswith("lam_")
        ]

    rows = read_csv(ROOT / "experiments" / "core_partic_sweep_selection.csv")
    needle = f"ditto_jr{participation:.1f}_lam"
    return [
        r
        for r in rows
        if r["dataset"] == split and r["algorithm"] == "Ditto" and r["run_id"].startswith(needle)
    ]


def plot_lambda_participation_landscape() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.35), sharey=False)
    split_titles = [
        ("Cifar100_practical", "CIFAR practical"),
        ("Cifar100_pathological", "CIFAR pathological"),
    ]
    styles = {
        1.0: ("Full", PALETTE["ditto"], "o", "-"),
        0.5: ("50%", PALETTE["muted"], "s", "--"),
        0.2: ("20%", PALETTE["ada"], "^", "-."),
    }
    for ax, (split, title) in zip(axes, split_titles):
        for jr in [1.0, 0.5, 0.2]:
            rows = rows_for_lambda_landscape(split, jr)
            points = []
            for row in rows:
                lam = coefficient_from_run_id(row["run_id"])
                if lam is None:
                    lam = f(row["coefficient_mean"])
                points.append((lam, score_for(row)))
            points = sorted(points, key=lambda p: (p[0] != 0, p[0]))
            xs = [log_position(p[0]) for p in points]
            ys = [100 * p[1] for p in points]
            label, color, marker, linestyle = styles[jr]
            ax.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.25,
                markersize=3.2,
                color=color,
                label=label,
            )
            if ys:
                best_idx = int(np.argmax(ys))
                ax.scatter(
                    [xs[best_idx]],
                    [ys[best_idx]],
                    s=30,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.1,
                    zorder=5,
                )
        ax.set_title(title)
        ax.grid(axis="y")
        ax.set_xlabel(r"fixed $\lambda$")
        ax.set_xticks([-3.5, -3, -2, -1, 0])
        ax.set_xticklabels(["0", ".001", ".01", ".1", "1"])
        ax.margins(x=0.08, y=0.13)
    axes[0].set_ylabel("personalized accuracy (%)")
    axes[1].legend(frameon=False, title="participation", loc="lower right", title_fontsize=7.5)
    axes[0].text(-0.10, 1.06, "A", transform=axes[0].transAxes, weight="bold")
    axes[1].text(-0.10, 1.06, "B", transform=axes[1].transAxes, weight="bold")
    save(fig, "lambda_participation_landscape")


def controller_rows(run_dirs: list[Path]) -> dict[int, list[float]]:
    by_round: dict[int, list[float]] = defaultdict(list)
    for run_dir in run_dirs:
        path = run_dir / "results" / "client_controller.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            by_round[int(float(row["round"]))].append(f(row["coefficient_final"]))
    return by_round


def run_dirs_from_summary(summary_path: Path, dataset: str, algorithm: str, contains: str) -> list[Path]:
    rows = read_csv(summary_path)
    run_dirs = []
    for row in rows:
        if row["dataset"] != dataset or row["algorithm"] != algorithm or contains not in row["run_id"]:
            continue
        run_dir = Path(row["run_dir"])
        run_dirs.append(run_dir if run_dir.is_absolute() else ROOT / run_dir)
    return run_dirs


def trajectory_stats(by_round: dict[int, list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rounds = np.array(sorted(by_round))
    mean = np.array([np.mean(by_round[r]) for r in rounds])
    lo = np.array([np.percentile(by_round[r], 10) for r in rounds])
    hi = np.array([np.percentile(by_round[r], 90) for r in rounds])
    return rounds, mean, lo, hi


def plot_controller_trajectories() -> None:
    cifar_dirs = run_dirs_from_summary(
        ROOT / "experiments" / "phase_dropout_summary.csv",
        "Cifar100_pathological",
        "AdaProxDitto",
        "cdr0.0_jr0.2",
    )
    shake_dirs = run_dirs_from_summary(
        ROOT / "experiments" / "shakespeare_core_summary.csv",
        "Shakespeare",
        "AdaProxDitto",
        "jr0.1",
    )
    data = [
        ("CIFAR pathological\n20% participation", trajectory_stats(controller_rows(cifar_dirs))),
        ("Shakespeare\n10% participation", trajectory_stats(controller_rows(shake_dirs))),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25), sharey=False)
    for ax, (title, (rounds, mean, lo, hi)) in zip(axes, data):
        ax.plot(rounds, mean, color=PALETTE["ada"], linewidth=1.5, label="mean")
        ax.fill_between(rounds, lo, hi, color=PALETTE["ada"], alpha=0.16, linewidth=0, label="10-90%")
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.set_ylabel(r"adaptive $\lambda$")
        ax.grid(axis="y")
        ax.margins(x=0.02, y=0.16)
        ax.legend(frameon=False, loc="best")
    save(fig, "controller_trajectories")


def plot_controller_sensitivity() -> None:
    def config_name(run_id: str) -> str:
        return run_id.rsplit("_seed", 1)[0]

    def ranked_config_means(path: Path) -> list[tuple[str, float, float]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in read_csv(path):
            grouped[config_name(row["run_id"])].append(100 * score_for(row))
        ranked = []
        for name, scores in grouped.items():
            ranked.append((name, float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0))
        return sorted(ranked, key=lambda item: item[1])

    cifar = ranked_config_means(ROOT / "experiments" / "cscan_cifar_path_jr0.2_summary.csv")
    shake = ranked_config_means(ROOT / "experiments" / "cscan_shakespeare_jr0.1_summary.csv")
    tune_once = np.mean(
        [
            100 * score_for(r)
            for r in read_csv(ROOT / "experiments" / "phase_dropout_summary.csv")
            if r["dataset"] == "Cifar100_pathological"
            and r["algorithm"] == "Ditto"
            and "cdr0.0_jr0.2" in r["run_id"]
        ]
    )
    tuned_shakespeare = max(
        100 * score_for(r)
        for r in read_csv(ROOT / "experiments" / "shakespeare_sweep_selection.csv")
        if r["algorithm"] == "Ditto"
    )

    panels = [
        ("CIFAR pathological\n20% participation", cifar, tune_once, "tune-once Ditto reference"),
        ("Shakespeare\n10% participation", shake, tuned_shakespeare, "tuned Ditto reference"),
    ]

    panel_ranges = []
    for _, ranked, reference, _ in panels:
        ys = np.array([score for _, score, _ in ranked])
        yerr = np.array([sd for _, _, sd in ranked])
        ymin = min(float(np.min(ys - yerr)), reference)
        ymax = max(float(np.max(ys + yerr)), reference)
        raw_span = ymax - ymin
        panel_ranges.append((ymin, ymax, max(raw_span * 1.36, raw_span + 0.16)))
    shared_y_span = max(span for _, _, span in panel_ranges)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.15))
    for ax, (title, ranked, reference, reference_label), (ymin, ymax, _), panel in zip(axes, panels, panel_ranges, ["A", "B"]):
        xs = np.arange(1, len(ranked) + 1)
        ys = np.array([score for _, score, _ in ranked])
        yerr = np.array([sd for _, _, sd in ranked])
        default_rank = next(i + 1 for i, (name, _, _) in enumerate(ranked) if name == "default")
        default_score = next(score for name, score, _ in ranked if name == "default")

        ax.axhline(reference, color=PALETTE["muted"], linewidth=1.0, linestyle="--")
        ax.axhspan(float(np.min(ys)), float(np.max(ys)), color=PALETTE["ada"], alpha=0.06, linewidth=0)
        ax.plot(xs, ys, color=PALETTE["muted"], linewidth=0.9, alpha=0.8)
        ax.errorbar(xs, ys, yerr=yerr, fmt="o", markersize=3.0, color=PALETTE["ditto"], linewidth=0.8, capsize=2, zorder=3)
        ax.scatter([default_rank], [default_score], s=42, marker="D", facecolor="white", edgecolor=PALETTE["ada"], linewidth=1.2, zorder=4)
        spread = float(np.max(ys) - np.min(ys))
        ax.text(0.03, 0.88, f"spread {spread:.2f} pts", transform=ax.transAxes, color=PALETTE["muted"], fontsize=7.2)
        ax.set_title(title)
        ax.set_xlabel("controller configuration rank")
        ax.grid(axis="y")
        ax.set_xlim(0.5, len(xs) + 0.5)
        ymid = 0.5 * (ymin + ymax)
        ax.set_ylim(ymid - 0.5 * shared_y_span, ymid + 0.5 * shared_y_span)
        pad = 0.08 * shared_y_span
        reference_above = reference >= float(np.max(ys))
        label_y = reference + 0.22 * pad if reference_above else reference - 0.34 * pad
        ax.text(
            len(xs) - 0.1,
            label_y,
            reference_label,
            ha="right",
            va="bottom" if reference_above else "top",
            color=PALETTE["muted"],
            fontsize=7.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.7},
        )
        ax.text(-0.09, 1.07, panel, transform=ax.transAxes, weight="bold")
    axes[0].set_ylabel("personalized accuracy (%)")
    save(fig, "controller_sensitivity")


def save(fig: mpl.figure.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.5)
    for ext, dpi in [("pdf", None), ("png", 300)]:
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(path.relative_to(ROOT))
    plt.close(fig)


def main() -> None:
    setup_style()
    plot_lambda_stress_grid()
    plot_coefficient_brittleness()
    plot_lambda_participation_landscape()
    plot_participation_shift()
    compact_only = any(
        "run_dir" not in read_csv(path)[0]
        for path in (
            ROOT / "experiments" / "phase_dropout_summary.csv",
            ROOT / "experiments" / "shakespeare_core_summary.csv",
        )
    )
    if compact_only:
        print(
            "Skipping shift_response and controller_trajectories: "
            "compact summary tables do not contain machine-local run directories."
        )
    else:
        plot_shift_response()
        plot_controller_trajectories()
    plot_controller_sensitivity()


if __name__ == "__main__":
    main()
