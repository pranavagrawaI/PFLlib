#!/usr/bin/env python3
"""Recompute key paper aggregates from the archived per-run summary rows."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments"


def read(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def personalized(rows: list[dict[str, str]]) -> list[float]:
    return [float(row["personalized_accuracy"]) for row in rows]


def mean(values: list[float]) -> float:
    if not values:
        raise AssertionError("filter selected no result rows")
    return statistics.mean(values)


def close(label: str, observed: float, expected: float, tolerance: float = 5e-5) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(
            f"{label}: observed {observed:.6f}, expected {expected:.6f} "
            f"(tolerance {tolerance:.6f})"
        )


def phase1_optima() -> dict[str, tuple[str, float]]:
    rows = read("phase1_selection.csv")
    answer: dict[str, tuple[str, float]] = {}
    for dataset in ("Cifar100_practical", "Cifar100_pathological"):
        candidates = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["algorithm"] == "Ditto"
            and row["run_id"].startswith("lam_")
        ]
        best = max(candidates, key=lambda row: float(row["personalized_accuracy"]))
        answer[dataset] = (best["run_id"].removeprefix("lam_"), float(best["personalized_accuracy"]))
    return answer


def filtered_mean(
    filename: str,
    *,
    dataset: str,
    algorithm: str,
    run_fragment: str = "",
) -> float:
    rows = [
        row
        for row in read(filename)
        if row["dataset"] == dataset
        and row["algorithm"] == algorithm
        and run_fragment in row["run_id"]
    ]
    return mean(personalized(rows))


def config_mean_range(filename: str) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = {}
    for row in read(filename):
        config = row["run_id"].rsplit("_seed", 1)[0]
        grouped.setdefault(config, []).append(float(row["personalized_accuracy"]))
    means = {config: mean(values) for config, values in grouped.items()}
    if len(means) != 11:
        raise AssertionError(f"{filename}: expected 11 configurations, found {len(means)}")
    return min(means.values()), max(means.values()), means["default"]


def main() -> None:
    optima = phase1_optima()
    if optima["Cifar100_practical"][0] != "0.5":
        raise AssertionError(f"unexpected practical optimum: {optima['Cifar100_practical'][0]}")
    if optima["Cifar100_pathological"][0] not in {"0", "0.0"}:
        raise AssertionError(f"unexpected pathological optimum: {optima['Cifar100_pathological'][0]}")

    stale = filtered_mean(
        "phase_dropout_summary.csv",
        dataset="Cifar100_pathological",
        algorithm="Ditto",
        run_fragment="cdr0.0_jr0.2",
    )
    adaptive = filtered_mean(
        "phase_dropout_summary.csv",
        dataset="Cifar100_pathological",
        algorithm="AdaProxDitto",
        run_fragment="cdr0.0_jr0.2",
    )
    close("pathological 20% frozen Ditto", stale, 0.5511718316)
    close("pathological 20% AdaDitto", adaptive, 0.5692102632)

    oracle = filtered_mean(
        "core_participation_summary.csv",
        dataset="Cifar100_pathological",
        algorithm="Ditto",
        run_fragment="jr0.2",
    )
    adaptive_oracle_comparison = filtered_mean(
        "core_participation_summary.csv",
        dataset="Cifar100_pathological",
        algorithm="AdaProxDitto",
        run_fragment="jr0.2",
    )
    close("pathological 20% retuned Ditto", oracle, 0.5865111629)
    close("pathological 20% AdaDitto core", adaptive_oracle_comparison, 0.5720759747)

    shake_ditto = filtered_mean(
        "shakespeare_core_summary.csv",
        dataset="Shakespeare",
        algorithm="Ditto",
    )
    shake_adaptive = filtered_mean(
        "shakespeare_core_summary.csv",
        dataset="Shakespeare",
        algorithm="AdaProxDitto",
    )
    close("Shakespeare tuned Ditto", shake_ditto, 0.4821547614)
    close("Shakespeare AdaDitto", shake_adaptive, 0.4820352261)

    cifar_scan = config_mean_range("cscan_cifar_path_jr0.2_summary.csv")
    shake_scan = config_mean_range("cscan_shakespeare_jr0.1_summary.csv")
    close("CIFAR controller-scan minimum", cifar_scan[0], 0.5627013218)
    close("CIFAR controller-scan maximum", cifar_scan[1], 0.5742752416)
    close("Shakespeare controller-scan minimum", shake_scan[0], 0.4818743132)
    close("Shakespeare controller-scan maximum", shake_scan[1], 0.4821593589)

    print("Verified archived aggregates")
    print(f"  CIFAR full-participation optima: practical lambda={optima['Cifar100_practical'][0]}, "
          f"pathological lambda={optima['Cifar100_pathological'][0]}")
    print(f"  Pathological 20% tune-once: Ditto={stale:.4f}, AdaDitto={adaptive:.4f}, "
          f"delta={100 * (adaptive - stale):+.2f} points")
    print(f"  Pathological 20% per-condition: Ditto={oracle:.4f}, "
          f"AdaDitto={adaptive_oracle_comparison:.4f}")
    print(f"  Shakespeare 10%: Ditto={shake_ditto:.4f}, AdaDitto={shake_adaptive:.4f}")
    print(f"  Controller-scan ranges: CIFAR={cifar_scan[0]:.4f}-{cifar_scan[1]:.4f}, "
          f"Shakespeare={shake_scan[0]:.4f}-{shake_scan[1]:.4f}")


if __name__ == "__main__":
    main()
