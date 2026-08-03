#!/usr/bin/env python3
"""Verify the independent controller spec and archived production traces."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from reference_controller import ControllerConfig, controller_step, update_reference


ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "experiments" / "traces"
CONFIG = ControllerConfig()
TOLERANCE = 2e-9


def close(label: str, observed: float, expected: float, tolerance: float = TOLERANCE) -> None:
    if not math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: observed={observed!r}, expected={expected!r}")


def verify_known_vectors() -> None:
    warmup = controller_step(
        CONFIG, previous=0.05, probe_loss=2.0, reference_loss=1.0, global_round=0
    )
    close("warmup final", warmup.final, 0.05)
    if not warmup.warmup_active:
        raise AssertionError("round zero must be warmup")

    positive = controller_step(
        CONFIG, previous=0.05, probe_loss=2.0, reference_loss=1.0, global_round=2
    )
    expected_gap = math.log((2.0 + 1e-8) / (1.0 + 1e-8))
    close("positive raw gap", positive.raw_gap or 0.0, expected_gap)
    close("positive target", positive.target, 0.08 + 0.8 * expected_gap)
    close("positive smoothed", positive.smoothed, 0.7 * 0.05 + 0.3 * positive.target)

    clipped = controller_step(
        CONFIG, previous=0.05, probe_loss=100.0, reference_loss=1.0, global_round=2
    )
    close("clipped gap", clipped.clipped_gap, 0.7)
    close("clipped target", clipped.target, 0.64)

    negative = controller_step(
        CONFIG, previous=0.05, probe_loss=0.5, reference_loss=1.0, global_round=2
    )
    close("negative gap", negative.clipped_gap, 0.0)
    close("zero-gap transition", negative.final, 0.059)

    missing = controller_step(
        CONFIG, previous=0.05, probe_loss=2.0, reference_loss=None, global_round=2
    )
    close("missing reference", missing.final, 0.059)

    bounded_config = ControllerConfig(maximum=0.1, gain=100.0)
    bounded = controller_step(
        bounded_config,
        previous=0.05,
        probe_loss=100.0,
        reference_loss=1.0,
        global_round=2,
    )
    close("upper bound", bounded.final, 0.1)

    first_reference = update_reference(CONFIG, None, [3.0, 1.0, 2.0])
    close("first reference", first_reference or 0.0, 2.0)
    second_reference = update_reference(CONFIG, first_reference, [6.0, 4.0, 5.0])
    close("EMA reference", second_reference or 0.0, 2.3)


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_trace(path: Path) -> tuple[int, int]:
    rows = read_trace(path)
    if not rows:
        raise AssertionError(f"{path}: empty trace")
    rows.sort(key=lambda row: (int(row["round"]), int(row["client_id"])))

    previous_by_client: dict[int, float] = {}
    by_round: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_round.setdefault(int(row["round"]), []).append(row)

    reference: float | None = None
    for round_index, round_rows in sorted(by_round.items()):
        logged_references = {float(row["Lg"]) for row in round_rows}
        if len(logged_references) != 1:
            raise AssertionError(f"{path}: round {round_index} has inconsistent Lg")
        logged_reference = logged_references.pop()
        close(
            f"{path}: round {round_index} reference",
            logged_reference,
            0.0 if reference is None else reference,
            tolerance=2e-8,
        )

        for row in round_rows:
            client_id = int(row["client_id"])
            expected_previous = previous_by_client.get(client_id, CONFIG.minimum)
            close(
                f"{path}: previous coefficient r{round_index} c{client_id}",
                float(row["coefficient_prev"]),
                expected_previous,
            )
            step = controller_step(
                CONFIG,
                previous=expected_previous,
                probe_loss=float(row["Li"]),
                reference_loss=reference,
                global_round=round_index,
            )
            close("raw gap", float(row["client_loss_gap_raw"]), step.raw_gap or 0.0)
            close("clipped gap", float(row["client_loss_gap_clipped"]), step.clipped_gap)
            close("target", float(row["coefficient_target"]), step.target)
            close("smoothed", float(row["coefficient_smoothed"]), step.smoothed)
            close("bounded", float(row["coefficient_bounded"]), step.bounded)
            close("final", float(row["coefficient_final"]), step.final)
            if int(row["warmup_active"]) != int(step.warmup_active):
                raise AssertionError(f"{path}: warmup flag mismatch")
            previous_by_client[client_id] = step.final

        reference = update_reference(
            CONFIG,
            reference,
            [float(row["Li"]) for row in round_rows],
        )

    return len(rows), len(by_round)


def main() -> None:
    verify_known_vectors()
    trace_paths = sorted(TRACES.rglob("client_controller.csv"))
    if len(trace_paths) != 7:
        raise AssertionError(f"expected seven archived controller traces, found {len(trace_paths)}")
    algorithms = {
        read_trace(path)[0]["algorithm"]
        for path in trace_paths
    }
    if algorithms != {"AdaProxDitto", "AdaProxFedProx"}:
        raise AssertionError(f"expected both adaptive integrations, found {sorted(algorithms)}")
    total_rows = 0
    summaries = []
    for path in trace_paths:
        row_count, round_count = verify_trace(path)
        total_rows += row_count
        summaries.append((path.relative_to(ROOT), row_count, round_count))

    print("Verified independent controller specification")
    print("  Known-vector and edge-case checks: passed")
    print(f"  Production traces: {len(summaries)} files, {total_rows} client-round rows")
    for path, row_count, round_count in summaries:
        print(f"    {path}: {row_count} rows across {round_count} rounds")


if __name__ == "__main__":
    main()
