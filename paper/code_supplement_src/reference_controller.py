#!/usr/bin/env python3
"""Dependency-free reference controller specified in IMPLEMENTATION_SPEC.md."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Iterable


@dataclass(frozen=True)
class ControllerConfig:
    base: float = 0.08
    minimum: float = 0.05
    maximum: float = 3.0
    gain: float = 0.8
    gap_clip: float = 0.7
    smoothing: float = 0.3
    reference_retention: float = 0.9
    epsilon: float = 1e-8
    warmup_rounds: int = 2

    def validate(self) -> None:
        if not 0.0 <= self.smoothing <= 1.0:
            raise ValueError("smoothing must be in [0, 1]")
        if not 0.0 <= self.reference_retention <= 1.0:
            raise ValueError("reference_retention must be in [0, 1]")
        if not 0.0 <= self.minimum <= self.maximum:
            raise ValueError("coefficient bounds are invalid")
        if self.gain < 0.0 or self.gap_clip < 0.0:
            raise ValueError("gain and gap_clip must be nonnegative")
        if self.epsilon <= 0.0 or self.warmup_rounds < 0:
            raise ValueError("epsilon must be positive and warmup nonnegative")


@dataclass(frozen=True)
class ControllerStep:
    raw_gap: float | None
    clipped_gap: float
    previous: float
    target: float
    smoothed: float
    bounded: float
    final: float
    warmup_active: bool


def controller_step(
    config: ControllerConfig,
    *,
    previous: float,
    probe_loss: float,
    reference_loss: float | None,
    global_round: int,
) -> ControllerStep:
    config.validate()
    if not math.isfinite(probe_loss) or probe_loss < 0.0:
        raw_gap = None
        clipped_gap = 0.0
    elif reference_loss is None or not math.isfinite(reference_loss):
        raw_gap = None
        clipped_gap = 0.0
    elif reference_loss < 0.0:
        raise ValueError("reference_loss must be nonnegative")
    else:
        raw_gap = math.log(
            (probe_loss + config.epsilon) / (reference_loss + config.epsilon)
        )
        clipped_gap = min(max(raw_gap, 0.0), config.gap_clip)

    target = config.base + config.gain * clipped_gap
    smoothed = (1.0 - config.smoothing) * previous + config.smoothing * target
    bounded = min(max(smoothed, config.minimum), config.maximum)
    warmup_active = global_round < config.warmup_rounds
    final = config.minimum if warmup_active else bounded
    return ControllerStep(
        raw_gap=raw_gap,
        clipped_gap=clipped_gap,
        previous=previous,
        target=target,
        smoothed=smoothed,
        bounded=bounded,
        final=final,
        warmup_active=warmup_active,
    )


def update_reference(
    config: ControllerConfig,
    previous_reference: float | None,
    probe_losses: Iterable[float],
) -> float | None:
    config.validate()
    losses = [float(value) for value in probe_losses]
    if not losses:
        return previous_reference
    if any(not math.isfinite(value) or value < 0.0 for value in losses):
        raise ValueError("probe losses must be nonnegative and finite")
    round_reference = float(statistics.median(losses))
    if previous_reference is None:
        return round_reference
    return (
        config.reference_retention * previous_reference
        + (1.0 - config.reference_retention) * round_reference
    )
