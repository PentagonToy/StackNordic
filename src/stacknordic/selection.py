"""Shared physical-time selection policy."""

from __future__ import annotations

import math

TimeSelection = list[float] | tuple[float, float] | None


def select_times(available: tuple[float, ...], times: TimeSelection) -> tuple[float, ...]:
    """Select all times, an inclusive tuple range, or exact list members."""

    if times is None:
        return available
    if isinstance(times, tuple):
        if len(times) != 2:
            raise ValueError("a time range must be a two-value tuple")
        requested = tuple(float(value) for value in times)
        start, end = requested
        if not all(math.isfinite(value) for value in requested):
            raise ValueError("times must contain finite physical times")
        if start > end:
            raise ValueError("time range start must not exceed its end")
        selected = tuple(value for value in available if start <= value <= end)
    elif isinstance(times, list):
        requested = tuple(float(value) for value in times)
        if not requested or any(not math.isfinite(value) for value in requested):
            raise ValueError("times must contain finite physical times")
        selected = tuple(
            value
            for value in available
            if any(
                math.isclose(value, item, rel_tol=1.0e-12, abs_tol=1.0e-14)
                for item in requested
            )
        )
        missing = [
            item
            for item in requested
            if not any(
                math.isclose(value, item, rel_tol=1.0e-12, abs_tol=1.0e-14)
                for value in available
            )
        ]
        if missing:
            values = ", ".join(f"{value:g}" for value in missing)
            raise KeyError(f"requested times are incomplete or unavailable: {values}")
    else:
        raise TypeError("times must be None, an exact list, or a two-value range tuple")
    if not selected:
        raise ValueError("times selected no complete processor results")
    return selected
