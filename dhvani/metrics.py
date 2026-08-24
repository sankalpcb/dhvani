"""Timing and throughput instrumentation (spec §9.1).

Deliberately dependency-free and deterministic in shape: percentile() uses
the nearest-rank method over a sorted copy (rank = ceil(p * n), 1-indexed),
so the same samples always give the same summary. Timing values themselves
vary run to run, which is why no test asserts a specific duration.
"""

import math
import time


class Timer:
    """Context manager measuring wall-clock milliseconds."""

    def __init__(self):
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        return False


def percentile(values, p: float) -> float:
    """Nearest-rank percentile. p is a fraction in [0, 1].

    rank = ceil(p * n), 1-indexed into the sorted copy, clamped to
    [1, n]. This is a deterministic order statistic (no interpolation
    between neighboring samples), which is what pins percentile(range(1,
    101), 0.5) == 50.0 rather than 50.5.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be between 0 and 1, got {p}")
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return float(ordered[0])
    rank = max(1, min(n, math.ceil(p * n)))
    return float(ordered[rank - 1])


def summarize(samples: dict) -> dict:
    """Per-series count, p50, p99 and total."""
    return {
        name: {
            "count": len(series),
            "p50": percentile(series, 0.50),
            "p99": percentile(series, 0.99),
            "total_ms": float(sum(series)),
        }
        for name, series in samples.items()
    }


def throughput(audio_ms: int, wall_ms: float) -> float:
    """Audio-hours processed per wall-clock hour. Zero when no time elapsed."""
    if wall_ms <= 0.0:
        return 0.0
    return float(audio_ms) / float(wall_ms)
