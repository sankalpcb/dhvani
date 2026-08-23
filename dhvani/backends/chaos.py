"""Deterministic fault injection for the async escalation path.

Spec §10.2: the chaos layer injects timeouts, 429s, 500s, partial batches,
duplicate deliveries and reordering, and the invariant suite asserts the
system still loses nothing and converges.

Faults are driven by an explicit seed rather than real randomness, so a
failing chaos test reproduces exactly. Nothing here talks to a network.
"""

import random

FAULTS = ("timeout", "rate_limit", "server_error", "partial", "duplicate", "reorder")


class TransientError(RuntimeError):
    """An injected failure that a caller is expected to survive and retry."""


class ChaosBackend:
    def __init__(self, inner, faults=(), seed: int = 0):
        unknown = [f for f in faults if f not in FAULTS]
        if unknown:
            raise ValueError(f"unknown fault(s): {unknown}; expected one of {FAULTS}")
        self.inner = inner
        self.name = inner.name
        self.variant_key = inner.variant_key
        self.faults = tuple(faults)
        self.seed = seed

    def cost_per_call(self, segment) -> float:
        return self.inner.cost_per_call(segment)

    def submit(self, segments: list) -> str:
        return self.inner.submit(segments)

    def poll(self, job_id: str):
        if "timeout" in self.faults:
            raise TransientError("injected timeout while polling")
        if "rate_limit" in self.faults:
            raise TransientError("injected 429 rate limit")
        if "server_error" in self.faults:
            raise TransientError("injected 500 server error")

        results = self.inner.poll(job_id)
        if results is None:
            return None

        rng = random.Random(f"{self.seed}:{job_id}")
        items = sorted(results.items())

        if "partial" in self.faults and len(items) > 1:
            keep = max(1, len(items) // 2)
            items = items[:keep]

        if "reorder" in self.faults:
            rng.shuffle(items)

        # "duplicate" models the same payload being delivered twice. Because
        # results are keyed by segment_id, a duplicate collapses into the same
        # dict -- which is exactly the property the merge relies on.
        if "duplicate" in self.faults:
            items = items + items

        return dict(items)
