"""Deterministic fault injection for the async escalation path.

Spec §10.2: the chaos layer injects timeouts, 429s, 500s, partial batches,
duplicate deliveries and reordering, and the invariant suite asserts the
system still loses nothing and converges.

Faults are driven by an explicit seed rather than real randomness, so a
failing chaos test reproduces exactly. Nothing here talks to a network.

Fix round 1 (Task 7 finding): "partial" used to truncate a job's results
to the same first-half subset on EVERY poll of that job, forever -- that
does not model a partial delivery, it models a permanently broken batch,
indistinguishable from a stuck job. Spec §10.2 lists it alongside
timeouts and 429s, which are transient: a caller is expected to retry and
eventually get through. "partial" now heals after its first delivery
attempt for a given job_id: the FIRST time a job would return results,
it is truncated as before; every poll of that same job after that
returns the full result set. This is tracked in `_partial_delivered`, an
instance dict keyed by job_id -- the same pattern SyncAsyncAdapter uses
for `_polls` -- so it stays deterministic (no wall-clock, no unseeded
randomness) and reproduces exactly for a given seed.
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
        # job_ids that have already had one "partial" truncation applied.
        # "partial" only ever truncates a job once -- see module docstring,
        # Fix round 1 -- so this is how it remembers to heal on the next poll.
        self._partial_delivered: set[str] = set()

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

        # Truncate only the first time this job would deliver results. A
        # job already marked as having received its one partial delivery
        # returns the full set from here on -- "partial" models a batch
        # that comes back incomplete once and completes on retry, not a
        # permanently broken job (see module docstring, Fix round 1).
        if ("partial" in self.faults and len(items) > 1
                and job_id not in self._partial_delivered):
            keep = max(1, len(items) // 2)
            items = items[:keep]
            self._partial_delivered.add(job_id)

        if "reorder" in self.faults:
            rng.shuffle(items)

        # "duplicate" models the same payload being delivered twice. Because
        # results are keyed by segment_id, a duplicate collapses into the same
        # dict -- which is exactly the property the merge relies on.
        if "duplicate" in self.faults:
            items = items + items

        return dict(items)
