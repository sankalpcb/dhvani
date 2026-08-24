"""Asynchronous backend protocol.

Spec §7: Chirp dynamic batch has up to 24h turnaround, so escalation cannot
be a blocking call. submit() registers a batch and returns a job id; poll()
returns None while the batch is outstanding and a {segment_id: result} dict
once it completes.

Job ids are derived from the submitted segment ids, not generated randomly.
That makes them deterministic (invariant I5) and makes re-submitting the same
batch a no-op at the Store level rather than a duplicate job.

SyncAsyncAdapter wraps any synchronous Backend as an AsyncBackend so the
whole async machinery is testable with no cloud, no network, and no
credentials (goal G5). pending_polls simulates turnaround latency.
"""

import hashlib
from typing import Protocol, runtime_checkable


class JobNotFound(RuntimeError):
    """poll() was given a job id this backend never issued."""


@runtime_checkable
class AsyncBackend(Protocol):
    name: str
    variant_key: str

    def cost_per_call(self, segment) -> float: ...

    def submit(self, segments: list) -> str: ...

    def poll(self, job_id: str) -> dict | None: ...


def job_id_for(variant_key: str, segments) -> str:
    """Content-derived job id: same variant plus same segments -> same id."""
    digest = hashlib.sha256(variant_key.encode("utf-8"))
    digest.update(b"\x00")  # Delimiter to prevent variant_key/segment_id boundary ambiguity
    for segment_id in sorted(s.segment_id for s in segments):
        digest.update(segment_id.encode("utf-8"))
        digest.update(b"\x00")  # Delimiter between segment ids
    return digest.hexdigest()[:32]


class SyncAsyncAdapter:
    """Presents a synchronous Backend through the AsyncBackend protocol."""

    def __init__(self, inner, pending_polls: int = 0):
        self.inner = inner
        self.name = inner.name
        self.variant_key = inner.variant_key
        self.pending_polls = pending_polls
        self._jobs: dict[str, list] = {}
        self._polls: dict[str, int] = {}
        self._results: dict[str, dict] = {}  # Cache for completed job results

    def cost_per_call(self, segment) -> float:
        return self.inner.cost_per_call(segment)

    def submit(self, segments: list) -> str:
        job_id = job_id_for(self.variant_key, segments)
        self._jobs[job_id] = list(segments)
        self._polls.setdefault(job_id, 0)
        return job_id

    def adopt(self, job_id: str, segments: list) -> None:
        """Register a job this adapter instance never submitted.

        jobs rows in the Store are durable, but _jobs is in-memory and
        rebuilt empty by every new process. A reconcile-only run (no
        --escalate in the same process) constructs a fresh adapter that
        never called submit(), so poll() would raise JobNotFound for a job
        that is genuinely outstanding. adopt() closes that gap by
        rebuilding the segment list from the durable record -- without
        re-submitting (which would be a fresh registration, not a resume)
        and without resetting an in-progress poll counter.
        """
        self._jobs[job_id] = list(segments)
        self._polls.setdefault(job_id, 0)

    def poll(self, job_id: str) -> dict | None:
        if job_id not in self._jobs:
            raise JobNotFound(f"unknown job id: {job_id}")
        if self._polls[job_id] < self.pending_polls:
            self._polls[job_id] += 1
            return None
        # Return cached results if already computed; compute and cache otherwise
        if job_id not in self._results:
            self._results[job_id] = {s.segment_id: self.inner.transcribe(s) for s in self._jobs[job_id]}
        return self._results[job_id]
