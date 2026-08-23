"""Spec §10.3 invariants for the async escalation path.

I1  no loss          -- every input segment appears exactly once in the track
I2  idempotent merge -- applying a batch result twice is a no-op
I6  convergence      -- once all jobs settle, the async track equals what a
                        fully synchronous pipeline would have produced
"""

import numpy as np
import pytest

from dhvani.backends.async_base import SyncAsyncAdapter
from dhvani.backends.chaos import ChaosBackend, TransientError
from dhvani.config import POLICY_ID
from dhvani.escalate import escalate
from dhvani.pipeline import TrackEntry
from dhvani.reconcile import reconcile
from dhvani.segmenter import Segment
from dhvani.store import Store
from dhvani.track import entries_from_json, entries_to_json


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def cost_per_call(self, segment):
        return 0.00075

    def transcribe(self, segment):
        return {"text": f"fixed-{segment.segment_id[:4]}", "signals": {}}


N = 8
ENTRIES = [TrackEntry(chr(97 + i) * 64, i * 3000, (i + 1) * 3000,
                      "raw", 0.65, "review") for i in range(N)]
SEGMENTS = {e.segment_id: Segment(e.segment_id, e.t_start_ms, e.t_end_ms,
                                  np.zeros(10, dtype=np.int16))
            for e in ENTRIES}
TABLE = {"tier1": {"0.6-0.7": 18.0}}


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        s.put_track("vid1", 1, POLICY_ID, entries_to_json(ENTRIES), 0.0)
        yield s


def _drain(backend, store, max_passes=20):
    """Reconcile until this backend has no open jobs left, surviving
    injected transients.

    NOT "stop at the first pass that makes no progress": an async backend
    with startup latency (SyncAsyncAdapter's pending_polls) legitimately
    returns nothing on its first poll(s) while a job is still outstanding,
    before it has ever had a chance to deliver a result. The original
    version of this helper broke out of the loop the moment one pass
    produced no update, which stopped draining test_i6's pending_polls=2
    backend after its very first (still-warming-up) pass -- one pass short
    of it ever being polled for real -- and made the test compare a track
    that was never actually driven to completion. Draining until this
    backend's own jobs are no longer open (done/failed, or simply absent)
    is the correct stopping condition; polling an already-settled job is a
    documented memoized no-op (see async_base.SyncAsyncAdapter.poll), so
    it is safe to keep calling reconcile() up to max_passes regardless.
    """
    version = store.latest_track_version("vid1")
    for _ in range(max_passes):
        try:
            version = reconcile("vid1", backend, store)
        except TransientError:
            continue
        if not any(
            job["tier"] == backend.name and job["variant_key"] == backend.variant_key
            for job in store.open_jobs()
        ):
            break
    return version


def _track(store, version):
    return entries_from_json(store.get_track("vid1", version)["content_json"])


@pytest.mark.parametrize("faults", [
    (), ("partial",), ("duplicate",), ("reorder",),
    ("partial", "reorder"), ("duplicate", "reorder"),
])
def test_i1_no_segment_is_ever_lost_or_duplicated(store, faults):
    backend = ChaosBackend(SyncAsyncAdapter(StubSync()), faults=faults, seed=11)
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    entries = _track(store, _drain(backend, store))
    ids = [e.segment_id for e in entries]
    assert len(ids) == N
    assert len(set(ids)) == N
    assert sorted(ids) == sorted(SEGMENTS)


@pytest.mark.parametrize("faults", [("timeout",), ("rate_limit",), ("server_error",)])
def test_i1_holds_when_every_poll_fails(store, faults):
    """A backend that never succeeds must leave the track intact, not corrupt."""
    backend = ChaosBackend(SyncAsyncAdapter(StubSync()), faults=faults, seed=5)
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    entries = _track(store, _drain(backend, store))
    assert [e.text for e in entries] == ["raw"] * N


def test_i2_reconciling_a_settled_job_repeatedly_is_a_no_op(store):
    backend = SyncAsyncAdapter(StubSync())
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version = _drain(backend, store)
    before = store.get_track("vid1", version)["content_json"]
    for _ in range(5):
        assert reconcile("vid1", backend, store) == version
    assert store.get_track("vid1", version)["content_json"] == before


def test_i6_async_converges_to_the_synchronous_result(store, tmp_path):
    """The headline property: once everything settles, async == sync.

    History (Task 7, fix round 1): this test originally failed. ChaosBackend's
    "partial" fault used to truncate a job's results to the same first-half
    subset on EVERY poll of that job, forever, with nothing dependent on
    attempt count -- that models a permanently broken job, not a transient
    partial delivery, so the excluded half was never deliverable by any
    number of reconcile() passes and the job stayed "running" forever. See
    dhvani/backends/chaos.py's module docstring ("Fix round 1") for the
    resolution: "partial" now truncates only the FIRST time a job would
    deliver results and returns the full set on every poll after that, so
    a job it touches still converges under sustained polling. This test
    exercises exactly that combination (partial + reorder + duplicate,
    with pending_polls startup latency) and asserts real convergence with
    no resubmission needed -- see task-7-report.md's "Fix round 1" section
    for the full trace of the original failure and the fix.
    """
    backend = ChaosBackend(SyncAsyncAdapter(StubSync(), pending_polls=2),
                           faults=("partial", "reorder", "duplicate"), seed=3)
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    async_entries = _track(store, _drain(backend, store, max_passes=60))

    with Store(str(tmp_path / "sync.db")) as sync_store:
        sync_store.put_track("vid1", 1, POLICY_ID, entries_to_json(ENTRIES), 0.0)
        plain = SyncAsyncAdapter(StubSync())
        escalate(ENTRIES, SEGMENTS, plain, sync_store, TABLE, 10.0)
        sync_entries = _track(sync_store, _drain(plain, sync_store))

    assert async_entries == sync_entries


def test_i6_convergence_is_reachable_from_a_partial_start(store):
    """Partial delivery must not strand segments permanently un-escalated."""
    backend = ChaosBackend(SyncAsyncAdapter(StubSync()), faults=("partial",), seed=2)
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    _drain(backend, store)

    clean = SyncAsyncAdapter(StubSync())
    escalate(ENTRIES, SEGMENTS, clean, store, TABLE, 10.0)
    entries = _track(store, _drain(clean, store))
    assert all(e.text.startswith("fixed-") for e in entries)


class PermanentlyPartialBackend:
    """An AsyncBackend whose poll() always returns only some of a job's
    registered segments -- on every single call, forever. Built directly
    against the AsyncBackend protocol (submit/poll/cost_per_call), NOT via
    ChaosBackend, so this test does not depend on ChaosBackend's "partial"
    fault at all.

    This pins the genuinely permanent failure mode that ChaosBackend's
    "partial" fault used to model by accident before Task 7 fix round 1
    (see dhvani/backends/chaos.py's module docstring): a batch job that
    never, ever finishes delivering. Real systems can get stuck exactly
    like this -- a worker that crashed mid-batch, a queue partition that
    lost the rest of the payload. No amount of reconcile() polling can
    make such a job settle; what must still hold is I1: nothing already
    in the track is ever lost or duplicated, the job is never falsely
    marked done, and the segments that never arrive keep exactly the text
    they had before escalation was ever attempted, rather than being
    blanked or corrupted.
    """

    name = "tier1"
    variant_key = "tier1|hi-IN"

    def __init__(self, inner):
        self.inner = inner

    def cost_per_call(self, segment):
        return self.inner.cost_per_call(segment)

    def submit(self, segments):
        return self.inner.submit(segments)

    def poll(self, job_id):
        full = self.inner.poll(job_id)
        if full is None:
            return None
        items = sorted(full.items())
        keep = max(1, len(items) // 2)
        return dict(items[:keep])


def test_i1_holds_under_a_permanently_partial_backend(store):
    """The permanent-failure counterpart to test_i6_convergence_is_reachable_
    from_a_partial_start: when a job's delivery is not merely delayed but
    genuinely never completes, the system must still lose nothing (I1),
    must never claim a job settled that did not (reconcile.py's
    done-only-when-complete rule), and must leave the segments it could
    not escalate exactly as they were, not blanked.
    """
    backend = PermanentlyPartialBackend(SyncAsyncAdapter(StubSync()))
    job_id = escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version = _drain(backend, store, max_passes=20)
    entries = _track(store, version)

    # I1: every segment appears exactly once, nothing lost or duplicated.
    ids = [e.segment_id for e in entries]
    assert len(ids) == N
    assert len(set(ids)) == N
    assert sorted(ids) == sorted(SEGMENTS)

    # The job can never cover all its registered segment_ids, so it must
    # never be falsely marked done -- it stays "running" and still shows
    # up in open_jobs() for a future retry (or resubmission) to find.
    job = store.get_job(job_id)
    assert job["state"] == "running"
    assert job_id in {j["job_id"] for j in store.open_jobs()}

    # The segments that were never delivered keep their original Tier 0
    # text/risk/band untouched -- degrade safely, don't blank or corrupt.
    delivered = [e for e in entries if e.text != "raw"]
    undelivered = [e for e in entries if e.text == "raw"]
    assert delivered and undelivered
    assert all(e.risk == 0.65 and e.band == "review" for e in undelivered)
    assert all(e.text.startswith("fixed-") for e in delivered)
