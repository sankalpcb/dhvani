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

    KNOWN FAILING -- see task-7-report.md for the full writeup. Root cause
    is in ChaosBackend.poll() (dhvani/backends/chaos.py), not in this test
    or in reconcile()/escalate()/track.py: the "partial" fault truncates a
    job's results to `sorted(items)[:len(items)//2]` on every single poll
    of that job, with no dependence on attempt count or any other evolving
    state, so a job it ever truncates is truncated identically forever.
    Combined with "duplicate"/"reorder" (which only reorder/repeat the
    already-truncated items, dict-collapsed on return, so they cannot
    reintroduce what "partial" dropped), the excluded half of the batch
    is never delivered by ANY number of reconcile() passes against this
    same job -- the job is left "running" forever, exactly as designed
    (reconcile.py only marks a job "done" once its results cover every
    segment_id it registered, spec I1). That correctly protects I1 (no
    entry is ever lost), but it means passive draining alone can never
    reach I6 for a job "partial" has touched, contradicting chaos.py's
    own module docstring ("the invariant suite asserts the system still
    loses nothing and converges"). test_i6_convergence_is_reachable_from_
    a_partial_start shows the actual recovery path: a caller must
    periodically re-run escalate() for still-outstanding segments, not
    just keep polling the same stuck job. This test intentionally keeps
    the original assertion (no resubmission) to pin the gap rather than
    hide it -- do not weaken it to make the suite green.
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
