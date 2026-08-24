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
from dhvani.scorer import extract, risk as compute_risk
from dhvani.segmenter import Segment
from dhvani.store import Store
from dhvani.track import entries_from_json, entries_to_json, merge_entries


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
    injected transients. Returns (version, converged).

    `converged` is True only if the loop actually reached a state where this
    backend has no open jobs left -- i.e. the drain finished because the
    system settled, not because max_passes ran out. Reporting it is what
    stops "never converged" from being indistinguishable from "converged":
    the helper used to return quietly in both cases, so every caller that
    believed it was asserting convergence was really asserting nothing about
    it at all. Callers that expect convergence must now assert it, and the
    ones that legitimately never converge (the transient-fault backends,
    PermanentlyPartialBackend) must assert `not converged` explicitly.

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
    converged = False
    for _ in range(max_passes):
        try:
            version = reconcile("vid1", backend, store)
        except TransientError:
            continue
        if not any(
            job["tier"] == backend.name and job["variant_key"] == backend.variant_key
            for job in store.open_jobs()
        ):
            converged = True
            break
    return version, converged


def _track(store, version):
    return entries_from_json(store.get_track("vid1", version)["content_json"])


@pytest.mark.parametrize("faults", [
    (), ("partial",), ("duplicate",), ("reorder",),
    ("partial", "reorder"), ("duplicate", "reorder"),
])
def test_i1_no_segment_is_ever_lost_or_duplicated(store, faults):
    backend = ChaosBackend(SyncAsyncAdapter(StubSync()), faults=faults, seed=11)
    escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version, converged = _drain(backend, store)
    assert converged
    entries = _track(store, version)
    ids = [e.segment_id for e in entries]
    assert len(ids) == N
    assert len(set(ids)) == N
    assert sorted(ids) == sorted(SEGMENTS)
    # Identity alone is not I1. A reconciler that polls and then throws every
    # result away keeps all N ids present and distinct -- the Tier 0 entries
    # are simply still sitting there -- so the checks above pass while nothing
    # was ever merged. Every one of these faults is survivable, so once the
    # drain converges each segment must actually carry its escalated text.
    assert [e.text for e in entries] == [
        f"fixed-{e.segment_id[:4]}" for e in entries
    ]


@pytest.mark.parametrize("faults", [("timeout",), ("rate_limit",), ("server_error",)])
def test_i1_holds_when_every_poll_fails(store, faults):
    """A backend that never succeeds must leave the track intact, not corrupt.

    Adjusted for I6 (dead-lettering, fix round: reconcile per-job isolation):
    these faults raise on EVERY poll, so before dead-lettering existed the
    job could never settle and this asserted `not converged` -- _drain()
    ran out of passes with the job stuck "running" forever. reconcile() now
    catches a raising poll() per job (instead of letting it propagate and
    abandon the whole pass) and, once bump_job_attempts() exceeds
    MAX_JOB_ATTEMPTS, dead-letters the job: state -> "failed", which drops
    it out of open_jobs(). That satisfies _drain()'s convergence check (no
    more open jobs for this backend), so `converged` is now True -- but via
    permanent failure, not success. Assert the job actually reached
    "failed" (not "done") and that the track was never touched, since no
    result was ever merged.
    """
    backend = ChaosBackend(SyncAsyncAdapter(StubSync()), faults=faults, seed=5)
    job_id = escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version, converged = _drain(backend, store)
    assert converged
    assert store.get_job(job_id)["state"] == "failed"
    entries = _track(store, version)
    assert [e.text for e in entries] == ["raw"] * N


def test_i2_reconciling_a_settled_job_repeatedly_is_a_no_op(store):
    backend = SyncAsyncAdapter(StubSync())
    escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version, converged = _drain(backend, store)
    assert converged
    before = store.get_track("vid1", version)["content_json"]
    for _ in range(5):
        assert reconcile("vid1", backend, store) == version
    assert store.get_track("vid1", version)["content_json"] == before


def test_i6_async_converges_to_the_synchronous_result(store):
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

    The expected side is built HERE, by hand, and never passes through
    reconcile(). It used to be produced by a second escalate() + _drain()
    with chaos switched off, which made the test circular: both sides of
    `async_entries == sync_entries` flowed through the same reconcile() and
    merge_entries(), so any bug in that shared path cancelled out on both
    sides and the equality still held. A reconciler that discarded every
    polled result left both tracks at their untouched Tier 0 state and this
    test passed. The ground truth below calls StubSync.transcribe() directly
    and applies merge_entries() exactly once, mirroring reconcile()'s risk
    computation (scorer.extract() over the returned text and the segment's
    duration, then scorer.risk()) so the comparison is like-for-like.
    """
    backend = ChaosBackend(SyncAsyncAdapter(StubSync(), pending_polls=2),
                           faults=("partial", "reorder", "duplicate"), seed=3)
    escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version, converged = _drain(backend, store, max_passes=60)
    assert converged
    async_entries = _track(store, version)

    stub = StubSync()
    updates = {}
    for entry in ENTRIES:
        result = stub.transcribe(SEGMENTS[entry.segment_id])
        duration = entry.t_end_ms - entry.t_start_ms
        features = extract(result["text"], result.get("signals", {}), duration)
        updates[entry.segment_id] = {
            "text": result["text"],
            "risk": compute_risk(features),
        }
    sync_entries = merge_entries(ENTRIES, updates)

    # Guard against the whole comparison silently degenerating into
    # "unescalated == unescalated": the expected track must really differ
    # from the Tier 0 input it was built from.
    assert all(e.text.startswith("fixed-") for e in sync_entries)
    assert sync_entries != ENTRIES

    assert async_entries == sync_entries


def test_i6_convergence_is_reachable_from_a_partial_start(store):
    """Partial delivery must not strand segments permanently un-escalated."""
    backend = ChaosBackend(SyncAsyncAdapter(StubSync()), faults=("partial",), seed=2)
    escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    _drain(backend, store)

    clean = SyncAsyncAdapter(StubSync())
    escalate("vid1", ENTRIES, SEGMENTS, clean, store, TABLE, 10.0)
    version, converged = _drain(clean, store)
    assert converged
    entries = _track(store, version)
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
    job_id = escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version, converged = _drain(backend, store, max_passes=20)
    # Adjusted for I6 (dead-lettering, fix round: reconcile per-job
    # isolation): this backend can never deliver a job's full set, so
    # before dead-lettering existed the job stayed "running" forever and
    # the drain ran out of passes without converging. reconcile() now
    # counts a persistently-incomplete delivery the same way it counts a
    # raising poll(): once bump_job_attempts() exceeds MAX_JOB_ATTEMPTS,
    # the job is dead-lettered (state -> "failed"), which drops it out of
    # open_jobs() -- satisfying _drain()'s convergence check. So
    # `converged` is now True, but by giving up on the job, not by it
    # completing.
    assert converged
    entries = _track(store, version)

    # I1: every segment appears exactly once, nothing lost or duplicated.
    ids = [e.segment_id for e in entries]
    assert len(ids) == N
    assert len(set(ids)) == N
    assert sorted(ids) == sorted(SEGMENTS)

    # The job can never cover all its registered segment_ids, so it must
    # never be falsely marked done -- it is dead-lettered ("failed")
    # instead and no longer shows up in open_jobs() for further retry.
    job = store.get_job(job_id)
    assert job["state"] == "failed"
    assert job_id not in {j["job_id"] for j in store.open_jobs()}

    # The segments that were never delivered keep their original Tier 0
    # text/risk/band untouched -- degrade safely, don't blank or corrupt.
    delivered = [e for e in entries if e.text != "raw"]
    undelivered = [e for e in entries if e.text == "raw"]
    assert delivered and undelivered
    assert all(e.risk == 0.65 and e.band == "review" for e in undelivered)
    assert all(e.text.startswith("fixed-") for e in delivered)
