import numpy as np
import pytest

from dhvani.backends.async_base import SyncAsyncAdapter
from dhvani.escalate import escalate
from dhvani.pipeline import TrackEntry
from dhvani.reconcile import reconcile
from dhvani.segmenter import Segment
from dhvani.store import Store
from dhvani.track import entries_to_json, entries_from_json
from dhvani.config import POLICY_ID


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def cost_per_call(self, segment):
        return 0.00075

    def transcribe(self, segment):
        return {"text": "escalated", "signals": {"ctc_rnnt_disagreement": 0.0}}


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


ENTRIES = [TrackEntry("a" * 64, 0, 3000, "raw", 0.65, "review"),
           TrackEntry("b" * 64, 3000, 6000, "raw", 0.05, "ship")]
SEGMENTS = {e.segment_id: Segment(e.segment_id, e.t_start_ms, e.t_end_ms,
                                  np.zeros(10, dtype=np.int16))
            for e in ENTRIES}
TABLE = {"tier1": {"0.6-0.7": 18.0}}


def _seed_v1(store):
    store.put_track("vid1", 1, POLICY_ID, entries_to_json(ENTRIES), 0.0)


def test_reconcile_with_no_jobs_leaves_the_version_alone(store):
    _seed_v1(store)
    assert reconcile("vid1", SyncAsyncAdapter(StubSync()), store) == 1


def test_reconcile_advances_the_version_when_results_arrive(store):
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    assert reconcile("vid1", backend, store) == 2


def test_reconciled_track_contains_the_escalated_text(store):
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version = reconcile("vid1", backend, store)
    merged = entries_from_json(store.get_track("vid1", version)["content_json"])
    assert merged[0].text == "escalated"
    assert merged[1].text == "raw"


def test_pending_job_does_not_advance_the_version(store):
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync(), pending_polls=5)
    escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    assert reconcile("vid1", backend, store) == 1


def test_completed_job_is_marked_done(store):
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    job_id = escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    reconcile("vid1", backend, store)
    assert store.get_job(job_id)["state"] == "done"


def test_reconciling_twice_does_not_advance_twice(store):
    """Invariant I2 at the reconciler level: a settled job is not re-merged."""
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    first = reconcile("vid1", backend, store)
    second = reconcile("vid1", backend, store)
    assert first == second == 2


def test_reconcile_never_loses_segments(store):
    """Invariant I1."""
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version = reconcile("vid1", backend, store)
    merged = entries_from_json(store.get_track("vid1", version)["content_json"])
    assert sorted(e.segment_id for e in merged) == \
           sorted(e.segment_id for e in ENTRIES)


class PartialDeliveryBackend:
    """Wraps a real AsyncBackend but truncates poll() results to a subset
    of the job's segments, simulating a batch that only partially completed.
    """

    def __init__(self, inner, keep: int):
        self.inner = inner
        self.name = inner.name
        self.variant_key = inner.variant_key
        self.keep = keep

    def cost_per_call(self, segment):
        return self.inner.cost_per_call(segment)

    def submit(self, segments):
        return self.inner.submit(segments)

    def poll(self, job_id):
        results = self.inner.poll(job_id)
        if results is None:
            return None
        keys = list(results.keys())[: self.keep]
        return {k: results[k] for k in keys}


def test_partial_delivery_leaves_job_running_not_done(store):
    """A batch that returns results for only some of its registered
    segment_ids must not settle the job -- the missing segments would
    otherwise never be retried, breaking convergence with a synchronous
    run (invariant I1 at the job level).

    escalate() only ever selects one of these two entries (TABLE has no
    bucket for entry b's risk), so the batch is submitted directly here,
    registering both segment_ids on one job, to exercise a genuine
    multi-segment partial delivery.
    """
    _seed_v1(store)
    inner = SyncAsyncAdapter(StubSync())
    batch = [SEGMENTS[e.segment_id] for e in ENTRIES]
    job_id = inner.submit(batch)
    store.put_job(job_id, inner.name, inner.variant_key,
                  [s.segment_id for s in batch], "vid1")

    backend = PartialDeliveryBackend(inner, keep=1)
    reconcile("vid1", backend, store)
    job = store.get_job(job_id)
    assert job["state"] == "running"
    assert job_id in [j["job_id"] for j in store.open_jobs()]


class StaleVersionStore:
    """Wraps a real Store so its FIRST latest_track_version() call returns
    a stale value, as if that read had happened before a concurrent writer
    committed a newer version -- everything else (get_track, put_track,
    job methods, and any later latest_track_version() call) goes straight
    to the real store, which already reflects the concurrent write.

    This is what lets a single-threaded test reproduce the race: by the
    time reconcile()'s own put_track() runs, the version it computed from
    the stale read collides with what the "other writer" already landed.
    """

    def __init__(self, inner, stale_version: int):
        self._inner = inner
        self._stale_version = stale_version
        self._served_stale = False

    def latest_track_version(self, source_id: str) -> int:
        if not self._served_stale:
            self._served_stale = True
            return self._stale_version
        return self._inner.latest_track_version(source_id)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_lost_race_on_put_track_does_not_settle_job_or_lose_result(store):
    """put_track() is INSERT OR IGNORE: on a (source_id, version) collision
    it returns False and writes nothing. If reconcile() ignored that and
    marked the job done anyway, the merged result would exist nowhere
    durable and the job would never be retried -- a silent I1 violation.

    Reproduces the race directly: seed v1, escalate a job, then have a
    second writer land v2 (with content that is NOT the merge) -- and make
    reconcile()'s own latest_track_version() read return the stale v1, as
    if that read happened before the other writer's commit, so its own
    put_track(vid1, 2, ...) collides with what the other writer already
    landed. reconcile() must not claim a version it did not write, must
    leave the job open for retry, and a later pass -- now racing against
    nothing -- must succeed.
    """
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    job_id = escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)

    # A concurrent writer lands version 2 first, with content that is not
    # the escalated merge (here, an untouched copy of v1's raw entries).
    store.put_track("vid1", 2, POLICY_ID, entries_to_json(ENTRIES), 0.0)

    # reconcile() reads a stale latest_track_version() (1) -- as if its
    # read raced ahead of the other writer's commit -- so it computes
    # new_version = 2 and collides with what's already there.
    stale_store = StaleVersionStore(store, stale_version=1)

    lost_race_version = reconcile("vid1", backend, stale_store)

    # reconcile() must report the version actually in the store (read via
    # its second, non-stale latest_track_version() call), not the version
    # it attempted -- and failed -- to write.
    assert lost_race_version == 2
    stored = entries_from_json(store.get_track("vid1", lost_race_version)["content_json"])
    assert [e.text for e in stored] == ["raw", "raw"]

    # The job's result was not persisted anywhere -- it must remain open
    # for retry, not settled as done.
    job = store.get_job(job_id)
    assert job["state"] == "running"
    assert job_id in [j["job_id"] for j in store.open_jobs()]

    # A later pass, now racing against nothing (real store, no stale
    # read), must succeed and actually merge the escalated text in -- this
    # is what proves recovery works rather than just failing quietly.
    recovered_version = reconcile("vid1", backend, store)
    assert recovered_version == 3
    merged = entries_from_json(store.get_track("vid1", recovered_version)["content_json"])
    assert merged[0].text == "escalated"
    assert store.get_job(job_id)["state"] == "done"


# --- C1: a reconcile() pass must never touch another source's jobs ---

OTHER_ENTRIES = [TrackEntry("c" * 64, 0, 3000, "raw", 0.65, "review")]
OTHER_SEGMENTS = {e.segment_id: Segment(e.segment_id, e.t_start_ms, e.t_end_ms,
                                        np.zeros(11, dtype=np.int16))
                  for e in OTHER_ENTRIES}


def test_reconcile_does_not_settle_another_sources_job(store):
    """C1: jobs were filtered on tier and variant_key only, never on the
    source they belong to, so reconcile("vid2") polled and marked done
    every outstanding tier1 job in the DB -- including vid1's.

    merge_entries() ignores foreign segment_ids, so vid1's *track* was not
    corrupted. The damage is quieter: vid1's job was settled, dropped out
    of open_jobs(), and was never retried, so the money already reserved
    for it bought nothing and its escalated text was unreachable forever.
    dhvani.cli's --db defaults to one shared path, so two videos really do
    share a store.
    """
    store.put_track("vid1", 1, POLICY_ID, entries_to_json(ENTRIES), 0.0)
    store.put_track("vid2", 1, POLICY_ID, entries_to_json(OTHER_ENTRIES), 0.0)

    backend = SyncAsyncAdapter(StubSync())
    job1 = escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    job2 = escalate("vid2", OTHER_ENTRIES, OTHER_SEGMENTS, backend, store,
                    TABLE, 10.0)
    assert job1 != job2

    # Reconcile ONLY vid2. vid1 has not been polled at all.
    assert reconcile("vid2", backend, store) == 2

    # vid1's job is untouched: still open, still retryable, not settled.
    assert store.get_job(job1)["state"] in ("pending", "running")
    assert job1 in [j["job_id"] for j in store.open_jobs("vid1")]
    assert job1 not in [j["job_id"] for j in store.open_jobs("vid2")]
    assert store.latest_track_version("vid1") == 1

    # And reconciling vid1 afterwards still delivers what was paid for.
    v1 = reconcile("vid1", backend, store)
    assert v1 == 2
    merged = entries_from_json(store.get_track("vid1", v1)["content_json"])
    assert merged[0].text == "escalated"
    assert store.get_job(job1)["state"] == "done"


def test_reconcile_ignores_a_foreign_job_even_at_the_same_version(store):
    """The foreign job must not even be polled: a job for another source
    contributes nothing this track can merge, so letting it in can only
    settle it wrongly."""
    store.put_track("vid1", 1, POLICY_ID, entries_to_json(ENTRIES), 0.0)
    store.put_track("vid2", 1, POLICY_ID, entries_to_json(OTHER_ENTRIES), 0.0)

    polled = []

    class RecordingBackend(SyncAsyncAdapter):
        def poll(self, job_id):
            polled.append(job_id)
            return super().poll(job_id)

    backend = RecordingBackend(StubSync())
    job1 = escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    job2 = escalate("vid2", OTHER_ENTRIES, OTHER_SEGMENTS, backend, store,
                    TABLE, 10.0)

    reconcile("vid2", backend, store)
    assert polled == [job2]
    assert job1 not in polled
