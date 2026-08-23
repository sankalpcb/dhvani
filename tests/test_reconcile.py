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
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    assert reconcile("vid1", backend, store) == 2


def test_reconciled_track_contains_the_escalated_text(store):
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version = reconcile("vid1", backend, store)
    merged = entries_from_json(store.get_track("vid1", version)["content_json"])
    assert merged[0].text == "escalated"
    assert merged[1].text == "raw"


def test_pending_job_does_not_advance_the_version(store):
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync(), pending_polls=5)
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    assert reconcile("vid1", backend, store) == 1


def test_completed_job_is_marked_done(store):
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    job_id = escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    reconcile("vid1", backend, store)
    assert store.get_job(job_id)["state"] == "done"


def test_reconciling_twice_does_not_advance_twice(store):
    """Invariant I2 at the reconciler level: a settled job is not re-merged."""
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    first = reconcile("vid1", backend, store)
    second = reconcile("vid1", backend, store)
    assert first == second == 2


def test_reconcile_never_loses_segments(store):
    """Invariant I1."""
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
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
                  [s.segment_id for s in batch])

    backend = PartialDeliveryBackend(inner, keep=1)
    reconcile("vid1", backend, store)
    job = store.get_job(job_id)
    assert job["state"] == "running"
    assert job_id in [j["job_id"] for j in store.open_jobs()]
