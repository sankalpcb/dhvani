import numpy as np
import pytest

from dhvani.backends.async_base import AsyncBackend, SyncAsyncAdapter, JobNotFound
from dhvani.segmenter import Segment


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def __init__(self):
        self.calls = 0

    def cost_per_call(self, segment):
        return 0.003

    def transcribe(self, segment):
        self.calls += 1
        return {"text": f"out-{segment.segment_id[:4]}", "signals": {}}


def _segs(n=2):
    return [Segment(chr(97 + i) * 64, i * 3000, (i + 1) * 3000,
                    np.zeros(10, dtype=np.int16)) for i in range(n)]


def test_adapter_satisfies_async_backend_protocol():
    assert isinstance(SyncAsyncAdapter(StubSync()), AsyncBackend)


def test_submit_returns_a_stable_job_id_for_the_same_segments():
    a = SyncAsyncAdapter(StubSync()).submit(_segs())
    b = SyncAsyncAdapter(StubSync()).submit(_segs())
    assert a == b, "job id must be content-derived, not random"


def test_submit_returns_different_ids_for_different_segments():
    assert SyncAsyncAdapter(StubSync()).submit(_segs(2)) != \
           SyncAsyncAdapter(StubSync()).submit(_segs(3))


def test_submit_does_not_call_the_inner_backend():
    """Submission is cheap; the work happens at poll time."""
    inner = StubSync()
    SyncAsyncAdapter(inner).submit(_segs())
    assert inner.calls == 0


def test_poll_returns_results_keyed_by_segment_id():
    a = SyncAsyncAdapter(StubSync())
    job_id = a.submit(_segs())
    out = a.poll(job_id)
    assert set(out) == {"a" * 64, "b" * 64}
    assert out["a" * 64]["text"].startswith("out-")


def test_poll_returns_none_while_pending():
    a = SyncAsyncAdapter(StubSync(), pending_polls=2)
    job_id = a.submit(_segs())
    assert a.poll(job_id) is None
    assert a.poll(job_id) is None
    assert a.poll(job_id) is not None


def test_poll_on_unknown_job_raises():
    with pytest.raises(JobNotFound):
        SyncAsyncAdapter(StubSync()).poll("no-such-job")


def test_cost_per_call_delegates_to_inner():
    assert SyncAsyncAdapter(StubSync()).cost_per_call(_segs()[0]) == 0.003
