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
    # runtime_checkable only verifies method presence, not method signatures or
    # non-method attributes like 'name' and 'variant_key'. This test confirms
    # methods exist but does not structurally guarantee full protocol compliance.
    assert isinstance(SyncAsyncAdapter(StubSync()), AsyncBackend)


def test_submit_returns_a_stable_job_id_for_the_same_segments():
    a = SyncAsyncAdapter(StubSync()).submit(_segs())
    b = SyncAsyncAdapter(StubSync()).submit(_segs())
    assert a == b, "job id must be content-derived, not random"


def test_submit_returns_different_ids_for_different_segments():
    assert SyncAsyncAdapter(StubSync()).submit(_segs(2)) != \
           SyncAsyncAdapter(StubSync()).submit(_segs(3))


def test_job_id_delimiter_prevents_variant_segment_collision():
    """Verify that job_id_for uses delimiters to prevent boundary ambiguity.

    Without delimiters, variant_key="ab" + segment="c"*64 would hash the same
    as variant_key="a" + segment="bc"*64, since the byte streams would be
    identical. The delimiter ensures they differ.
    """
    from dhvani.backends.async_base import job_id_for

    # Create two segments with ids that would collide without delimiters
    seg1 = Segment("c" * 64, 0, 3000, np.zeros(10, dtype=np.int16))
    seg2 = Segment("b" + "c" * 63, 0, 3000, np.zeros(10, dtype=np.int16))

    id1 = job_id_for("ab", [seg1])
    id2 = job_id_for("a", [seg2])

    assert id1 != id2, "Delimiter must prevent variant/segment boundary collision"


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


def test_poll_caches_completed_results():
    """Verify that poll() memoizes results after completion, not re-calling inner.

    This prevents wasteful re-invocation (and silent re-spend) when the
    reconciler or other code polls a completed job multiple times.
    """
    inner = StubSync()
    a = SyncAsyncAdapter(inner)
    job_id = a.submit(_segs())

    # First poll completes and calls inner.transcribe() for each segment
    result1 = a.poll(job_id)
    calls_after_first_poll = inner.calls

    # Second poll should return cached result without calling inner
    result2 = a.poll(job_id)
    calls_after_second_poll = inner.calls

    assert result1 == result2, "Results must be identical"
    assert calls_after_second_poll == calls_after_first_poll, \
        "Second poll must not re-call inner.transcribe()"


def test_poll_caching_respects_pending_polls():
    """Verify that pending_polls latency still works correctly with caching."""
    inner = StubSync()
    a = SyncAsyncAdapter(inner, pending_polls=1)
    job_id = a.submit(_segs())

    # First poll: pending
    assert a.poll(job_id) is None
    assert inner.calls == 0, "No call during pending phase"

    # Second poll: completes and caches
    result = a.poll(job_id)
    assert result is not None
    calls_after_completion = inner.calls

    # Third poll: should return cached result
    cached = a.poll(job_id)
    assert cached == result
    assert inner.calls == calls_after_completion, "Cached result not re-computed"
