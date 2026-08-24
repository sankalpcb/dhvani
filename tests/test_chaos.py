import numpy as np
import pytest

from dhvani.backends.async_base import SyncAsyncAdapter
from dhvani.backends.chaos import ChaosBackend, TransientError
from dhvani.segmenter import Segment


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def cost_per_call(self, segment):
        return 0.00075

    def transcribe(self, segment):
        return {"text": f"out-{segment.segment_id[:4]}", "signals": {}}


def _segs(n=4):
    return [Segment(chr(97 + i) * 64, i * 3000, (i + 1) * 3000,
                    np.zeros(10, dtype=np.int16)) for i in range(n)]


def _chaos(faults, seed=0, pending=0):
    return ChaosBackend(SyncAsyncAdapter(StubSync(), pending_polls=pending),
                        faults=faults, seed=seed)


def test_no_faults_is_transparent():
    b = _chaos([])
    out = b.poll(b.submit(_segs()))
    assert len(out) == 4


def test_timeout_fault_raises_transient_error():
    b = _chaos(["timeout"])
    with pytest.raises(TransientError, match="timeout"):
        b.poll(b.submit(_segs()))


def test_rate_limit_fault_raises_transient_error():
    b = _chaos(["rate_limit"])
    with pytest.raises(TransientError, match="429"):
        b.poll(b.submit(_segs()))


def test_server_error_fault_raises_transient_error():
    b = _chaos(["server_error"])
    with pytest.raises(TransientError, match="500"):
        b.poll(b.submit(_segs()))


def test_partial_fault_returns_a_strict_subset():
    b = _chaos(["partial"])
    out = b.poll(b.submit(_segs()))
    assert 0 < len(out) < 4


def test_duplicate_fault_still_returns_a_dict_keyed_by_segment_id():
    """A duplicate delivery cannot produce duplicate keys — that is the point."""
    b = _chaos(["duplicate"])
    job_id = b.submit(_segs())
    first, second = b.poll(job_id), b.poll(job_id)
    assert first == second


def test_reorder_fault_changes_iteration_order_but_not_content():
    b_plain = _chaos([])
    plain = b_plain.poll(b_plain.submit(_segs()))
    b = _chaos(["reorder"], seed=7)
    shuffled = b.poll(b.submit(_segs()))
    assert list(shuffled) != list(plain) or len(plain) < 2
    assert shuffled == plain, "reordering must not change the mapping"


def test_faults_are_deterministic_for_a_given_seed():
    a = _chaos(["partial"], seed=3)
    c = _chaos(["partial"], seed=3)
    assert a.poll(a.submit(_segs())) == c.poll(c.submit(_segs()))


def test_unknown_fault_name_is_rejected():
    with pytest.raises(ValueError, match="unknown fault"):
        _chaos(["earthquake"])
