import numpy as np
import pytest

from dhvani.backends.async_base import SyncAsyncAdapter
from dhvani.backends.tier1_chirp import cost_for_duration_ms
from dhvani.escalate import escalate
from dhvani.pipeline import TrackEntry
from dhvani.segmenter import Segment
from dhvani.store import Store, BudgetExceeded


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def cost_per_call(self, segment):
        from dhvani.backends.tier1_chirp import cost_for_duration_ms
        return cost_for_duration_ms(segment.t_end_ms - segment.t_start_ms)

    def transcribe(self, segment):
        return {"text": "escalated", "signals": {}}


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def _entries():
    return [TrackEntry("a" * 64, 0, 3000, "raw", 0.65, "review"),
            TrackEntry("b" * 64, 3000, 6000, "raw", 0.05, "ship")]


def _segments():
    """Real Segment objects — Tier1Chirp.transcribe() reads segment.pcm."""
    return {e.segment_id: Segment(e.segment_id, e.t_start_ms, e.t_end_ms,
                                  np.zeros(10, dtype=np.int16))
            for e in _entries()}


SEGMENTS = _segments()
TABLE = {"tier1": {"0.6-0.7": 18.0}}


def test_zero_budget_submits_nothing(store):
    assert escalate("vid1", _entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                    store, TABLE, budget_usd=0.0) is None


def test_empty_delta_table_submits_nothing(store):
    """No measured improvement means no candidate has positive delta."""
    assert escalate("vid1", _entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                    store, {}, budget_usd=10.0) is None


def test_escalation_registers_a_job_with_the_selected_segments(store):
    job_id = escalate("vid1", _entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                      store, TABLE, budget_usd=10.0)
    assert job_id is not None
    job = store.get_job(job_id)
    assert job["segment_ids"] == ["a" * 64]
    assert job["state"] == "pending"
    assert job["variant_key"] == "tier1|hi-IN"


def test_low_risk_segments_are_not_escalated(store):
    job_id = escalate("vid1", _entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                      store, TABLE, budget_usd=10.0)
    assert "b" * 64 not in store.get_job(job_id)["segment_ids"]


def test_spend_is_reserved_before_submission(store):
    escalate("vid1", _entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
             store, TABLE, budget_usd=10.0)
    assert store.total_spend() > 0.0


def test_escalation_fails_closed_at_the_ceiling(store):
    # The only candidate ever selected here is "a" (risk 0.65, bucket
    # "0.6-0.7", the sole key in TABLE); "b" (risk 0.05) has no matching
    # bucket, delta 0.0, and is excluded by plan()'s invariant I3. That
    # single 3000ms segment costs cost_for_duration_ms(3000) == $0.00075
    # (rounds up to the 15s billing increment at $0.003/min). Reserving
    # the brief's literal $19.999 first leaves $0.001 of headroom — more
    # than the batch actually costs — so it would NOT breach the ceiling
    # (19.999 + 0.00075 == 19.99975 <= 20.0, legally under the "boundary:
    # projected == MAX_SPEND_USD is allowed" rule in Store.reserve_spend).
    # Reserve just enough that the batch's real cost tips it over instead.
    from dhvani.backends.tier1_chirp import cost_for_duration_ms
    almost_full = 20.0 - cost_for_duration_ms(3000) + 0.0001
    store.reserve_spend("tier1", almost_full)
    with pytest.raises(BudgetExceeded):
        escalate("vid1", _entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                 store, TABLE, budget_usd=10.0)


def test_resubmitting_the_same_batch_is_idempotent(store):
    backend = SyncAsyncAdapter(StubSync())
    first = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)
    second = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)
    assert first == second


def _multi_entries():
    """Three entries whose risk buckets all carry positive delta, so the
    router genuinely selects 3 candidates instead of the single-candidate
    path every other test in this file exercises."""
    return [TrackEntry("c" * 64, 0, 3000, "raw", 0.65, "review"),
            TrackEntry("d" * 64, 3000, 6000, "raw", 0.75, "review"),
            TrackEntry("e" * 64, 6000, 9000, "raw", 0.85, "review")]


def _multi_segments():
    return {e.segment_id: Segment(e.segment_id, e.t_start_ms, e.t_end_ms,
                                  np.zeros(10, dtype=np.int16))
            for e in _multi_entries()}


MULTI_SEGMENTS = _multi_segments()
MULTI_TABLE = {"tier1": {"0.6-0.7": 10.0, "0.7-0.8": 10.0, "0.8-0.9": 10.0}}


def test_multi_candidate_batch_reserves_summed_cost_in_one_call(store):
    """Regression guard for reserving per-candidate in a loop: the batch's
    total cost must be reserved with a single reserve_spend() call, not N
    partial ones."""
    calls = []
    original_reserve = store.reserve_spend

    def spy(tier, cost_usd):
        calls.append((tier, cost_usd))
        return original_reserve(tier, cost_usd)

    store.reserve_spend = spy

    job_id = escalate("vid1", _multi_entries(), MULTI_SEGMENTS, SyncAsyncAdapter(StubSync()),
                      store, MULTI_TABLE, budget_usd=10.0)

    assert job_id is not None
    per_segment_cost = cost_for_duration_ms(3000)
    assert len(calls) == 1
    assert calls[0] == ("tier1", pytest.approx(3 * per_segment_cost))
    assert store.total_spend() == pytest.approx(3 * per_segment_cost)


def test_multi_candidate_reservation_is_atomic_on_failure(store):
    """When the ledger has room for only part of the batch, nothing may be
    partially reserved and no job may be created -- the failure must be
    all-or-nothing. This fails against a per-candidate reservation loop,
    which would reserve the first 2 of 3 candidates before raising on the
    3rd, permanently burning budget for a batch that was never submitted."""
    per_segment_cost = cost_for_duration_ms(3000)
    # Leave room for exactly 2 of the 3 candidates, not all 3.
    pre_reserved = 20.0 - (2 * per_segment_cost) - 0.0000001
    store.reserve_spend("tier1", pre_reserved)
    total_before = store.total_spend()

    with pytest.raises(BudgetExceeded):
        escalate("vid1", _multi_entries(), MULTI_SEGMENTS, SyncAsyncAdapter(StubSync()),
                 store, MULTI_TABLE, budget_usd=10.0)

    assert store.total_spend() == pytest.approx(total_before)
    assert store.open_jobs() == []
