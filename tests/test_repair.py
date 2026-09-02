"""Tier 2 repair orchestration and graceful degradation (design §3).

M5's demo is "graceful degradation at quota exhaustion", so most of these
tests are about what happens when the quota runs out mid-run: the run must
finish, the captions must still ship, and the unrepaired segments must be
both marked and recoverable.

Deferred work needs no jobs row. A segment still wanting repair is exactly
one with a positive tier2 delta and no tier2 hypothesis, and hypotheses are
content-addressed, so the next run recomputes the same set and resumes.
"""

import pytest

from dhvani.quota import QuotaGate, TokenBucket
from dhvani.repair import repair
from dhvani.store import Store
from dhvani.backends.tier2_gemini import Tier2Gemini


class Entry:
    def __init__(self, segment_id, risk):
        self.segment_id = segment_id
        self.risk = risk


class Segment:
    def __init__(self, segment_id):
        self.segment_id = segment_id
        self.t_start_ms = 0
        self.t_end_ms = 4000
        self.pcm = b""


class FakeGemini:
    def __init__(self):
        self.calls = 0

    def repair(self, text, lang):
        self.calls += 1
        return f"repaired:{text}"


# risk 0.05 lands in bucket 0.0-0.1
TABLE = {"tier2": {"0.0-0.1": 5.0}}
NO_GAIN = {"tier2": {"0.0-0.1": -1.0}}


def _fixture(n=3, cap=10, capacity=100, table=TABLE):
    store = Store(":memory:")
    entries = [Entry(f"s{i}", 0.05) for i in range(n)]
    segments = {e.segment_id: Segment(e.segment_id) for e in entries}
    for e in entries:
        store.put_hypothesis(e.segment_id, "tier0", f"text{e.segment_id}", {}, 0.0)
    client = FakeGemini()
    backend = Tier2Gemini(
        hypothesis_source=lambda sid: f"text{sid}", client=client, model="fake"
    )
    gate = QuotaGate(
        store=store, tier="tier2", cap=cap,
        bucket=TokenBucket(capacity=capacity, refill_per_sec=1.0),
        today=lambda: "2026-09-02",
    )
    return store, entries, segments, backend, gate, client, table


def test_repairs_are_stored_as_tier2_hypotheses():
    store, entries, segments, backend, gate, client, table = _fixture(n=2)
    with store:
        out = repair("src", entries, segments, backend, store, table, gate)

        assert sorted(out.repaired) == ["s0", "s1"]
        assert out.deferred == []
        got = store.get_hypothesis("s0", "tier2", variant_key=backend.variant_key)
        assert got["text"] == "repaired:texts0"


def test_quota_exhaustion_does_not_raise_and_defers_the_rest():
    """The milestone's demo. The run completes; it does not abort."""
    store, entries, segments, backend, gate, client, table = _fixture(n=5, cap=2)
    with store:
        out = repair("src", entries, segments, backend, store, table, gate)

        assert len(out.repaired) == 2
        assert len(out.deferred) == 3
        assert out.reason == "quota_exhausted"
        assert client.calls == 2, "kept calling after the quota was gone"


def test_deferred_segments_resume_on_a_later_run():
    """No jobs row: the remaining work is recomputed and picked up."""
    store, entries, segments, backend, gate, client, table = _fixture(n=4, cap=2)
    with store:
        first = repair("src", entries, segments, backend, store, table, gate)
        assert len(first.repaired) == 2

        # A new day: fresh quota, same inputs.
        tomorrow = QuotaGate(
            store=store, tier="tier2", cap=2,
            bucket=TokenBucket(capacity=100, refill_per_sec=1.0),
            today=lambda: "2026-09-03",
        )
        second = repair("src", entries, segments, backend, store, table, tomorrow)

        assert sorted(first.repaired + second.repaired) == ["s0", "s1", "s2", "s3"]
        assert second.deferred == []


def test_an_already_repaired_segment_costs_no_quota():
    store, entries, segments, backend, gate, client, table = _fixture(n=2)
    with store:
        repair("src", entries, segments, backend, store, table, gate)
        used_after_first = store.quota_used("tier2", "2026-09-02")

        repair("src", entries, segments, backend, store, table, gate)

        assert store.quota_used("tier2", "2026-09-02") == used_after_first
        assert client.calls == 2, "re-repaired work already done"


def test_non_positive_delta_is_never_repaired():
    """Invariant I3 applies to Tier 2 exactly as it does to Tier 1."""
    store, entries, segments, backend, gate, client, table = _fixture(n=3, table=NO_GAIN)
    with store:
        out = repair("src", entries, segments, backend, store, NO_GAIN, gate)

        assert out.repaired == []
        assert client.calls == 0
        assert store.quota_used("tier2", "2026-09-02") == 0


# --- marking unrepaired segments (design §5.2) ---

from dhvani.pipeline import TrackEntry  # noqa: E402
from dhvani.repair import mark_unrepaired  # noqa: E402
from dhvani.track import entries_from_json, entries_to_json  # noqa: E402


def _entry(sid, risk=0.05):
    return TrackEntry(segment_id=sid, t_start_ms=0, t_end_ms=1000,
                      text="t", risk=risk, band="ship")


def test_deferred_segments_are_marked_and_others_are_not():
    entries = [_entry("s0"), _entry("s1")]

    marked = mark_unrepaired(entries, ["s1"])

    by_id = {e.segment_id: e for e in marked}
    assert by_id["s1"].repair_unavailable is True
    assert by_id["s0"].repair_unavailable is False


def test_marking_does_not_change_the_band():
    """repair_unavailable is a separate fact from risk. 'We did not get to
    improve this' is not the same as 'this is risky'."""
    entries = [_entry("s0", risk=0.9)]

    marked = mark_unrepaired(entries, ["s0"])

    assert marked[0].band == "ship", "marking must not touch the band"
    assert marked[0].risk == 0.9


def test_the_marker_survives_a_json_round_trip():
    marked = mark_unrepaired([_entry("s0")], ["s0"])
    assert entries_from_json(entries_to_json(marked))[0].repair_unavailable is True


def test_old_track_json_without_the_field_still_loads():
    """Backward compatibility: tracks written before Tier 2 existed."""
    legacy = ('[{"segment_id": "s0", "t_start_ms": 0, "t_end_ms": 1000, '
              '"text": "t", "risk": 0.05, "band": "ship"}]')
    assert entries_from_json(legacy)[0].repair_unavailable is False


# --- the vendor's own refusal (spike, 2026-09-02) ---

from dhvani.backends.base import FixtureMissing  # noqa: E402


class ExplodingBackend:
    """Stands in for Google rejecting a call our local counter allowed."""

    name = "tier2"
    variant_key = "model=fake;lang=hi-IN;prompt=test"

    def __init__(self, exc, fail_on=None):
        self.exc = exc
        self.fail_on = fail_on
        self.seen = []

    def cost_per_call(self, segment):
        return 0.0

    def transcribe(self, segment):
        self.seen.append(segment.segment_id)
        if self.fail_on is None or segment.segment_id in self.fail_on:
            raise self.exc
        return {"text": "ok", "signals": {}}


def test_a_vendor_refusal_defers_instead_of_crashing_the_run():
    """Our local cap is a guess -- the account's real RPD is not published.
    If we allow a call Google refuses, that must degrade, not abort."""
    store, entries, segments, _, gate, _, table = _fixture(n=3)
    backend = ExplodingBackend(RuntimeError("429 RESOURCE_EXHAUSTED"))
    with store:
        out = repair("src", entries, segments, backend, store, table, gate)

        assert out.repaired == []
        assert sorted(out.deferred) == ["s0", "s1", "s2"]
        assert out.reason == "backend_error"


def test_one_failing_segment_does_not_abandon_the_others():
    store, entries, segments, _, gate, _, table = _fixture(n=3)
    backend = ExplodingBackend(RuntimeError("boom"), fail_on={"s1"})
    with store:
        out = repair("src", entries, segments, backend, store, table, gate)

        assert sorted(out.repaired) == ["s0", "s2"]
        assert out.deferred == ["s1"]


def test_a_missing_fixture_is_still_a_hard_error():
    """Replay must never silently become a degraded run. 'Replay never
    falls back to live' is load-bearing, and a swallowed FixtureMissing
    would turn an offline mistake into a quiet no-op."""
    store, entries, segments, _, gate, _, table = _fixture(n=2)
    backend = ExplodingBackend(FixtureMissing("no fixture for seg"))
    with store:
        with pytest.raises(FixtureMissing):
            repair("src", entries, segments, backend, store, table, gate)
