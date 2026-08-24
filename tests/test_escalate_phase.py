import json
import numpy as np
import pytest

from dhvani.calibrate import estimate_cost, escalate_selected, write_table
from dhvani.segmenter import Segment
from dhvani.store import Store, BudgetExceeded

# NOTE: test_budget_failure_leaves_no_table_behind is intentionally omitted
# here. It drives the CLI rather than escalate_selected directly, because
# the "writes nothing" guarantee is a property of the CLI's ordering
# (escalate before write_table), not of escalate_selected alone. It belongs
# to Task 6, which is where dhvani.cli_calibrate is created.


class StubTier1:
    name = "tier1"
    variant_key = "tier1|hi-IN|"

    def cost_per_call(self, segment):
        from dhvani.backends.tier1_chirp import cost_for_duration_ms
        return cost_for_duration_ms(segment.t_end_ms - segment.t_start_ms)

    def transcribe(self, segment):
        return {"text": "chirp output", "signals": {}}


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def _selected(n=3):
    return [{"segment_id": f"s{i:04d}" + "0" * 59, "risk": 0.65,
             "lang": "hi-IN", "duration_ms": 3000} for i in range(n)]


def _segments(selected):
    return {s["segment_id"]: Segment(s["segment_id"], 0, s["duration_ms"],
                                     np.zeros(10, dtype=np.int16))
            for s in selected}


def _seed_refs(store, selected):
    for s in selected:
        store.put_reference(s["segment_id"], "alpha beta gamma delta", "hi-IN")
        store.put_hypothesis(s["segment_id"], "tier0", "alpha beta gamma WRONG",
                             {}, 0.0, "tier0|hi|m")


def test_estimate_uses_the_single_cost_model():
    from dhvani.backends.tier1_chirp import cost_for_duration_ms
    assert estimate_cost(_selected(4)) == pytest.approx(4 * cost_for_duration_ms(3000))


def test_estimate_of_empty_is_zero():
    assert estimate_cost([]) == 0.0


def test_escalate_produces_one_row_per_selected_segment(store):
    sel = _selected(3)
    _seed_refs(store, sel)
    rows = escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")
    assert len(rows) == 3
    assert set(rows[0]) == {"risk", "reference", "tier0_text", "tier1_text"}


def test_escalate_reserves_spend_before_calling(store):
    sel = _selected(2)
    _seed_refs(store, sel)
    escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")
    assert store.total_spend() > 0.0


def test_rerunning_escalation_reserves_nothing_further(store):
    """Idempotent spend: cached tier1 hypotheses must not be re-paid."""
    sel = _selected(3)
    _seed_refs(store, sel)
    escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")
    after_first = store.total_spend()
    escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")
    assert store.total_spend() == pytest.approx(after_first)


def test_escalation_fails_closed_at_the_ceiling(store):
    sel = _selected(3)
    _seed_refs(store, sel)
    store.reserve_spend("tier1", 20.0 - 0.0001)
    with pytest.raises(BudgetExceeded):
        escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")


def test_segments_missing_a_reference_are_skipped(store):
    """A segment with no ground truth cannot produce a meaningful delta."""
    sel = _selected(2)
    _seed_refs(store, sel[:1])
    rows = escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")
    assert len(rows) == 1


def test_write_table_records_provenance(tmp_path):
    rows = [{"risk": 0.65, "reference": "a b c d",
             "tier0_text": "a b c X", "tier1_text": "a b c d"}] * 25
    sel = _selected(25)
    path = tmp_path / "delta_table.json"
    table = write_table(rows, sel, str(path), spend_usd=0.019, langs=["hi-IN"])

    written = json.loads(path.read_text())
    assert "tier1" in written
    meta = written["meta"]
    assert meta["policy_id"] and meta["risk_weights"]
    assert meta["segments_escalated"] == 25
    assert meta["spend_usd"] == pytest.approx(0.019)
    assert meta["languages"] == ["hi-IN"]
    assert table["tier1"] == written["tier1"]


def test_write_table_records_per_bucket_counts(tmp_path):
    rows = [{"risk": 0.65, "reference": "a b", "tier0_text": "a b",
             "tier1_text": "a b"}] * 22
    sel = _selected(22)
    path = tmp_path / "t.json"
    write_table(rows, sel, str(path), spend_usd=0.0, langs=["hi-IN"])
    assert json.loads(path.read_text())["meta"]["bucket_n"]["0.6-0.7"] == 22
