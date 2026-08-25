import json
import numpy as np
import pytest

from dhvani.backends.base import Recorded
from dhvani.calibrate import (
    MIN_BUCKET_SAMPLES,
    escalate_selected,
    estimate_cost,
    write_table,
)
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


def _recorded(store, tmp_path):
    """The stub wrapped exactly the way dhvani.cli_calibrate wraps the real
    Tier1Chirp.

    C3: escalate_selected() used to call store.reserve_spend() itself AND
    hand the call to a Recorded wrapper that reserves again, so a nominal
    $1.00 call reserved $2.00. The reservation now lives only in Recorded,
    which is the only layer that knows whether a call is actually paid.
    These tests therefore exercise the wrapped stack rather than a bare
    stub -- moving them closer to what the CLI does, not further from it.
    """
    return Recorded(StubTier1(), "live", str(tmp_path / "fixtures"), store)


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


def test_escalate_reserves_spend_before_calling(store, tmp_path):
    sel = _selected(2)
    _seed_refs(store, sel)
    escalate_selected(sel, _recorded(store, tmp_path), store, _segments(sel),
                      "tier0|hi|m")
    assert store.total_spend() > 0.0


def test_escalate_reserves_each_paid_call_exactly_once(store, tmp_path):
    """C3 regression: with escalate_selected() reserving AND Recorded
    reserving, the ledger read double the true cost -- so --confirm showed
    the operator half of what was actually taken."""
    from dhvani.backends.tier1_chirp import cost_for_duration_ms

    sel = _selected(3)
    _seed_refs(store, sel)
    escalate_selected(sel, _recorded(store, tmp_path), store, _segments(sel),
                      "tier0|hi|m")
    assert store.total_spend() == pytest.approx(3 * cost_for_duration_ms(3000))


def test_rerunning_escalation_reserves_nothing_further(store, tmp_path):
    """Idempotent spend: cached tier1 hypotheses must not be re-paid."""
    sel = _selected(3)
    _seed_refs(store, sel)
    escalate_selected(sel, _recorded(store, tmp_path), store, _segments(sel),
                      "tier0|hi|m")
    after_first = store.total_spend()
    assert after_first > 0.0, "the first pass must actually have paid"
    escalate_selected(sel, _recorded(store, tmp_path), store, _segments(sel),
                      "tier0|hi|m")
    assert store.total_spend() == pytest.approx(after_first)


def test_escalation_fails_closed_at_the_ceiling(store, tmp_path):
    sel = _selected(3)
    _seed_refs(store, sel)
    store.reserve_spend("tier1", 20.0 - 0.0001)
    with pytest.raises(BudgetExceeded):
        escalate_selected(sel, _recorded(store, tmp_path), store, _segments(sel),
                          "tier0|hi|m")


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
    assert meta["segments_selected"] == 25
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


# --- I4: the sample floor must guard the SURVIVING rows, not the scored ones ---

def test_write_table_drops_buckets_that_fell_below_the_floor(tmp_path):
    """stratify() applies MIN_BUCKET_SAMPLES to the scored population, but
    escalate_selected() drops rows afterwards. A bucket that entered with
    22 scored segments and left with 3 rows used to publish a 3-sample mean
    as a measured value."""
    thin = [{"risk": 0.35, "reference": "a b c d",
             "tier0_text": "a b c X", "tier1_text": "a b c d"}] * 3
    fat = [{"risk": 0.65, "reference": "a b c d",
            "tier0_text": "a b c X", "tier1_text": "a b c d"}] * MIN_BUCKET_SAMPLES

    path = tmp_path / "delta_table.json"
    table = write_table(thin + fat, _selected(23), str(path),
                        spend_usd=0.0, langs=["hi-IN"])

    written = json.loads(path.read_text())
    assert "0.3-0.4" not in written["tier1"], "under-floor bucket must not be published"
    assert "0.6-0.7" in written["tier1"]
    assert table["tier1"] == written["tier1"], "return value must match the file"
    assert written["meta"]["dropped_buckets"] == ["0.3-0.4"], (
        "a bucket dropped for thinness must be reported, not vanish silently"
    )
    assert written["meta"]["bucket_n"]["0.3-0.4"] == 3, (
        "bucket_n keeps the pre-drop count as the evidence for the drop"
    )


def test_write_table_keeps_a_bucket_exactly_at_the_floor(tmp_path):
    rows = [{"risk": 0.65, "reference": "a b", "tier0_text": "a b",
             "tier1_text": "a b"}] * MIN_BUCKET_SAMPLES
    path = tmp_path / "t.json"
    write_table(rows, _selected(MIN_BUCKET_SAMPLES), str(path),
                spend_usd=0.0, langs=["hi-IN"])
    assert "0.6-0.7" in json.loads(path.read_text())["tier1"]


# --- I7: an empty or floor-less table must not be written at all ---

def test_write_table_refuses_to_write_an_empty_table(tmp_path):
    from dhvani.calibrate import NoMeasuredBuckets

    path = tmp_path / "delta_table.json"
    with pytest.raises(NoMeasuredBuckets) as exc:
        write_table([], _selected(25), str(path), spend_usd=0.0, langs=["hi-IN"])
    assert str(path) in str(exc.value)
    assert not path.exists(), "must leave any previous, real table in place"


def test_write_table_refuses_when_no_bucket_clears_the_floor(tmp_path):
    """Rows existed, but every bucket was too thin to mean anything. That is
    still the absence of measurement, not a measurement of zero."""
    from dhvani.calibrate import NoMeasuredBuckets

    rows = ([{"risk": 0.35, "reference": "a b", "tier0_text": "a X",
              "tier1_text": "a b"}] * 3
            + [{"risk": 0.65, "reference": "a b", "tier0_text": "a X",
                "tier1_text": "a b"}] * 4)
    path = tmp_path / "delta_table.json"
    with pytest.raises(NoMeasuredBuckets):
        write_table(rows, _selected(7), str(path), spend_usd=0.0, langs=["hi-IN"])
    assert not path.exists()


# --- I8: the two counts must be distinguishable ---

def test_write_table_separates_selected_from_escalated(tmp_path):
    """meta.segments_escalated was len(selected), so it could read 812
    beside an empty tier1 map. The gap between the two is the skip count."""
    rows = [{"risk": 0.65, "reference": "a b c d",
             "tier0_text": "a b c X", "tier1_text": "a b c d"}] * 21
    path = tmp_path / "t.json"
    write_table(rows, _selected(40), str(path), spend_usd=0.0, langs=["hi-IN"])

    meta = json.loads(path.read_text())["meta"]
    assert meta["segments_selected"] == 40
    assert meta["segments_escalated"] == 21


# --- transient Tier 1 failures must not abandon a paid run ---

class FlakyTier1(StubTier1):
    """Fails `failures` times with a transient error, then succeeds.

    Models the real fault that killed a live run at call 45 of 124: Google
    returned `503 502:Bad Gateway`, escalate_selected had no retry, and the
    whole batch aborted.
    """

    def __init__(self, failures, error=None):
        self.failures = failures
        self.calls = 0
        self.error = error or RuntimeError("503 502:Bad Gateway")

    def transcribe(self, segment):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return {"text": "chirp output", "signals": {}}


def test_a_transient_tier1_failure_is_retried(store, tmp_path, monkeypatch):
    """reconcile() survives a raising poll() by design; the synchronous
    calibration path -- the one that spends money -- had nothing."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    sel = _selected(1)
    _seed_refs(store, sel)
    flaky = FlakyTier1(failures=1)

    rows = escalate_selected(sel, Recorded(flaky, "live", str(tmp_path / "f"), store),
                             store, _segments(sel), "tier0|hi|m")

    assert len(rows) == 1, "a transient failure must not lose the segment"
    assert flaky.calls == 2, "must have retried exactly once"


def test_one_flaky_segment_does_not_abandon_the_rest_of_the_batch(store, tmp_path,
                                                                  monkeypatch):
    """The 503 aborted 79 remaining segments. Every segment that CAN be
    transcribed must be."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    sel = _selected(3)
    _seed_refs(store, sel)

    rows = escalate_selected(sel, Recorded(FlakyTier1(failures=2), "live",
                                           str(tmp_path / "f"), store),
                             store, _segments(sel), "tier0|hi|m")

    assert len(rows) == 3


def test_a_persistently_failing_tier1_still_raises(store, tmp_path, monkeypatch):
    """Bounded, not infinite: a backend that is genuinely down must surface,
    not spin forever reserving spend on every attempt."""
    from dhvani.calibrate import MAX_TIER1_ATTEMPTS

    monkeypatch.setattr("time.sleep", lambda s: None)
    sel = _selected(1)
    _seed_refs(store, sel)
    flaky = FlakyTier1(failures=999)

    with pytest.raises(RuntimeError):
        escalate_selected(sel, Recorded(flaky, "live", str(tmp_path / "f"), store),
                          store, _segments(sel), "tier0|hi|m")

    assert flaky.calls == MAX_TIER1_ATTEMPTS


def test_a_budget_breach_is_never_retried(store, tmp_path, monkeypatch):
    """BudgetExceeded is terminal, not transient. Retrying it would reserve
    spend again on every attempt and hammer the ceiling that just refused."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    sel = _selected(1)
    _seed_refs(store, sel)
    flaky = FlakyTier1(failures=999, error=BudgetExceeded("ceiling"))

    with pytest.raises(BudgetExceeded):
        escalate_selected(sel, flaky, store, _segments(sel), "tier0|hi|m")

    assert flaky.calls == 1, "a ceiling refusal must fail immediately"


def test_a_missing_pcm_cache_entry_is_never_retried(store, tmp_path, monkeypatch):
    """Also terminal: the audio is not going to appear on attempt three."""
    from dhvani.calibrate import PcmCacheMiss

    monkeypatch.setattr("time.sleep", lambda s: None)
    sel = _selected(1)
    _seed_refs(store, sel)
    flaky = FlakyTier1(failures=999, error=PcmCacheMiss("no pcm"))

    with pytest.raises(PcmCacheMiss):
        escalate_selected(sel, flaky, store, _segments(sel), "tier0|hi|m")

    assert flaky.calls == 1


def test_retries_back_off_between_attempts(store, tmp_path, monkeypatch):
    """A tight loop against a struggling backend is its own denial of
    service, and each attempt reserves spend."""
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    sel = _selected(1)
    _seed_refs(store, sel)

    escalate_selected(sel, Recorded(FlakyTier1(failures=2), "live",
                                    str(tmp_path / "f"), store),
                      store, _segments(sel), "tier0|hi|m")

    assert len(slept) == 2, f"expected a pause per retry, got {slept}"
    assert slept[1] > slept[0], f"backoff must grow: {slept}"
