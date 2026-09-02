from dhvani.calibrate import stratify, histogram, MIN_BUCKET_SAMPLES


def _scored(n, risk, prefix="s"):
    return [{"segment_id": f"{prefix}{i:04d}" + "0" * 58, "risk": risk} for i in range(n)]


def test_histogram_counts_by_bucket():
    scored = _scored(3, 0.65) + _scored(2, 0.05, "t")
    assert histogram(scored) == {"0.0-0.1": 2, "0.6-0.7": 3}


def test_histogram_of_empty_is_empty():
    assert histogram([]) == {}


def test_stratify_caps_at_n_per_bucket():
    assert len(stratify(_scored(500, 0.65), n_per_bucket=100)) == 100


def test_stratify_takes_all_when_under_the_cap():
    assert len(stratify(_scored(40, 0.65), n_per_bucket=100)) == 40


def test_stratify_omits_thin_buckets():
    """A bucket under MIN_BUCKET_SAMPLES is dropped, not sampled noisily."""
    thin = _scored(MIN_BUCKET_SAMPLES - 1, 0.35, "thin")
    fat = _scored(50, 0.65, "fat")
    out = stratify(thin + fat)
    assert {s["segment_id"][:3] for s in out} == {"fat"}


def test_stratify_keeps_a_bucket_exactly_at_the_floor():
    at_floor = _scored(MIN_BUCKET_SAMPLES, 0.35)
    assert len(stratify(at_floor)) == MIN_BUCKET_SAMPLES


def test_stratify_spans_multiple_buckets():
    out = stratify(_scored(30, 0.15, "a") + _scored(30, 0.85, "b"))
    prefixes = {s["segment_id"][0] for s in out}
    assert prefixes == {"a", "b"}


def test_stratify_is_deterministic_and_order_independent():
    """Invariant I5: a re-run must select the same segments, so the cache hits."""
    scored = _scored(200, 0.65)
    assert stratify(scored) == stratify(list(reversed(scored)))


def test_stratify_of_empty_is_empty():
    assert stratify([]) == []


# --- speaker-disjoint selection (spec §9.4) ---

import pytest

from dhvani.calibrate import district_spread
from dhvani.corpus import disjoint_by


def _spk(n, risk, speaker, district="D", prefix="s"):
    """Scored rows carrying the reference metadata collect() records."""
    return [{"segment_id": f"{prefix}{i:04d}" + "0" * 58, "risk": risk,
             "speaker_id": speaker, "district": district} for i in range(n)]


def test_a_speaker_is_selected_at_most_once():
    """Spec §9.4: if one speaker dominates, the risk function learns
    speaker identity as a difficulty proxy and the numbers are fiction."""
    scored = [
        *_spk(MIN_BUCKET_SAMPLES, 0.05, speaker="S1", prefix="a"),
        *_spk(MIN_BUCKET_SAMPLES, 0.05, speaker="S2", prefix="b"),
    ]
    selected = stratify(scored)

    speakers = [row["speaker_id"] for row in selected]
    assert sorted(speakers) == ["S1", "S2"], f"one speaker repeated: {speakers}"


def test_the_selected_set_satisfies_disjoint_by():
    """The function the spec asks for, asserted on the actual output."""
    scored = [
        *_spk(MIN_BUCKET_SAMPLES, 0.05, speaker="S1", prefix="a"),
        *_spk(MIN_BUCKET_SAMPLES, 0.05, speaker="S2", prefix="b"),
    ]
    assert disjoint_by(stratify(scored), "speaker_id") is True


def test_disjoint_by_reads_mappings_as_well_as_objects():
    """Scored rows are dicts; corpus items are objects. One function."""
    assert disjoint_by([{"speaker_id": "a"}, {"speaker_id": "b"}], "speaker_id") is True
    assert disjoint_by([{"speaker_id": "a"}, {"speaker_id": "a"}], "speaker_id") is False


def test_rows_without_speaker_metadata_are_not_dropped():
    """Backward compatibility: results/scored.json predates this field, and
    absent metadata is not a collision."""
    assert len(stratify(_scored(MIN_BUCKET_SAMPLES, 0.05))) == MIN_BUCKET_SAMPLES


def test_a_repeated_district_is_reported_but_never_dropped():
    """District-disjointness is IMPOSSIBLE at calibration scale -- 150
    samples across at most 145 districts -- so it is a diagnostic, not a
    gate. Enforcing it would make calibration unrunnable."""
    scored = [
        *_spk(MIN_BUCKET_SAMPLES, 0.05, speaker="S1", district="D1", prefix="a"),
        *_spk(MIN_BUCKET_SAMPLES, 0.05, speaker="S2", district="D1", prefix="b"),
    ]
    selected = stratify(scored)

    assert len(selected) == 2, "a shared district must not drop a sample"
    assert district_spread(selected) == {"D1": 2}
