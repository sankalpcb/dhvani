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
