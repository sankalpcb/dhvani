import pytest
from dhvani.metrics import Timer, percentile, summarize, throughput


def test_timer_measures_a_nonnegative_span():
    with Timer() as t:
        sum(range(1000))
    assert t.elapsed_ms >= 0.0


def test_percentile_endpoints():
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 0.5) == 50.0
    assert percentile(values, 0.99) == 99.0
    assert percentile(values, 1.0) == 100.0


def test_percentile_of_single_value():
    assert percentile([7.0], 0.5) == 7.0


def test_percentile_of_empty_is_zero():
    assert percentile([], 0.5) == 0.0


def test_percentile_rejects_out_of_range_p():
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile([1.0], 1.5)


def test_summarize_reports_count_p50_p99_and_total():
    out = summarize({"tier0": [1.0, 2.0, 3.0, 4.0]})
    assert out["tier0"]["count"] == 4
    assert out["tier0"]["p50"] == 2.0
    assert out["tier0"]["total_ms"] == 10.0


def test_summarize_handles_an_empty_series():
    out = summarize({"tier1": []})
    assert out["tier1"] == {"count": 0, "p50": 0.0, "p99": 0.0, "total_ms": 0.0}


def test_throughput_one_hour_of_audio_in_one_hour_is_one():
    assert throughput(3_600_000, 3_600_000.0) == pytest.approx(1.0)


def test_throughput_is_zero_when_no_time_elapsed():
    assert throughput(1000, 0.0) == 0.0
