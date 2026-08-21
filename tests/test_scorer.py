import pytest
from dhvani.scorer import Features, extract, risk

ZERO = Features(0.0, 0.0, 0.0, 0.0, 0.0)


def test_all_zero_features_give_zero_risk():
    assert risk(ZERO) == 0.0


def test_all_max_features_give_risk_one():
    assert risk(Features(1.0, 1.0, 1.0, 1.0, 1.0)) == pytest.approx(1.0)


def test_risk_is_bounded():
    assert 0.0 <= risk(Features(2.0, 2.0, 2.0, 2.0, 2.0)) <= 1.0
    assert 0.0 <= risk(Features(-1.0, -1.0, 0.0, 0.0, 0.0)) <= 1.0


def test_risk_is_monotonic_in_each_feature():
    base = risk(ZERO)
    for i in range(5):
        vals = [0.0] * 5
        vals[i] = 1.0
        assert risk(Features(*vals)) > base, f"feature {i} is not monotonic"


def test_risk_is_deterministic():
    f = Features(0.3, 0.4, 0.5, 0.2, 0.0)
    assert risk(f) == risk(f)


def test_extract_flags_short_segments():
    assert extract("hello", {}, duration_ms=800).short_segment == 1.0
    assert extract("hello", {}, duration_ms=3000).short_segment == 0.0


def test_extract_reads_script_entropy_from_text():
    f = extract("अआइई abcd", {}, duration_ms=3000)
    assert f.script_mix_entropy == pytest.approx(1.0, abs=0.01)


def test_extract_normalizes_missing_decoder_signals_to_zero():
    f = extract("hello world", {}, duration_ms=3000)
    assert f.ctc_rnnt_disagreement == 0.0
    assert f.mean_neg_logprob == 0.0
