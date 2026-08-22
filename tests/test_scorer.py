import pytest
from dhvani.config import RISK_WEIGHTS, TAU_FLAG
from dhvani.scorer import Features, extract, risk

ZERO = Features(0.0, 0.0, 0.0, 0.0, 0.0)

# Signals the model actually supplies. mean_neg_logprob is not among them:
# IndicConformer does not expose it, so Tier0Conformer hardcodes 0.0.
LIVE_FEATURES = [
    "ctc_rnnt_disagreement", "script_mix_entropy",
    "romanization_smell", "short_segment",
]


def test_all_zero_features_give_zero_risk():
    assert risk(ZERO) == 0.0


def test_all_max_features_give_risk_one():
    assert risk(Features(1.0, 1.0, 1.0, 1.0, 1.0)) == pytest.approx(1.0)


def test_risk_is_bounded():
    assert 0.0 <= risk(Features(2.0, 2.0, 2.0, 2.0, 2.0)) <= 1.0
    assert 0.0 <= risk(Features(-1.0, -1.0, 0.0, 0.0, 0.0)) <= 1.0


def test_risk_is_monotonic_in_each_live_feature():
    """Fix round 2, minor 6: this used to loop over all five features.
    mean_neg_logprob now carries weight 0.0 (the model does not expose it,
    so it is permanently 0.0 in production and its 0.25 was silently
    capping reachable risk at 0.75), so it is deliberately NOT monotonic
    and is asserted inert below instead. Every feature the model actually
    supplies must still move the score."""
    base = risk(ZERO)
    for name in LIVE_FEATURES:
        vals = {f: 0.0 for f in Features.__dataclass_fields__}
        vals[name] = 1.0
        assert risk(Features(**vals)) > base, f"{name} is not monotonic"


# --- Fix round 2, minor 6: the review band must be reachable ---

def test_reachable_maximum_risk_is_one():
    """THE regression guard for minor 6.

    mean_neg_logprob is permanently 0.0 in production (IndicConformer does
    not expose it) and used to carry weight 0.25, so the maximum risk any
    real segment could score was 0.75 -- only 0.10 above TAU_FLAG (0.65),
    and reachable only with every other signal simultaneously at its
    absolute maximum. Spec 6.2's primary output contract ("nothing below
    tau_flag ships silently") was effectively defeated.

    With the unavailable signal forced to 0.0, as production always has
    it, and every live signal at 1.0, risk must reach a full 1.0.
    """
    reachable_max = risk(Features(
        ctc_rnnt_disagreement=1.0,
        mean_neg_logprob=0.0,   # as production always has it
        script_mix_entropy=1.0,
        romanization_smell=1.0,
        short_segment=1.0,
    ))
    assert reachable_max == pytest.approx(1.0)
    assert reachable_max > TAU_FLAG, "the review band must be reachable"


def test_unavailable_signal_carries_no_weight():
    """Pinned explicitly so the dead signal cannot silently reacquire
    weight while it is still unavailable. If the blocked spike (spec 14)
    later exposes a real per-token logprob, this test failing is the
    correct signal to re-derive the weights."""
    assert RISK_WEIGHTS["mean_neg_logprob"] == 0.0
    assert risk(Features(0.0, 1.0, 0.0, 0.0, 0.0)) == 0.0


def test_risk_weights_sum_to_exactly_one():
    assert sum(RISK_WEIGHTS.values()) == 1.0


def test_live_weights_kept_their_relative_proportions():
    """The 0.25 was redistributed proportionally, not reassigned by taste:
    every live signal keeps its share relative to the others."""
    old = {"ctc_rnnt_disagreement": 0.35, "script_mix_entropy": 0.20,
           "romanization_smell": 0.15, "short_segment": 0.05}
    for name in LIVE_FEATURES:
        assert RISK_WEIGHTS[name] == pytest.approx(old[name] / sum(old.values()))


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
