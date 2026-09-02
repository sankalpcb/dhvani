import pytest
from dhvani.delta_table import build


def test_empty_input_gives_empty_table():
    assert build([]) == {"tier1": {}}


def test_measures_improvement_per_bucket():
    rows = [{
        "risk": 0.65,
        "reference": "alpha beta gamma delta",
        "tier0_text": "alpha beta gamma WRONG",
        "tier1_text": "alpha beta gamma delta",
    }]
    table = build(rows)
    # tier0 toWER = 0.25 (25 points), tier1 = 0.0 -> delta = 25.0
    assert table["tier1"]["0.6-0.7"] == pytest.approx(25.0)


def test_negative_delta_is_preserved_not_clamped():
    """Tier 1 genuinely loses on some segments; the router filters, not the table."""
    rows = [{
        "risk": 0.15,
        "reference": "alpha beta",
        "tier0_text": "alpha beta",
        "tier1_text": "alpha WRONG",
    }]
    assert build(rows)["tier1"]["0.1-0.2"] < 0.0


def test_averages_within_a_bucket():
    rows = [
        {"risk": 0.65, "reference": "a b c d", "tier0_text": "a b c X", "tier1_text": "a b c d"},
        {"risk": 0.66, "reference": "a b c d", "tier0_text": "a b c d", "tier1_text": "a b c d"},
    ]
    assert build(rows)["tier1"]["0.6-0.7"] == pytest.approx(12.5)


# --- generalizing build() to a second tier (M5) ---

def test_build_can_measure_a_named_tier():
    rows = [{"risk": 0.05, "reference": "a b c",
             "tier0_text": "a b c", "tier2_text": "a b c"}]
    assert set(build(rows, tier="tier2")) == {"tier2"}


def test_the_default_tier_is_unchanged():
    """Existing callers pass no tier and must keep getting tier1."""
    rows = [{"risk": 0.05, "reference": "a b c",
             "tier0_text": "a x c", "tier1_text": "a b c"}]
    assert set(build(rows)) == {"tier1"}


def test_a_tier_is_scored_against_what_it_actually_repaired():
    """Tier 2 repairs the BEST available hypothesis, so its baseline is
    Tier 1's text when Tier 1 ran -- not Tier 0's.

    Scoring tier2 against tier0 when it actually improved on tier1 would
    credit Tier 2 with Tier 1's gain, which is how a cascade convinces
    itself a tier is worth more than it is.
    """
    rows = [{
        "risk": 0.05,
        "reference": "one two three",
        "tier0_text": "wrong wrong wrong",   # Tier 0 was terrible
        "before_text": "one two wrong",      # ...but Tier 1 already fixed most of it
        "tier2_text": "one two three",       # Tier 2 fixed the last token
    }]
    delta = build(rows, tier="tier2")["tier2"]["0.0-0.1"]

    # Against before_text: 1 of 3 tokens fixed -> ~33 points, not ~100.
    assert 20.0 < delta < 45.0, f"scored against the wrong baseline: {delta}"


def test_before_text_defaults_to_tier0():
    rows = [{"risk": 0.05, "reference": "a b", "tier0_text": "a x",
             "tier2_text": "a b"}]
    assert build(rows, tier="tier2")["tier2"]["0.0-0.1"] > 0.0
