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
