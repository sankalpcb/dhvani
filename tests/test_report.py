import pytest
from dhvani.pipeline import TrackEntry
from dhvani.report import frontier


def _entry(sid, risk):
    return TrackEntry(sid, 0, 3000, "text", risk, "marked")


def test_frontier_is_monotonic_in_budget():
    entries = [_entry(f"s{i}", 0.65) for i in range(10)]
    table = {"tier1": {"0.6-0.7": 18.0}}
    rows = frontier(entries, table, budgets=[0.0, 0.001, 0.01, 1.0])
    counts = [r["escalated"] for r in rows]
    assert counts == sorted(counts)


def test_zero_budget_escalates_nothing():
    entries = [_entry("s0", 0.65)]
    rows = frontier(entries, {"tier1": {"0.6-0.7": 18.0}}, budgets=[0.0])
    assert rows[0]["escalated"] == 0
    assert rows[0]["cost_usd"] == 0.0


def test_cost_never_exceeds_budget():
    entries = [_entry(f"s{i}", 0.65) for i in range(50)]
    table = {"tier1": {"0.6-0.7": 18.0}}
    for row in frontier(entries, table, budgets=[0.0, 0.002, 0.02]):
        assert row["cost_usd"] <= row["budget_usd"] + 1e-9
