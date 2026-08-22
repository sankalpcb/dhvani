import pytest
from dhvani.pipeline import TrackEntry
from dhvani.report import TIER1_USD_PER_MIN, frontier
from dhvani.router import Candidate, plan


def _entry(sid, risk, duration_ms=3000):
    return TrackEntry(sid, 0, duration_ms, "text", risk, "marked")


def test_frontier_total_delta_is_monotonic_in_budget():
    """The headline chart must never show less value at a higher budget.

    router.plan() is a ratio-greedy heuristic that makes a single
    left-to-right pass over candidates sorted by delta/cost. With
    heterogeneous costs -- the expected shape of real segment durations
    (2-8s) -- a larger budget can admit one expensive, high-ratio segment
    that blocks several cheaper, lower-ratio segments which fit at a
    smaller budget, so plan()'s own escalated *count* can drop as budget
    rises (pinned directly below in
    test_plan_escalated_count_can_drop_at_higher_budget). frontier()
    corrects for this at the report layer with a monotone envelope, so
    total_delta -- the actual value metric -- must never decrease across
    an ascending budget sweep.

    Uses the reviewer's exact counterexample: one segment with cost=10,
    delta=100 (ratio 10) vs. five segments each with cost=1, delta=5
    (ratio 5).
    """
    big_cost = 10.0
    small_cost = 1.0
    big_duration_ms = round(big_cost / TIER1_USD_PER_MIN * 60000)
    small_duration_ms = round(small_cost / TIER1_USD_PER_MIN * 60000)

    entries = [
        _entry("big", 0.95, big_duration_ms),
        *[_entry(f"small{i}", 0.65, small_duration_ms) for i in range(5)],
    ]
    table = {"tier1": {"0.9-1.0": 100.0, "0.6-0.7": 5.0}}

    rows = frontier(entries, table, budgets=[0.0, small_cost * 5, big_cost])

    totals = [r["total_delta"] for r in rows]
    assert totals == sorted(totals)
    # Sanity: the envelope's cost/budget invariant still holds throughout.
    for row in rows:
        assert row["cost_usd"] <= row["budget_usd"] + 1e-9


def test_plan_escalated_count_can_drop_at_higher_budget():
    """Pins a known heuristic limitation, in the same spirit as
    test_greedy_is_knowingly_suboptimal_with_heterogeneous_costs in
    tests/test_router.py: router.plan()'s fixed left-to-right greedy pass
    can select FEWER segments at a higher budget than at a lower one,
    because a single expensive, high-ratio candidate can consume budget
    that would otherwise admit several cheaper, lower-ratio candidates.

    This is documented and accepted, not fixed in router.plan() itself
    (Task 6 already pins plan()'s knapsack suboptimality as accepted
    behavior); frontier() compensates for it at the report layer with a
    monotone envelope on total_delta (see
    test_frontier_total_delta_is_monotonic_in_budget above). This test
    keeps the underlying plan() behavior visible rather than silently
    hidden by that fix.
    """
    big = Candidate(segment_id="big", tier="tier1", risk=0.95, cost_usd=10.0, delta=100.0)
    smalls = [
        Candidate(segment_id=f"small{i}", tier="tier1", risk=0.65, cost_usd=1.0, delta=5.0)
        for i in range(5)
    ]
    candidates = [big, *smalls]

    at_five = plan(candidates, budget_usd=5.0)
    at_ten = plan(candidates, budget_usd=10.0)

    assert len(at_five) == 5   # all five small items fit within budget=5
    assert len(at_ten) == 1    # only the big item fits at budget=10 -- count DROPS
    assert sum(c.delta for c in at_five) == pytest.approx(25.0)
    assert sum(c.delta for c in at_ten) == pytest.approx(100.0)


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
