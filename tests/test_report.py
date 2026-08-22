import math

import numpy as np
import pytest

import dhvani.report as report_mod
from dhvani.backends.tier1_chirp import (
    BILLING_INCREMENT_SEC, Tier1Chirp, cost_for_duration_ms,
)
from dhvani.pipeline import TrackEntry
from dhvani.report import frontier
from dhvani.router import Candidate, plan
from dhvani.segmenter import Segment


def _entry(sid, risk, duration_ms=3000):
    return TrackEntry(sid, 0, duration_ms, "text", risk, "marked")


def _duration_costing_at_least(target_usd: float) -> int:
    """Smallest whole number of billing increments priced >= target_usd.

    Tier 1 is billed in whole BILLING_INCREMENT_SEC blocks, so an
    arbitrary dollar figure is generally not an achievable segment price.
    Tests that want "a segment costing about $10" must therefore pick a
    duration and read its real price back out of the pricing function,
    never divide a dollar figure by a rate.
    """
    increment_ms = BILLING_INCREMENT_SEC * 1000
    per_increment = cost_for_duration_ms(increment_ms)
    return increment_ms * math.ceil(target_usd / per_increment)


# --- Fix round 2, I1: exactly one Tier 1 cost model in the codebase ---

@pytest.mark.parametrize("duration_ms", [2000, 3000, 8000, 15000, 60000])
def test_frontier_prices_a_segment_exactly_as_the_backend_bills_it(duration_ms):
    """The frontier is the artifact used to choose a budget, so its prices
    must be the prices that will actually be billed.

    report.py used to reimplement Tier 1 pricing as exact wall-clock
    (rate * duration / 60000), while Tier1Chirp.cost_per_call rounds up to
    a whole BILLING_INCREMENT_SEC block -- understating real spend by up
    to 7.5x on a 2000 ms segment ($0.000100 reported vs $0.000750
    billed). A run planned from the chart would then hit BudgetExceeded
    partway through.

    Budgeting exactly the backend's own price for one segment must admit
    exactly that segment and report exactly that spend.
    """
    backend_cost = Tier1Chirp(client=object()).cost_per_call(
        Segment(segment_id="c" * 64, t_start_ms=0, t_end_ms=duration_ms,
                pcm=np.zeros(4, dtype=np.int16))
    )
    rows = frontier(
        [_entry("s0", 0.65, duration_ms)],
        {"tier1": {"0.6-0.7": 18.0}},
        budgets=[backend_cost],
    )
    assert rows[0]["escalated"] == 1, (
        "the frontier priced this segment above what the backend bills, "
        "so a budget equal to the real price did not admit it"
    )
    assert rows[0]["cost_usd"] == pytest.approx(backend_cost)


def test_report_defines_no_tier1_rate_of_its_own():
    """There must be exactly one rate definition in the codebase. A second
    copy in report.py is what let the two cost models drift apart."""
    assert not hasattr(report_mod, "TIER1_USD_PER_MIN"), (
        "report.py must import Tier 1 pricing, not redefine the rate"
    )


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

    Uses the reviewer's exact counterexample: one segment with cost~=10,
    delta=100 (ratio ~10) vs. five segments each with cost~=1, delta=5
    (ratio ~5). The costs are read back out of the real pricing function
    rather than assumed, because Tier 1 bills in whole 15s blocks and
    exact round-dollar segment prices do not exist (fix round 2, I1).
    """
    big_duration_ms = _duration_costing_at_least(10.0)
    small_duration_ms = _duration_costing_at_least(1.0)
    big_cost = cost_for_duration_ms(big_duration_ms)
    small_cost = cost_for_duration_ms(small_duration_ms)
    assert big_cost / 100.0 < small_cost / 5.0, "big must have the better ratio"

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
