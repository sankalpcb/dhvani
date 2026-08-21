import pytest
from dhvani.router import Candidate, plan, bucket_of, delta_for


def c(sid, delta, cost=0.01, risk=0.5, tier="tier1"):
    return Candidate(segment_id=sid, tier=tier, risk=risk, cost_usd=cost, delta=delta)


def test_zero_budget_escalates_nothing():
    """Graceful degradation: B=0 still produces valid output."""
    assert plan([c("a", 10.0), c("b", 5.0)], budget_usd=0.0) == []


def test_large_budget_escalates_everything_positive():
    chosen = plan([c("a", 10.0), c("b", 5.0)], budget_usd=1000.0)
    assert {x.segment_id for x in chosen} == {"a", "b"}


def test_never_escalates_non_positive_delta():
    """Invariant I3: no negative-value escalation."""
    chosen = plan([c("a", 0.0), c("b", -3.0), c("c", 1.0)], budget_usd=1000.0)
    assert [x.segment_id for x in chosen] == ["c"]


def test_respects_budget():
    """Invariant I4."""
    cands = [c(str(i), delta=1.0, cost=0.01) for i in range(100)]
    chosen = plan(cands, budget_usd=0.05)
    assert sum(x.cost_usd for x in chosen) <= 0.05 + 1e-9
    assert len(chosen) == 5


def test_prefers_higher_delta_per_cost():
    cheap_good = c("cheap", delta=10.0, cost=0.01)   # ratio 1000
    dear_good = c("dear", delta=20.0, cost=1.00)     # ratio 20
    chosen = plan([dear_good, cheap_good], budget_usd=0.01)
    assert [x.segment_id for x in chosen] == ["cheap"]


def test_is_deterministic_under_ties():
    """Invariant I5: ties break by segment_id, never by input order."""
    a = [c("b", 1.0), c("a", 1.0)]
    b = [c("a", 1.0), c("b", 1.0)]
    assert plan(a, 0.01) == plan(b, 0.01)


def test_bucket_of_partitions_unit_interval():
    assert bucket_of(0.0) == "0.0-0.1"
    assert bucket_of(0.65) == "0.6-0.7"
    assert bucket_of(1.0) == "0.9-1.0"


def test_delta_for_reads_table_and_defaults_to_zero():
    table = {"tier1": {"0.6-0.7": 18.2}}
    assert delta_for(0.65, "tier1", table) == 18.2
    assert delta_for(0.05, "tier1", table) == 0.0
    assert delta_for(0.65, "tier2", table) == 0.0
