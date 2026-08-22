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


def test_greedy_is_knowingly_suboptimal_with_heterogeneous_costs():
    """Document a known limitation: ratio-greedy on 0/1 knapsack is suboptimal.

    With heterogeneous costs (e.g., A costs 6x what B costs), greedy can leave
    budget on the table. This test pins that behavior as accepted, not a bug:
    in practice, segment costs are near-uniform (proportional to duration),
    so this pathological case does not arise. A workload with highly
    heterogeneous costs should use exact DP instead.

    Counterexample:
    - A: cost=6, delta=6.6, ratio=1.10 (greedy picks this)
    - B: cost=5, delta=5.4, ratio=1.08
    - C: cost=5, delta=5.4, ratio=1.08
    - Budget=10

    Greedy selects only A for total delta 6.6, wasting $4 of budget.
    Optimal selection is B+C at exactly $10 for total delta 10.8 (64% better).
    """
    a = c("a", delta=6.6, cost=6.0)
    b = c("b", delta=5.4, cost=5.0)
    c_cand = c("c", delta=5.4, cost=5.0)
    chosen = plan([a, b, c_cand], budget_usd=10.0)
    # Greedy picks only A, not the optimal B+C
    assert len(chosen) == 1
    assert chosen[0].segment_id == "a"
    total_delta = sum(x.delta for x in chosen)
    assert total_delta == 6.6  # Suboptimal; optimal is 10.8


def test_bucket_of_handles_negative_risk():
    """bucket_of clamps negative risk to [0, 1] range."""
    assert bucket_of(-0.5) == "0.0-0.1"
    assert bucket_of(-0.1) == "0.0-0.1"


def test_bucket_of_handles_risk_above_one():
    """bucket_of clamps risk > 1 to [0, 1] range."""
    assert bucket_of(1.5) == "0.9-1.0"
    assert bucket_of(2.0) == "0.9-1.0"
