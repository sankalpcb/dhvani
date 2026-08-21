"""Budget-constrained escalation policy.

Pure function, no I/O. This is the intellectual core of the system: given a
fixed spend per audio-hour, decide which segments deserve expensive treatment.

This solves the 0/1 knapsack problem using a ratio-greedy heuristic (delta/cost).
Greedy has no optimality guarantee for 0/1 knapsack in general, but is chosen here
because segment costs are near-uniform in this domain (proportional to duration,
2–8s), where it performs well. A workload with highly heterogeneous costs would
warrant exact dynamic programming. See test_greedy_is_knowingly_suboptimal_with_heterogeneous_costs
for a concrete counterexample with artificial costs.
"""

from dataclasses import dataclass

N_BUCKETS = 10


@dataclass(frozen=True)
class Candidate:
    segment_id: str
    tier: str
    risk: float
    cost_usd: float
    delta: float  # expected toWER points reduced, from the measured delta table


def bucket_of(risk: float) -> str:
    """Map a risk score to its decile bucket label.

    Clamps risk to [0, 1] range to ensure valid bucket labels.
    E.g., bucket_of(-0.5) -> "0.0-0.1", bucket_of(1.5) -> "0.9-1.0"
    """
    clamped_risk = max(0.0, min(risk, 1.0))
    idx = int(clamped_risk * N_BUCKETS)
    idx = min(idx, N_BUCKETS - 1)
    return f"{idx / N_BUCKETS:.1f}-{(idx + 1) / N_BUCKETS:.1f}"


def delta_for(risk: float, tier: str, delta_table: dict) -> float:
    """Measured expected improvement for this risk bucket and tier."""
    return float(delta_table.get(tier, {}).get(bucket_of(risk), 0.0))


def plan(candidates: list[Candidate], budget_usd: float) -> list[Candidate]:
    """Select escalations maximizing expected improvement within budget.

    Uses ratio-greedy (delta/cost) heuristic. This is a known suboptimal algorithm
    for 0/1 knapsack; see test_greedy_is_knowingly_suboptimal_with_heterogeneous_costs
    for a concrete counterexample. Accepted here because segment costs are near-uniform
    in practice, where greedy performs well.

    Invariant I3: candidates with delta <= 0 are never selected.
    Invariant I4: total selected cost never exceeds budget_usd.
    Invariant I5: ties break on segment_id, so output is order-independent.
    """
    eligible = [c for c in candidates if c.delta > 0.0 and c.cost_usd > 0.0]
    eligible.sort(key=lambda c: (-(c.delta / c.cost_usd), c.segment_id, c.tier))

    chosen: list[Candidate] = []
    spent = 0.0
    for cand in eligible:
        if spent + cand.cost_usd <= budget_usd + 1e-9:
            chosen.append(cand)
            spent += cand.cost_usd
    return chosen
