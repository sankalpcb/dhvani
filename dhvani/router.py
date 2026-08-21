"""Budget-constrained escalation policy.

Pure function, no I/O. This is the intellectual core of the system: given a
fixed spend per audio-hour, decide which segments deserve expensive treatment.
Greedy by delta/cost is the standard approximation to fractional knapsack.
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
    """Map a risk score to its decile bucket label."""
    idx = min(int(risk * N_BUCKETS), N_BUCKETS - 1)
    return f"{idx / N_BUCKETS:.1f}-{(idx + 1) / N_BUCKETS:.1f}"


def delta_for(risk: float, tier: str, delta_table: dict) -> float:
    """Measured expected improvement for this risk bucket and tier."""
    return float(delta_table.get(tier, {}).get(bucket_of(risk), 0.0))


def plan(candidates: list[Candidate], budget_usd: float) -> list[Candidate]:
    """Select escalations maximizing expected improvement within budget.

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
