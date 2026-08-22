"""Cost/quality frontier: the headline artifact.

Pricing lives in dhvani.backends.tier1_chirp.cost_for_duration_ms and is
imported, never reimplemented here. Fix round 2 (I1): this module used to
carry its own copy of the rate and price candidates at exact wall-clock,
while the backend rounded up to a whole billing increment — so the chart
people choose a budget from understated real spend by up to 7.5x.
"""

from dhvani.backends.tier1_chirp import cost_for_duration_ms
from dhvani.router import Candidate, delta_for, plan


def frontier(entries, delta_table: dict, budgets: list[float]) -> list[dict]:
    """Escalation behaviour across a sweep of budgets.

    router.plan() is a ratio-greedy heuristic (Task 6): it makes a single
    left-to-right pass over candidates sorted by delta/cost, so a larger
    budget can admit one expensive, high-ratio segment that "uses up"
    budget which would otherwise have admitted several cheaper,
    lower-ratio segments. Real segment costs are heterogeneous in
    production (durations vary 2-8s), so this is the expected shape of
    the data, not a corner case -- and it can make plan()'s own output
    (its escalated count, and in principle its total delta) look worse at
    a higher budget than at a lower one.

    That is never true of the underlying cost/quality tradeoff itself: at
    budget B a caller can always choose to spend less, so the best value
    achievable at B is by definition always >= the best value achievable
    at any budget b <= B. A dip is purely an artifact of the greedy
    heuristic, not a real property of the frontier -- and this report is
    the project's headline artifact, so a dip would misleadingly read as
    a bug to anyone looking at the chart.

    frontier() corrects for this with a monotone envelope: sweeping
    budgets in ascending order, each row reports the best selection seen
    so far (ranked by total_delta) at any budget <= its own, rather than
    necessarily plan()'s own output for that exact budget. budget_usd
    always reflects the row's own budget in the sweep; cost_usd always
    reflects the actual spend of the selection being reported (which can
    therefore be below budget_usd, including when an earlier, cheaper
    selection is being reported for a later, larger budget) -- spend is
    never fabricated to match the budget.
    """
    candidates = [
        Candidate(
            segment_id=e.segment_id,
            tier="tier1",
            risk=e.risk,
            cost_usd=cost_for_duration_ms(e.t_end_ms - e.t_start_ms),
            delta=delta_for(e.risk, "tier1", delta_table),
        )
        for e in entries
    ]

    rows = []
    best = None  # best selection seen so far, ranked by total_delta
    for budget in sorted(budgets):
        chosen = plan(candidates, budget)
        current = {
            "escalated": len(chosen),
            "cost_usd": sum(c.cost_usd for c in chosen),
            "mean_risk": (
                sum(c.risk for c in chosen) / len(chosen) if chosen else 0.0
            ),
            "total_delta": sum(c.delta for c in chosen),
        }
        if best is None or current["total_delta"] >= best["total_delta"]:
            best = current
        rows.append({"budget_usd": budget, **best})
    return rows


def render_markdown(rows: list[dict]) -> str:
    out = ["| budget ($) | escalated | spent ($) | mean risk | total delta |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['budget_usd']:.4f} | {r['escalated']} | "
            f"{r['cost_usd']:.4f} | {r['mean_risk']:.3f} | "
            f"{r['total_delta']:.2f} |"
        )
    return "\n".join(out)
