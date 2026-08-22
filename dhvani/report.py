"""Cost/quality frontier: the headline artifact."""

from dhvani.router import Candidate, delta_for, plan

TIER1_USD_PER_MIN = 0.003


def frontier(entries, delta_table: dict, budgets: list[float]) -> list[dict]:
    """Escalation behaviour across a sweep of budgets."""
    candidates = [
        Candidate(
            segment_id=e.segment_id,
            tier="tier1",
            risk=e.risk,
            cost_usd=TIER1_USD_PER_MIN * (e.t_end_ms - e.t_start_ms) / 60000.0,
            delta=delta_for(e.risk, "tier1", delta_table),
        )
        for e in entries
    ]

    rows = []
    for budget in sorted(budgets):
        chosen = plan(candidates, budget)
        rows.append({
            "budget_usd": budget,
            "escalated": len(chosen),
            "cost_usd": sum(c.cost_usd for c in chosen),
            "mean_risk": (
                sum(c.risk for c in chosen) / len(chosen) if chosen else 0.0
            ),
        })
    return rows


def render_markdown(rows: list[dict]) -> str:
    out = ["| budget ($) | escalated | spent ($) | mean risk |",
           "|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['budget_usd']:.4f} | {r['escalated']} | "
            f"{r['cost_usd']:.4f} | {r['mean_risk']:.3f} |"
        )
    return "\n".join(out)
