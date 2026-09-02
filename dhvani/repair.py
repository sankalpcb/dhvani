"""Tier 2 repair: select, gate on quota, repair, and degrade gracefully.

Design §3. M5's demo is graceful degradation at quota exhaustion, so the
contract that matters most is what this does when the quota runs out
mid-run: it stops calling, reports what it deferred, and RETURNS. It does
not raise, and the caller ships its captions either way.

No jobs row, deliberately -- a simplification found while implementing.
The design sketched exhausted work as an outstanding job reconciled later,
mirroring Tier 1. But Tier 1 needs that machinery because a Chirp batch is
in flight at the vendor with a handle only the jobs table remembers.
Nothing is in flight here: a Gemini call either returned or never happened.
"Still wants repair" is therefore already derivable -- positive tier2
delta, no tier2 hypothesis -- and because hypotheses are content-addressed,
the next run recomputes exactly the remaining set and resumes. Adding a
jobs row would add a second, weaker source of truth for a fact the
hypothesis cache already holds.

Invariant I3 applies unchanged: router.plan() excludes non-positive delta,
so a measured finding that repair makes things worse disables the tier by
itself, exactly as the negative Tier 1 deltas already do.
"""

from dataclasses import dataclass, field, replace

from dhvani.backends.base import FixtureMissing
from dhvani.quota import QuotaExhausted, RateLimited
from dhvani.router import Candidate, delta_for, plan


@dataclass
class RepairOutcome:
    """What one repair pass managed to do.

    `deferred` is not an error list. A deferred segment keeps its upstream
    text, ships now, and is retried on the next run -- so callers report it
    rather than failing on it.
    """

    repaired: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    reason: str | None = None

    @property
    def degraded(self) -> bool:
        return bool(self.deferred)


def repair(source_id, entries, segments, backend, store, delta_table, gate,
           budget_usd: float = 0.0):
    """Repair what the delta table says is worth repairing and quota allows.

    Returns a RepairOutcome. Never raises on exhaustion: that is a normal
    operating condition, not a failure.

    budget_usd defaults to 0.0 because Tier 2 is free -- cost_per_call() is
    0.0, and router._ratio() already ranks a zero-cost positive-delta
    candidate ahead of everything priced (FIX ROUND 3, C2), so a zero
    budget still admits every free candidate. The parameter exists so a
    caller can share one budget across tiers later without this signature
    changing.
    """
    candidates = [
        Candidate(
            segment_id=e.segment_id,
            tier="tier2",
            risk=e.risk,
            cost_usd=backend.cost_per_call(segments[e.segment_id]),
            delta=delta_for(e.risk, "tier2", delta_table),
        )
        for e in entries
        if e.segment_id in segments
        # Already repaired at this exact model+prompt: repeating the call
        # returns the same text and costs a request from a budget of 1,000
        # a day. Keyed on the backend's variant_key so a prompt change
        # correctly re-repairs rather than reading a stale result.
        and store.get_hypothesis(e.segment_id, backend.name,
                                 variant_key=backend.variant_key) is None
    ]

    outcome = RepairOutcome()
    chosen = plan(candidates, budget_usd)

    for cand in chosen:
        try:
            gate.acquire()
        except QuotaExhausted:
            # Hard stop for today. Everything from here on is deferred,
            # including this candidate.
            outcome.deferred.extend(
                c.segment_id for c in chosen[chosen.index(cand):]
            )
            outcome.reason = "quota_exhausted"
            break
        except RateLimited:
            # Soft. Nothing was consumed at the vendor or in the ledger,
            # so this segment is simply retried on a later pass rather
            # than abandoning the rest of the batch.
            outcome.deferred.append(cand.segment_id)
            outcome.reason = outcome.reason or "rate_limited"
            continue

        try:
            result = backend.transcribe(segments[cand.segment_id])
        except FixtureMissing:
            # Deliberately NOT degraded. Replay never falls back to live,
            # and a missing fixture is a hard error by design -- swallowing
            # it here would turn an offline mistake into a silent no-op and
            # let a run claim it repaired nothing when it was never able to
            # try. This is the one failure that must still abort.
            raise
        except Exception:
            # Everything else degrades. The local cap is a GUESS -- the
            # account's real requests-per-day is not published anywhere
            # readable (see quota.quota_day and the 2026-09-02 spike), so
            # authorizing a call Google then refuses is an expected
            # condition, not a bug. Same reasoning reconcile() uses to wrap
            # each poll(): one failing unit must not abandon the pass.
            outcome.deferred.append(cand.segment_id)
            outcome.reason = outcome.reason or "backend_error"
            continue

        store.put_hypothesis(
            cand.segment_id, backend.name, result["text"],
            result.get("signals", {}), 0.0, variant_key=backend.variant_key,
        )
        outcome.repaired.append(cand.segment_id)

    return outcome


def mark_unrepaired(entries, deferred):
    """Flag entries whose repair was deferred (design §5.2).

    Returns a new list; TrackEntry is frozen. The band is untouched on
    purpose -- see the field's comment in pipeline.py.
    """
    wanted = set(deferred)
    return [
        replace(e, repair_unavailable=True) if e.segment_id in wanted else e
        for e in entries
    ]
