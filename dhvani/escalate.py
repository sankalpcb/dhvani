"""Turn a caption track into a submitted escalation batch.

Spec §6: the router decides WHICH segments are worth expensive treatment
under a budget. This module turns that decision into a persisted job.

Spend is reserved for the entire batch BEFORE submit() is called, matching
the Phase 1 rule that money is accounted before the paid call, never after —
a crash between submission and accounting would otherwise under-count and
let the USD 20 ceiling be breached on restart.
"""

from dhvani.backends.tier1_chirp import cost_for_duration_ms
from dhvani.router import Candidate, delta_for, plan


def _duration_ms(segment) -> int:
    return segment.t_end_ms - segment.t_start_ms


def escalate(source_id, entries, segments, backend, store, delta_table,
             budget_usd):
    """Plan escalations, reserve their cost, and submit them. Returns job id.

    segments maps segment_id -> Segment. Real Segment objects are required,
    not stubs: Tier1Chirp.transcribe() reads segment.pcm.

    source_id is threaded through to the jobs table (FIX ROUND 3, C1) so
    reconcile() can poll only its own source's jobs. Without it, one
    video's reconcile() pass settled every other video's outstanding job.
    """
    candidates = [
        Candidate(
            segment_id=e.segment_id,
            tier="tier1",
            risk=e.risk,
            cost_usd=cost_for_duration_ms(_duration_ms(segments[e.segment_id])),
            delta=delta_for(e.risk, "tier1", delta_table),
        )
        for e in entries
        # Intentional silent exclusion: an entry with no matching Segment
        # has no real pcm, and Tier1Chirp.transcribe() requires it -- such
        # an entry cannot be escalated at all, so it is dropped here,
        # before the router ever sees it, rather than raised.
        if e.segment_id in segments
    ]

    chosen = plan(candidates, budget_usd)
    if not chosen:
        return None

    # Reserve the batch's summed cost in ONE atomic call before submitting,
    # not once per candidate. A per-candidate loop can partially succeed
    # and then raise on a later candidate, leaving spend reserved for a
    # batch that was never submitted anywhere -- that money buys nothing
    # and cannot be recovered (unlike the deliberate double-reservation on
    # resubmit, where the spend maps to a real, submitted job). A single
    # reserve_spend() call makes the failure atomic: if it raises, nothing
    # was reserved and nothing was submitted.
    total_cost = sum(cand.cost_usd for cand in chosen)
    store.reserve_spend(backend.name, total_cost)

    batch = [segments[c.segment_id] for c in chosen]

    job_id = backend.submit(batch)
    store.put_job(job_id, backend.name, backend.variant_key,
                  [s.segment_id for s in batch], source_id)
    return job_id
