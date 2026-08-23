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


def escalate(entries, segments, backend, store, delta_table, budget_usd):
    """Plan escalations, reserve their cost, and submit them. Returns job id.

    segments maps segment_id -> Segment. Real Segment objects are required,
    not stubs: Tier1Chirp.transcribe() reads segment.pcm.
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
        if e.segment_id in segments
    ]

    chosen = plan(candidates, budget_usd)
    if not chosen:
        return None

    # Reserve the whole batch atomically-per-call before anything is sent.
    for cand in chosen:
        store.reserve_spend(backend.name, cand.cost_usd)

    batch = [segments[c.segment_id] for c in chosen]

    job_id = backend.submit(batch)
    store.put_job(job_id, backend.name, backend.variant_key,
                  [s.segment_id for s in batch])
    return job_id
