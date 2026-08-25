"""Turn a caption track into a submitted escalation batch.

Spec §6: the router decides WHICH segments are worth expensive treatment
under a budget. This module turns that decision into a persisted job.

Spend is reserved for the entire batch BEFORE submit() is called, matching
the Phase 1 rule that money is accounted before the paid call, never after —
a crash between submission and accounting would otherwise under-count and
let the USD 20 ceiling be breached on restart.

FIX ROUND 3 (C2): candidates are priced with backend.cost_per_call(), not
with cost_for_duration_ms() called directly. Pricing unconditionally at the
live rate made a replay run reserve real dollars against the USD 20 ceiling
for calls that never happen. cost_per_call is already on the AsyncBackend
protocol, SyncAsyncAdapter delegates it, and Recorded.cost_per_call returns
0.0 in replay -- while Tier1Chirp.cost_per_call still routes through
cost_for_duration_ms(), so live pricing is unchanged and there is still
exactly one cost model.

FIX ROUND 4: the batch is deduplicated against work this backend has
already paid for or already has outstanding, BEFORE it is priced.

The bug this closes: escalate() was idempotent in the jobs table but not in
the spend ledger. backend.submit() is content-addressed (job_id_for hashes
variant_key plus the segment ids) and store.put_job() is INSERT OR IGNORE,
so re-running --escalate over an unfinished track registered no new work --
and charged for it again anyway, because the reservation happened before
anything knew the batch was a repeat. The operator loop for a track that
has not converged is exactly "--escalate --reconcile, again", so the
overcharge was per invocation with nothing bounding the total but the
ceiling itself.

The fix keys spend to the same thing work is keyed to: a segment is worth
paying for at (backend.name, backend.variant_key) exactly once. That is
checkable before submit(), so the money-before-the-paid-call ordering is
untouched, and so is the atomic whole-batch reservation below -- this
shrinks reserve_spend()'s input rather than restructuring the call. It is
also the rule pipeline.run() has always applied to Tier 0, which consults
store.get_hypothesis() before spending a Tier 0 call; Tier 1 simply never
had the equivalent.
"""

from dhvani.router import Candidate, delta_for, plan


def _outstanding(source_id, backend, store):
    """segment_ids this backend already has in flight for this source.

    Spend for these was reserved when their job was submitted. They are
    filtered by (tier, variant_key) for the same reason reconcile() filters
    open_jobs(): another backend's outstanding work says nothing about
    whether THIS backend has been paid to transcribe a segment.
    """
    return {
        segment_id
        for job in store.open_jobs(source_id)
        if job["tier"] == backend.name
        and job["variant_key"] == backend.variant_key
        for segment_id in job["segment_ids"]
    }


def escalate(source_id, entries, segments, backend, store, delta_table,
             budget_usd):
    """Plan escalations, reserve their cost, and submit them. Returns job id.

    Returns None when nothing new needs submitting -- including the case
    where every candidate is already delivered or already outstanding.
    Callers already treat None as "no batch went out" (it is what a zero
    budget and an empty delta table return), and no caller needs to tell
    those cases apart: cli.py rebuilds what it must poll from
    store.open_jobs(), not from this return value.

    segments maps segment_id -> Segment. Real Segment objects are required,
    not stubs: Tier1Chirp.transcribe() reads segment.pcm.

    source_id is threaded through to the jobs table (FIX ROUND 3, C1) so
    reconcile() can poll only its own source's jobs. Without it, one
    video's reconcile() pass settled every other video's outstanding job.
    """
    outstanding = _outstanding(source_id, backend, store)

    candidates = [
        Candidate(
            segment_id=e.segment_id,
            tier="tier1",
            risk=e.risk,
            cost_usd=backend.cost_per_call(segments[e.segment_id]),
            delta=delta_for(e.risk, "tier1", delta_table),
        )
        for e in entries
        # Intentional silent exclusion: an entry with no matching Segment
        # has no real pcm, and Tier1Chirp.transcribe() requires it -- such
        # an entry cannot be escalated at all, so it is dropped here,
        # before the router ever sees it, rather than raised.
        if e.segment_id in segments
        # Already in flight: its cost is on the ledger and its result is
        # coming. Re-submitting it is what double-charged (FIX ROUND 4).
        and e.segment_id not in outstanding
        # Already delivered: same backend, same variant, same audio, so a
        # second paid call returns the same text. Keyed on the backend's
        # own (name, variant_key) rather than on segment_id alone -- a
        # Tier 0 hypothesis exists for every segment the pipeline has
        # transcribed, and treating that as Tier 1 work already done would
        # switch escalation off entirely.
        and store.get_hypothesis(e.segment_id, backend.name,
                                 variant_key=backend.variant_key) is None
    ]

    chosen = plan(candidates, budget_usd)
    if not chosen:
        return None

    # Reserve the batch's summed cost in ONE atomic call before submitting,
    # not once per candidate. A per-candidate loop can partially succeed
    # and then raise on a later candidate, leaving spend reserved for a
    # batch that was never submitted anywhere -- that money buys nothing
    # and cannot be recovered. A single reserve_spend() call makes the
    # failure atomic: if it raises, nothing was reserved and nothing was
    # submitted.
    total_cost = sum(cand.cost_usd for cand in chosen)
    store.reserve_spend(backend.name, total_cost)

    batch = [segments[c.segment_id] for c in chosen]

    job_id = backend.submit(batch)
    registered = store.put_job(job_id, backend.name, backend.variant_key,
                               [s.segment_id for s in batch], source_id)
    if not registered:
        # put_job() is INSERT OR IGNORE and returns False when the row was
        # already there. Since the batch was deduplicated above, every
        # segment in it has no hypothesis and no open job -- so the only
        # way this id can already exist is a job reconcile() dead-lettered
        # after MAX_JOB_ATTEMPTS, whose results never arrived. Its row is
        # still state="failed", open_jobs() will never surface it, and
        # reconcile() will never poll it -- while the reservation above has
        # already been made and, in live mode, the call really went out.
        #
        # Ignoring this return value was the difference between paying for
        # a retry and paying for nothing. Reopen the row so the batch just
        # submitted is actually collectable. attempts is deliberately NOT
        # reset: the job resumes above MAX_JOB_ATTEMPTS, so it gets exactly
        # one more poll before being dead-lettered again, and an operator
        # who wants further attempts pays for each one explicitly.
        store.set_job_state(job_id, "running")
    return job_id
