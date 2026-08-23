"""Poll outstanding escalation jobs and fold their results into a new track.

Spec §7 step 3-4: results arrive late, out of order, possibly partial,
possibly twice. This module is the only place that advances a track version.

A job is polled at most once per reconcile() pass. A job still pending is
left open; a job that returns results is merged and the track version is
bumped. Merging is delegated to track.merge_entries(), which is pure and
idempotent, so a duplicate delivery cannot corrupt the track.

A job is marked done only when its returned results cover every segment_id
it registered (job["segment_ids"]). A batch that returns results for only
some of its segments is a partial delivery: the job is left running so a
later reconcile() pass polls it again and the still-missing segments get
another chance to arrive. Settling the job on any non-empty result would
strand those segments forever -- the track could then never converge to
the state a synchronous run would produce (invariant I1).

put_track() is INSERT OR IGNORE: on a (source_id, version) collision -- a
second writer landed that version first -- it returns False and writes
nothing. A job is marked done only after that write is confirmed to have
landed. Losing the race and marking the job done anyway would make its
result unrecoverable (it exists nowhere durable, and open_jobs() would
never surface the job again to retry it), which is exactly the I1
violation this module exists to prevent.
"""

from dhvani.config import POLICY_ID
from dhvani.scorer import extract, risk as compute_risk
from dhvani.track import entries_from_json, entries_to_json, merge_entries


def reconcile(source_id: str, backend, store) -> int:
    """Merge any completed jobs for this backend. Returns the latest version."""
    version = store.latest_track_version(source_id)
    if version == 0:
        return 0

    current = store.get_track(source_id, version)
    entries = entries_from_json(current["content_json"])
    # The track already knows every segment's bounds, so durations need not be
    # threaded through the caller.
    durations = {e.segment_id: e.t_end_ms - e.t_start_ms for e in entries}

    updates: dict[str, dict] = {}
    settled: list[str] = []

    for job in store.open_jobs():
        if job["tier"] != backend.name or job["variant_key"] != backend.variant_key:
            continue

        store.bump_job_attempts(job["job_id"])
        results = backend.poll(job["job_id"])
        if results is None:
            store.set_job_state(job["job_id"], "running")
            continue

        for segment_id, result in results.items():
            duration = durations.get(segment_id, 0)
            features = extract(result["text"], result.get("signals", {}), duration)
            updates[segment_id] = {
                "text": result["text"],
                "risk": compute_risk(features),
            }
            store.put_hypothesis(
                segment_id, backend.name, result["text"],
                result.get("signals", {}), 0.0, backend.variant_key,
            )

        if set(results.keys()) >= set(job["segment_ids"]):
            settled.append(job["job_id"])
        else:
            # Partial delivery: some registered segment_ids did not come
            # back in this batch. Leave the job running so open_jobs()
            # still returns it and a later pass retries the missing ones.
            store.set_job_state(job["job_id"], "running")

    if updates:
        merged = merge_entries(entries, updates)
        new_version = version + 1
        wrote = store.put_track(source_id, new_version, POLICY_ID,
                                entries_to_json(merged), current["cost_usd"])
        if not wrote:
            # Lost the race: another writer already committed source_id's
            # next version between our read of latest_track_version() and
            # this put_track() call. INSERT OR IGNORE means our merge was
            # never written -- it exists only in local variables now, not
            # in the store. Claiming new_version, or marking any settled
            # job done, would make that merge permanently unrecoverable.
            #
            # Fail safe: report the version actually in the store, and
            # leave every job we would have settled as running so the next
            # reconcile() pass re-polls (results are memoized on the
            # backend -- no extra spend) and re-merges against whatever the
            # other writer landed.
            for job_id in settled:
                store.set_job_state(job_id, "running")
            return store.latest_track_version(source_id)
    else:
        new_version = version

    for job_id in settled:
        store.set_job_state(job_id, "done")

    return new_version
