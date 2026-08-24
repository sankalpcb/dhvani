"""Calibration harness: measures the delta table the router needs.

Spec: docs/superpowers/specs/2026-08-24-dhvani-calibration-design.md

The router cannot pick its own calibration set — it selects by delta, and
delta is what is being measured (spec §1.1). So calibration escalates a
STRATIFIED sample across every risk bucket, deliberately including low-risk
ones, because that is the only way to discover the negative deltas that
invariant I3 exists to filter.
"""

from collections import defaultdict

from dhvani.router import bucket_of

# A bucket with fewer samples than this is OMITTED from the table rather
# than included with a noisy average. Omission degrades to "do not
# escalate"; a noisy average degrades to "escalate wrongly, and pay".
MIN_BUCKET_SAMPLES = 20

N_PER_BUCKET = 100


def histogram(scored: list[dict]) -> dict:
    """Bucket label -> count. Printed before any paid call so the risk
    distribution is visible while it is still free to act on."""
    counts: dict[str, int] = defaultdict(int)
    for item in scored:
        counts[bucket_of(item["risk"])] += 1
    return dict(sorted(counts.items()))


def stratify(scored: list[dict], n_per_bucket: int = N_PER_BUCKET) -> list[dict]:
    """Sample up to n_per_bucket from each risk bucket. Pure and deterministic.

    Selection sorts by segment_id, so a re-run picks the same segments and
    hits the content-addressed cache instead of paying again.
    """
    by_bucket: dict[str, list] = defaultdict(list)
    for item in scored:
        by_bucket[bucket_of(item["risk"])].append(item)

    chosen: list[dict] = []
    for bucket in sorted(by_bucket):
        members = by_bucket[bucket]
        if len(members) < MIN_BUCKET_SAMPLES:
            continue
        members.sort(key=lambda i: i["segment_id"])
        chosen.extend(members[:n_per_bucket])
    return chosen
