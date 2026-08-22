"""Measures expected toWER improvement per risk bucket, once, offline.

This is measurement, not training: no parameters are fitted, no model is
produced. The output is a lookup table committed to the repo.
"""

from collections import defaultdict

from dhvani.evaluator import to_wer
from dhvani.router import bucket_of


def build(rows: list[dict]) -> dict:
    """rows: {risk, reference, tier0_text, tier1_text} -> {tier: {bucket: delta}}

    Delta is in toWER *points* (percentage points), so a 0.25 -> 0.0 toWER
    improvement is 25.0. Negative deltas are preserved: Tier 1 genuinely loses
    on some segments, and the router (invariant I3) is what filters them out.
    """
    buckets = defaultdict(list)
    for row in rows:
        before = to_wer(row["reference"], row["tier0_text"])
        after = to_wer(row["reference"], row["tier1_text"])
        buckets[bucket_of(row["risk"])].append((before - after) * 100.0)

    return {"tier1": {b: sum(v) / len(v) for b, v in sorted(buckets.items())}}
