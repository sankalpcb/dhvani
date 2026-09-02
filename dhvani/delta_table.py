"""Measures expected toWER improvement per risk bucket, once, offline.

This is measurement, not training: no parameters are fitted, no model is
produced. The output is a lookup table committed to the repo.
"""

from collections import defaultdict

from dhvani.evaluator import to_wer
from dhvani.router import bucket_of


def build(rows: list[dict], tier: str = "tier1") -> dict:
    """rows: {risk, reference, tier0_text, <tier>_text} -> {tier: {bucket: delta}}

    Delta is in toWER *points* (percentage points), so a 0.25 -> 0.0 toWER
    improvement is 25.0. Negative deltas are preserved: Tier 1 genuinely loses
    on some segments, and the router (invariant I3) is what filters them out.

    `tier` names both the output key and the row field read as the improved
    text, so one function measures any tier. It defaults to "tier1" so every
    existing caller is unaffected.

    BASELINE. "before" is `before_text` when a row carries it, and
    `tier0_text` otherwise. That distinction is load-bearing for Tier 2,
    which repairs the BEST AVAILABLE hypothesis -- Tier 1's when Tier 1 ran,
    Tier 0's when it did not. Scoring Tier 2 against Tier 0 in the first case
    would hand Tier 2 the credit for Tier 1's improvement as well as its own,
    which is precisely how a cascade talks itself into believing a tier earns
    its keep. Tier 1 rows carry no before_text and keep scoring against
    Tier 0, which is what they actually improve on.
    """
    buckets = defaultdict(list)
    after_key = f"{tier}_text"
    for row in rows:
        before = to_wer(row["reference"], row.get("before_text", row["tier0_text"]))
        after = to_wer(row["reference"], row[after_key])
        buckets[bucket_of(row["risk"])].append((before - after) * 100.0)

    return {tier: {b: sum(v) / len(v) for b, v in sorted(buckets.items())}}
