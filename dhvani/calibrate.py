"""Calibration harness: measures the delta table the router needs.

Spec: docs/superpowers/specs/2026-08-24-dhvani-calibration-design.md

The router cannot pick its own calibration set — it selects by delta, and
delta is what is being measured (spec §1.1). So calibration escalates a
STRATIFIED sample across every risk bucket, deliberately including low-risk
ones, because that is the only way to discover the negative deltas that
invariant I3 exists to filter.
"""

import json
from collections import defaultdict
from datetime import date

from dhvani.backends.tier1_chirp import cost_for_duration_ms
from dhvani.config import POLICY_ID, RISK_WEIGHTS
from dhvani.delta_table import build as build_delta_table
from dhvani.router import bucket_of
from dhvani.scorer import extract, risk as compute_risk
from dhvani.segmenter import Segment

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


def collect(corpus, tier0, store, langs, per_lang: int) -> list[dict]:
    """Phase 1: transcribe and score a corpus locally. Slow, free, resumable.

    One utterance is one segment (spec §1.2), so the segmenter is bypassed
    and every segment keeps its own reference. Already-transcribed segments
    are read from the store rather than re-run — that is what lets a
    multi-hour run be killed and restarted without losing work.
    """
    scored: list[dict] = []

    for lang in langs:
        for item in corpus.stream(lang, limit=per_lang):
            store.put_segment(item.segment_id, f"calib:{lang}", 0,
                              item.duration_ms, lang)
            store.put_reference(item.segment_id, item.reference, lang,
                                item.speaker_id, item.district)

            cached = store.get_hypothesis(item.segment_id, "tier0", tier0.variant_key)
            if cached is None:
                segment = Segment(item.segment_id, 0, item.duration_ms, item.pcm)
                result = tier0.transcribe(segment)
                store.put_hypothesis(item.segment_id, "tier0", result["text"],
                                     result["signals"], 0.0, tier0.variant_key)
            else:
                result = {"text": cached["text"], "signals": cached["signals"]}

            features = extract(result["text"], result["signals"], item.duration_ms)
            scored.append({
                "segment_id": item.segment_id,
                "risk": compute_risk(features),
                "lang": lang,
                "duration_ms": item.duration_ms,
                # The Tier 0 cache key this hypothesis was stored under.
                # Persisted rather than re-derived in phase 2: the two
                # phases are separate processes, possibly days apart, and
                # reconstructing a backend in the escalate branch to ask
                # for its variant_key would silently return the WRONG key
                # after a language or model change between them. Phase 2
                # must look up what phase 1 actually wrote.
                "tier0_variant": tier0.variant_key,
            })

    return scored


def estimate_cost(selected: list[dict]) -> float:
    """Pre-flight estimate, printed before the first paid call.

    Prices through cost_for_duration_ms — the single Tier 1 cost model — so
    the estimate cannot drift from what is actually reserved.
    """
    return sum(cost_for_duration_ms(s["duration_ms"]) for s in selected)


def escalate_selected(selected, tier1, store, segments_by_id,
                      tier0_variant: str = "") -> list[dict]:
    """Phase 2: run Tier 1 over the stratified sample and assemble rows.

    Spend is reserved BEFORE each paid call, and a cached Tier 1 hypothesis
    is reused without reserving again — re-running a calibration pass must
    not re-charge for work already done.

    The Tier 0 cache key is read PER ITEM from the scored record collect()
    wrote (`tier0_variant`), falling back to the `tier0_variant` argument
    only for callers that predate that field. A single key for the whole
    batch cannot be right once collect() runs one backend per language:
    "lang=hi;model_id=..." and "lang=kn;model_id=..." are different cache
    entries, and looking either up under the other's key returns None,
    which this function reads as "no Tier 0 output" and silently skips.
    """
    rows: list[dict] = []

    for item in selected:
        segment_id = item["segment_id"]
        reference = store.get_reference(segment_id)
        tier0 = store.get_hypothesis(segment_id, "tier0",
                                     item.get("tier0_variant", tier0_variant))
        if reference is None or tier0 is None:
            # No ground truth or no Tier 0 output means no meaningful delta.
            continue

        cached = store.get_hypothesis(segment_id, "tier1", tier1.variant_key)
        if cached is None:
            segment = segments_by_id[segment_id]
            cost = tier1.cost_per_call(segment)
            store.reserve_spend(tier1.name, cost)
            result = tier1.transcribe(segment)
            store.put_hypothesis(segment_id, "tier1", result["text"],
                                 result.get("signals", {}), cost, tier1.variant_key)
            tier1_text = result["text"]
        else:
            tier1_text = cached["text"]

        rows.append({
            "risk": item["risk"],
            "reference": reference["reference"],
            "tier0_text": tier0["text"],
            "tier1_text": tier1_text,
        })

    return rows


def write_table(rows, selected, path: str, spend_usd: float, langs) -> dict:
    """Build the delta table and write it with provenance.

    build()'s contract is untouched; meta is additive. Nothing enforces meta
    (spec non-goal N3), but a stale table becomes visible rather than silent.
    """
    table = build_delta_table(rows)

    bucket_n: dict[str, int] = defaultdict(int)
    for row in rows:
        bucket_n[bucket_of(row["risk"])] += 1

    payload = dict(table)
    payload["meta"] = {
        "policy_id": POLICY_ID,
        "risk_weights": dict(RISK_WEIGHTS),
        "bucket_n": dict(sorted(bucket_n.items())),
        "languages": list(langs),
        "segments_escalated": len(selected),
        "spend_usd": round(spend_usd, 6),
        "measured_at": date.today().isoformat(),
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return table
