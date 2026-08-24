"""Calibration harness: measures the delta table the router needs.

Spec: docs/superpowers/specs/2026-08-24-dhvani-calibration-design.md

The router cannot pick its own calibration set — it selects by delta, and
delta is what is being measured (spec §1.1). So calibration escalates a
STRATIFIED sample across every risk bucket, deliberately including low-risk
ones, because that is the only way to discover the negative deltas that
invariant I3 exists to filter.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import date

import numpy as np

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

DEFAULT_PCM_CACHE = "calibration_pcm"
"""Where phase 1 parks the audio phase 2 needs.

Calibration is deliberately two processes: collect is slow, free and local;
escalate is fast, paid and remote, and an operator inspects the histogram in
between. That split means the escalate process no longer holds the corpus
audio in memory, and Tier1Chirp.transcribe() sends segment.pcm INLINE (see
its recognize_pcm call) — there is no GCS object for it to read instead.

So phase 1 writes each normalized segment's PCM here as
<cache_dir>/<segment_id>.npy (numpy.save of the mono int16 array) and phase
2 loads it back. Content-addressed like everything else: segment_id is the
SHA256 of exactly these bytes, so a stale file is impossible — a changed
array is a changed name.
"""


class PcmCacheMiss(RuntimeError):
    """Phase 2 needed a segment's audio and phase 1 had not cached it."""


def pcm_cache_path(cache_dir: str, segment_id: str) -> str:
    return os.path.join(cache_dir, f"{segment_id}.npy")


def save_pcm(cache_dir: str, segment_id: str, pcm) -> str:
    """Write one segment's normalized PCM, unless it is already there.

    Skipping an existing file keeps collect() resumable at the same cost as
    its hypothesis cache: a killed multi-hour run restarts without rewriting
    gigabytes it already has.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = pcm_cache_path(cache_dir, segment_id)
    if not os.path.exists(path):
        np.save(path, pcm)
    return path


def load_pcm(cache_dir: str, segment_id: str):
    """Read one segment's PCM back, or raise naming what is missing.

    Hard error by design. The alternatives are both worse and both were
    live in this file's history: skipping the segment silently shrinks the
    measured sample without saying so, and substituting a zero array sends
    silence to a paid transcriber, which returns "" and poisons every delta
    in the bucket while the table still looks plausible.
    """
    path = pcm_cache_path(cache_dir, segment_id)
    if not os.path.exists(path):
        raise PcmCacheMiss(
            f"no cached PCM for segment {segment_id}: expected {path}. "
            f"Phase 2 sends the audio inline to Tier 1, so it cannot proceed "
            f"without it. Re-run `dhvani-calibrate collect` with "
            f"--pcm-cache {cache_dir}."
        )
    return np.load(path)


class LazySegments:
    """segment_id -> Segment, reading PCM from the cache on demand.

    Deliberately lazy rather than a dict comprehension over `selected`.
    escalate_selected() skips segments with no reference or no Tier 0
    hypothesis and never touches their audio, and it reuses cached Tier 1
    hypotheses without re-transcribing — so eagerly loading every selected
    segment's PCM would both hold the whole sample in memory at once and
    turn a missing file for a segment that was never going to be escalated
    into a fatal error. A miss is fatal only when the audio is actually
    needed.
    """

    def __init__(self, selected, cache_dir: str):
        self._duration_ms = {s["segment_id"]: s["duration_ms"] for s in selected}
        self._cache_dir = cache_dir

    def __contains__(self, segment_id: str) -> bool:
        return segment_id in self._duration_ms

    def __len__(self) -> int:
        return len(self._duration_ms)

    def __getitem__(self, segment_id: str) -> Segment:
        return Segment(segment_id, 0, self._duration_ms[segment_id],
                       load_pcm(self._cache_dir, segment_id))


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


def collect(corpus, tier0, store, langs, per_lang: int,
            pcm_cache_dir: str) -> list[dict]:
    """Phase 1: transcribe and score a corpus locally. Slow, free, resumable.

    One utterance is one segment (spec §1.2), so the segmenter is bypassed
    and every segment keeps its own reference. Already-transcribed segments
    are read from the store rather than re-run — that is what lets a
    multi-hour run be killed and restarted without losing work.

    pcm_cache_dir is required, not optional: phase 2 has no other source for
    the audio (see DEFAULT_PCM_CACHE), so a collect run that quietly skipped
    caching would produce a scored.json that cannot be escalated.
    """
    scored: list[dict] = []
    seen: set[str] = set()

    for lang in langs:
        for item in corpus.stream(lang, limit=per_lang):
            save_pcm(pcm_cache_dir, item.segment_id, item.pcm)

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

            if item.segment_id in seen:
                # R9: segment_id is content-addressed, so N byte-identical
                # utterances are ONE segment with one reference and one
                # hypothesis — but appending N scored entries would let that
                # single segment push a bucket over MIN_BUCKET_SAMPLES by
                # itself and would weight its delta N times in build()'s
                # mean. The store rows above are INSERT OR IGNORE and are
                # already idempotent; this makes the scored list match them.
                continue
            seen.add(item.segment_id)

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

    Spend is NOT reserved here. `tier1` is a Recorded wrapper (that is what
    the CLI passes), and Recorded.transcribe() atomically reserves against
    the USD 20 ceiling immediately before the paid call. Reserving here as
    well double-charged every call — a nominal $1.00 call reserved $2.00 —
    which made --confirm understate the real spend by half and inflated
    meta.spend_usd.

    Recorded is also the only layer that knows whether a call is actually
    paid: in replay mode it returns the fixture without reserving anything,
    and cost_per_call() is 0.0. A reservation one level up cannot see that
    distinction, which is exactly the class of bug fix round 3 (C2) had to
    patch out of dhvani/cli.py.

    A cached Tier 1 hypothesis is still reused without calling the backend
    at all, so re-running a calibration pass re-charges nothing.

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
            # cost is recorded against the hypothesis for attribution only;
            # the reservation that enforces the ceiling happens inside
            # tier1.transcribe() (Recorded). See the docstring.
            cost = tier1.cost_per_call(segment)
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

    I4: MIN_BUCKET_SAMPLES is re-applied HERE, against the surviving row
    count. stratify() applies it to the SCORED population, but rows are
    dropped afterwards inside escalate_selected() (no reference, no Tier 0
    hypothesis), and build() averages whatever it is handed. A bucket with
    22 scored segments that loses 19 to skipping used to publish a
    3-sample mean as a measured value — precisely the "escalate wrongly,
    and pay for it" outcome the floor exists to prevent. Omission degrades
    to "do not escalate"; a noisy average degrades to spending money.
    """
    table = build_delta_table(rows)

    bucket_n: dict[str, int] = defaultdict(int)
    for row in rows:
        bucket_n[bucket_of(row["risk"])] += 1

    dropped = sorted(b for b in table["tier1"] if bucket_n[b] < MIN_BUCKET_SAMPLES)
    if dropped:
        table = {"tier1": {b: v for b, v in table["tier1"].items()
                           if b not in set(dropped)}}
        print(f"dropped {len(dropped)} bucket(s) below the "
              f"{MIN_BUCKET_SAMPLES}-sample floor after skipping: "
              + ", ".join(f"{b} (n={bucket_n[b]})" for b in dropped),
              file=sys.stderr)

    payload = dict(table)
    payload["meta"] = {
        "policy_id": POLICY_ID,
        "risk_weights": dict(RISK_WEIGHTS),
        # bucket_n keeps the PRE-drop counts on purpose: it is the evidence
        # for why a bucket named in dropped_buckets is absent from the table.
        "bucket_n": dict(sorted(bucket_n.items())),
        "dropped_buckets": dropped,
        "min_bucket_samples": MIN_BUCKET_SAMPLES,
        "languages": list(langs),
        "segments_escalated": len(selected),
        "spend_usd": round(spend_usd, 6),
        "measured_at": date.today().isoformat(),
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return table
