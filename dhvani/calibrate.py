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
from dhvani.quota import QuotaExhausted, RateLimited
from dhvani.router import bucket_of
from dhvani.scorer import extract, risk as compute_risk
from dhvani.segmenter import Segment
from dhvani.backends.base import FixtureMissing
from dhvani.store import BudgetExceeded

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


class PcmCacheError(RuntimeError):
    """Phase 2 could not get a segment's audio out of the PCM cache.

    A common base so cli_calibrate can turn every way the cache can fail
    into one message and exit 3, rather than catching each subclass by name
    and letting the next one through as a traceback.
    """


class PcmCacheMiss(PcmCacheError):
    """Phase 2 needed a segment's audio and phase 1 had not cached it."""


class PcmCacheCorrupt(PcmCacheError):
    """The file is there but numpy cannot read an array out of it.

    Distinct from PcmCacheMiss because the remedy is the opposite one. A
    miss is repaired by re-running collect. A truncated file is NOT:
    save_pcm() skips any path that already exists -- that skip is what
    makes a killed multi-hour collect resumable -- so collect steps over
    the wreckage and phase 2 fails again identically. The file has to be
    deleted first, and the error has to say so or the operator loops.
    """


class NoMeasuredBuckets(RuntimeError):
    """A calibration run produced nothing worth publishing.

    I7: write_table had no guard on len(rows) and the CLI wrote
    unconditionally and returned 0, so a run in which every segment was
    skipped emitted {"tier1": {}} and reported success. The router reads
    delta_table.json as measurement; an empty or floor-less one is the
    absence of measurement, and must not be mistaken for "Tier 1 never
    helps". Refusing to write leaves any previous, real table in place.
    """


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
    try:
        return np.load(path)
    except Exception as exc:
        # A collect run killed mid-np.save leaves a partial file. numpy
        # reports that inconsistently -- EOFError for an empty file,
        # ValueError for a short one -- and for a file truncated inside the
        # magic string it reports "This file contains pickled (object)
        # data" and suggests allow_pickle=True, which is wrong here and
        # unsafe in general. Raise one diagnostic that names the segment,
        # names the file, and gives the remedy that actually works.
        raise PcmCacheCorrupt(
            f"cached PCM for segment {segment_id} is unreadable: {path} "
            f"({type(exc).__name__}: {exc}). A collect run killed partway "
            f"through np.save leaves a file like this. Re-running "
            f"`dhvani-calibrate collect` will NOT repair it -- save_pcm() "
            f"skips paths that already exist, which is what makes collect "
            f"resumable -- so delete {path} (or pass a fresh --pcm-cache) "
            f"and re-run collect to rewrite it."
        ) from exc


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


def histogram(scored: list[dict]) -> dict[str, int]:
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
    speakers_taken: set = set()
    for bucket in sorted(by_bucket):
        members = by_bucket[bucket]
        if len(members) < MIN_BUCKET_SAMPLES:
            continue
        members.sort(key=lambda i: i["segment_id"])

        # Spec §9.4: speaker-disjoint. One speaker contributing many
        # utterances lets the risk function learn speaker identity as a
        # difficulty proxy, and then "all reported numbers are fiction".
        # Enforced HERE rather than upstream because this is the only point
        # that knows what was actually selected -- the thing disjoint_by's
        # docstring says it guards ("the SELECTED set, not merely what was
        # requested").
        #
        # The floor check above deliberately reads len(members), not the
        # post-filter count: MIN_BUCKET_SAMPLES asks whether the bucket has
        # enough evidence to publish, and dropping duplicate speakers from a
        # qualifying bucket does not retract that.
        #
        # A None speaker_id is not a collision (matching disjoint_by), which
        # also keeps this backward compatible with the committed
        # results/scored.json, written before collect() recorded the field.
        taken_here = 0
        for item in members:
            if taken_here >= n_per_bucket:
                break
            speaker = item.get("speaker_id")
            if speaker is not None:
                if speaker in speakers_taken:
                    continue
                speakers_taken.add(speaker)
            chosen.append(item)
            taken_here += 1
    return chosen


def district_spread(selected: list[dict]) -> dict[str, int]:
    """How many selected samples fall in each district.

    A DIAGNOSTIC, deliberately not a gate. Spec §9.4 asks for
    district-disjointness alongside speaker-disjointness, but that is
    unachievable by pigeonhole at calibration scale: IndicVoices spans 145
    districts and a standard run selects ~150 samples, so at least one
    district must repeat. Enforcing it would make calibration unrunnable
    rather than more honest.

    What it is good for is spotting concentration -- a table built almost
    entirely from two districts is a result worth qualifying, and this is
    the number that says so.
    """
    spread: dict[str, int] = {}
    for row in selected:
        district = row.get("district")
        if district is not None:
            spread[district] = spread.get(district, 0) + 1
    return spread


def calibration_source(dataset_id: str, lang: str) -> str:
    """The segments.source_id a calibration run records.

    Provenance for a human reading the store, not a key: nothing queries
    segments by source_id, so this is kept readable rather than sanitized
    or hashed the way ids.source_id has to be.

    It used to be "calib:{lang}", which said which language a row was but
    not which corpus produced it -- so a second dataset collected into the
    same --db left rows indistinguishable from the first's.

    Note the limit of this, which is inherent rather than an oversight: two
    corpora containing byte-identical audio produce ONE segment row, since
    segment_id is content-addressed and put_segment is INSERT OR IGNORE.
    That row keeps whichever dataset reached it first, and that is correct
    -- it really is one segment, with one reference and one hypothesis.
    """
    return f"calib:{dataset_id}:{lang}"


def collect(corpus, make_tier0, store, langs, per_lang: int,
            pcm_cache_dir: str, max_per_speaker: int | None = None) -> list[dict]:
    """Phase 1: transcribe and score a corpus locally. Slow, free, resumable.

    One utterance is one segment (spec §1.2), so the segmenter is bypassed
    and every segment keeps its own reference. Already-transcribed segments
    are read from the store rather than re-run — that is what lets a
    multi-hour run be killed and restarted without losing work.

    I6: `make_tier0` is a factory taking a language code, NOT a single
    backend instance. It used to be one instance, built once by the CLI
    with Tier0Conformer's default lang="hi" and then reused for every
    language in `langs` — so Kannada and Malayalam audio was decoded as
    Hindi (two thirds of the corpus scored on garbage), and because
    variant_key was identical across languages the store could not even
    tell the three runs apart.

    pcm_cache_dir is required, not optional: phase 2 has no other source for
    the audio (see DEFAULT_PCM_CACHE), so a collect run that quietly skipped
    caching would produce a scored.json that cannot be escalated.

    corpus.dataset_id is recorded into each segment's source_id via
    calibration_source(), so collecting a second dataset into one --db does
    not leave rows that cannot be told apart.
    """
    scored: list[dict] = []
    seen: set[str] = set()

    for lang in langs:
        tier0 = make_tier0(lang)
        for item in corpus.stream(lang, limit=per_lang,
                                  max_per_speaker=max_per_speaker):
            save_pcm(pcm_cache_dir, item.segment_id, item.pcm)

            store.put_segment(item.segment_id,
                              calibration_source(corpus.dataset_id, lang), 0,
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
                # Spec §9.4 metadata, carried so stratify() can enforce
                # speaker-disjointness on what it SELECTS. Both are already
                # persisted by put_reference() above, but that store row is
                # not what selection reads -- and recording them there while
                # omitting them here is precisely why disjoint_by() sat
                # unused: the data existed and never reached the point that
                # needed it.
                "speaker_id": item.speaker_id,
                "district": item.district,
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


MAX_TIER1_ATTEMPTS = 3
"""How many times one segment's Tier 1 call is attempted before giving up.

A live run died at call 45 of 124 on a single `503 502:Bad Gateway`:
escalate_selected had no retry at all, so one transient cloud hiccup
abandoned the 79 segments behind it. reconcile() has carried per-job
isolation and MAX_JOB_ATTEMPTS for exactly this since Phase 2 -- the
synchronous calibration path, which is the one that spends money, had
nothing.

Bounded rather than generous on purpose: every attempt goes through
Recorded, which reserves spend BEFORE the call, so attempts cost money
whether or not the provider ultimately bills for a failed request. That
over-reservation is the intended direction -- an overstated cost fails safe
against the ceiling, an understated one breaches it -- but it is a reason to
retry a few times, not many.
"""

TIER1_BACKOFF_SEC = 2.0
"""Base delay between attempts, doubling each time.

A tight retry loop against a struggling backend is its own denial of
service, and would burn the attempt budget in milliseconds -- which is no
help at all against an outage measured in seconds.
"""


def _transcribe_with_retry(tier1, segment, attempts: int = MAX_TIER1_ATTEMPTS):
    """tier1.transcribe(), retried on transient failure.

    BudgetExceeded and PcmCacheError are re-raised immediately. Both are
    terminal, not transient: the ceiling will not move between attempts and
    the audio will not appear, so retrying either would only reserve more
    spend against a limit that just refused, or stall on a file that is not
    coming back.
    """
    import time

    for attempt in range(1, attempts + 1):
        try:
            return tier1.transcribe(segment)
        except (BudgetExceeded, PcmCacheError):
            raise
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(TIER1_BACKOFF_SEC * (2 ** (attempt - 1)))


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
    at all, so re-running a calibration pass re-charges nothing. That is what
    makes an aborted run cheap to resume, and it is load-bearing: a live run
    really did die partway and was resumed for the price of its remaining
    segments.

    Transient failures are retried (see MAX_TIER1_ATTEMPTS). Each attempt
    goes through Recorded and therefore reserves spend again, so a segment
    that fails twice before succeeding is charged three times. That is the
    same deliberate over-reservation the rest of this path uses -- an
    overstated cost fails safe against the ceiling, an understated one
    breaches it -- and it is why the attempt budget is small.

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
            result = _transcribe_with_retry(tier1, segment)
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


def write_table(rows, selected, path: str, spend_usd: float, langs,
                tier: str = "tier1") -> dict:
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
    if not rows:
        raise NoMeasuredBuckets(
            f"no rows to measure: all {len(selected)} selected segments were "
            f"skipped for lack of a reference or a Tier 0 hypothesis. "
            f"Refusing to write {path} -- an empty table is indistinguishable "
            f"from 'Tier 1 never helps'."
        )

    table = build_delta_table(rows, tier=tier)

    bucket_n: dict[str, int] = defaultdict(int)
    for row in rows:
        bucket_n[bucket_of(row["risk"])] += 1

    dropped = sorted(b for b in table[tier] if bucket_n[b] < MIN_BUCKET_SAMPLES)
    if dropped:
        table = {tier: {b: v for b, v in table[tier].items()
                        if b not in set(dropped)}}
        print(f"dropped {len(dropped)} bucket(s) below the "
              f"{MIN_BUCKET_SAMPLES}-sample floor after skipping: "
              + ", ".join(f"{b} (n={bucket_n[b]})" for b in dropped),
              file=sys.stderr)

    if not table[tier]:
        raise NoMeasuredBuckets(
            f"no risk bucket reached the {MIN_BUCKET_SAMPLES}-sample floor "
            f"from {len(rows)} row(s): "
            + ", ".join(f"{b} (n={n})" for b, n in sorted(bucket_n.items()))
            + f". Refusing to write {path}."
        )

    payload = dict(table)
    payload["meta"] = {
        # Which tier this table measures. Recorded because a table with a
        # "tier2" key and no note of how it was produced is one rename away
        # from being read as a Tier 1 measurement.
        "tier": tier,
        "policy_id": POLICY_ID,
        "risk_weights": dict(RISK_WEIGHTS),
        # bucket_n keeps the PRE-drop counts on purpose: it is the evidence
        # for why a bucket named in dropped_buckets is absent from the table.
        "bucket_n": dict(sorted(bucket_n.items())),
        "dropped_buckets": dropped,
        "min_bucket_samples": MIN_BUCKET_SAMPLES,
        "languages": list(langs),
        # I8: this was `segments_escalated: len(selected)`, which counted
        # what was CHOSEN, not what survived — so it could read 812 beside
        # an empty tier1 map. The two numbers are now separate, and their
        # gap is the skip count.
        "segments_selected": len(selected),
        "segments_escalated": len(rows),
        "spend_usd": round(spend_usd, 6),
        "measured_at": date.today().isoformat(),
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return table


def _default_sleep(seconds):
    import time
    time.sleep(seconds)


PACING_WAIT_SEC = 2.0
"""How long to wait for the token bucket to refill before retrying a call.

Not a backoff: the bucket refills at a known rate, so this is simply the
interval at which a batch re-offers work it was told was too early.
"""

MAX_TIER2_ATTEMPTS = 4
"""Attempts per segment against a transient vendor refusal.

Four rather than three: the free tier's 429 is a per-MINUTE pacing refusal
(measured limit: 15 requests/minute for gemini-3.5-flash-lite), so the
backoff below has to be able to outlast a whole window.
"""

TIER2_BACKOFF_SEC = 8.0
"""Base delay, doubling. 8 -> 16 -> 32 covers a 60-second quota window.

Larger than Tier 1's 2.0s because that retries a server hiccup, while this
waits out a fixed-window quota that will not clear early no matter how
politely it is asked.
"""

MAX_PACING_WAITS = 60
"""Bound on consecutive waits for ONE segment.

Without it a misconfigured bucket -- capacity smaller than one call, say --
would spin forever rather than failing. Sixty waits at PACING_WAIT_SEC is
two minutes on a single segment, which is far longer than any real refill
and short enough that a stuck run is noticed.
"""


def repair_selected(selected, tier2, store, gate, hypotheses: dict,
                    segments_by_id, tier1_variant: str = "",
                    tier0_variant: str = "", sleep=None) -> list[dict]:
    """Phase 3: run Tier 2 over the sample and assemble rows for build().

    The Tier 1 analogue is escalate_selected(); two differences matter.

    THE SCARCE RESOURCE IS QUOTA, NOT MONEY. There is no ceiling to breach
    and no ledger to overstate -- there is a daily allowance that refills.
    So exhaustion STOPS the pass and returns the rows already earned rather
    than raising: a partial measurement is worth keeping, and the next run
    resumes for free because every repair is cached by (segment_id, tier,
    variant_key). Tier 1's path raises on BudgetExceeded because spending
    past a ceiling is a different kind of wrong from running out of a
    daily allowance.

    THE BASELINE IS WHAT TIER 2 ACTUALLY REPAIRED. `before_text` is Tier 1's
    hypothesis when one exists at `tier1_variant`, and Tier 0's otherwise --
    because that is the text handed to the model. Scoring against Tier 0
    when Tier 1 already ran would credit Tier 2 with Tier 1's gain on top of
    its own. `tier0_text` is recorded regardless so a reader can see the
    whole chain rather than just the last hop.

    `hypotheses` is the dict the backend's hypothesis_source reads. It is
    filled here, immediately before each call, because only this function
    knows which upstream text a given segment should be repaired from.
    """
    rows: list[dict] = []
    outcome_deferred: list[str] = []

    for item in selected:
        segment_id = item["segment_id"]
        reference = store.get_reference(segment_id)
        tier0 = store.get_hypothesis(segment_id, "tier0",
                                     item.get("tier0_variant", tier0_variant))
        if reference is None or tier0 is None:
            # No ground truth or no Tier 0 output means no meaningful delta.
            continue

        before_text = tier0["text"]
        if tier1_variant:
            tier1 = store.get_hypothesis(segment_id, "tier1", tier1_variant)
            if tier1 is not None:
                before_text = tier1["text"]

        cached = store.get_hypothesis(segment_id, tier2.name, tier2.variant_key)
        if cached is None:
            # The two limits are handled differently HERE too, and the
            # distinction is the whole reason they are separate exceptions.
            #
            # RateLimited is soft: nothing was consumed, the call merely came
            # too soon, and the bucket refills in seconds. Breaking the batch
            # on it silently measures fewer segments than were selected --
            # which is how a bucket slips under MIN_BUCKET_SAMPLES and a run
            # reports having measured nothing. So it waits and re-offers.
            #
            # QuotaExhausted is hard: the daily allowance does not refill
            # until tomorrow, so waiting is pointless and stopping is right.
            granted = False
            for _ in range(MAX_PACING_WAITS + 1):
                try:
                    gate.acquire()
                    granted = True
                    break
                except RateLimited:
                    (sleep or _default_sleep)(PACING_WAIT_SEC)
                except QuotaExhausted:
                    break
            if not granted:
                break
            hypotheses[segment_id] = before_text

            # The vendor's own refusal must not end the run. A live pass died
            # at segment 30 of 100 on a 429 that propagated out of
            # transcribe() -- our local counter said there was quota left,
            # and Google disagreed. dhvani/repair.py (the pipeline path)
            # already degraded on backend errors; this path did not, so the
            # same lesson had to be learned twice.
            #
            # Retried rather than immediately deferred, because 429 here is
            # a PACING refusal that clears in seconds -- Google returns a
            # retryDelay with it -- and a calibration run wants all 100
            # segments, not a partial table it has to explain.
            result = None
            for attempt in range(1, MAX_TIER2_ATTEMPTS + 1):
                try:
                    result = tier2.transcribe(segments_by_id[segment_id])
                    break
                except FixtureMissing:
                    # Replay never falls back to live, and a missing fixture
                    # is a hard error by design: degrading here would let an
                    # offline mistake masquerade as "nothing to measure".
                    raise
                except Exception:
                    if attempt == MAX_TIER2_ATTEMPTS:
                        break
                    (sleep or _default_sleep)(
                        TIER2_BACKOFF_SEC * (2 ** (attempt - 1)))
            if result is None:
                outcome_deferred.append(segment_id)
                continue

            store.put_hypothesis(segment_id, tier2.name, result["text"],
                                 result.get("signals", {}), 0.0,
                                 tier2.variant_key)
            tier2_text = result["text"]
        else:
            tier2_text = cached["text"]

        rows.append({
            "risk": item["risk"],
            "reference": reference["reference"],
            "tier0_text": tier0["text"],
            "before_text": before_text,
            "tier2_text": tier2_text,
        })

    return rows
