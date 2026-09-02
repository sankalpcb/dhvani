"""Calibration CLI: dhvani-calibrate collect | escalate

Phase 1 (collect) is slow, free and local. Phase 2 (escalate) is fast, paid
and remote, and refuses to spend without --confirm.

Goal G5: importing this module must not require any ML dep, cloud SDK, or
`datasets` package, and must touch no network or credentials. The imports
of Tier0Conformer and IndicVoicesCorpus (collect) and of Recorded /
Tier1Chirp (escalate) therefore stay INSIDE their branches below, not at
module level.
"""

import argparse
import json
import os
import sys

from dhvani.calibrate import (
    DEFAULT_PCM_CACHE,
    MIN_BUCKET_SAMPLES,
    district_spread,
    LazySegments,
    NoMeasuredBuckets,
    PcmCacheError,
    collect,
    escalate_selected,
    estimate_cost,
    histogram,
    repair_selected,
    stratify,
    write_table,
)
from dhvani.config import GEMINI_DAILY_QUOTA
from dhvani.store import Store

DEFAULT_LANGS = ["kn-IN", "ml-IN", "hi-IN"]


def _print_histogram(scored, stream=None):
    # stream must resolve sys.stderr at CALL time, not at function
    # definition time: a default of `stream=sys.stderr` binds once, at
    # import, and goes stale under pytest's capsys, which swaps in a new
    # sys.stderr object per test rather than patching the existing one.
    if stream is None:
        stream = sys.stderr
    hist = histogram(scored)
    print(f"collected {len(scored)} segments", file=stream)
    for bucket, count in hist.items():
        bar = "#" * min(count // 10, 50)
        # stratify() silently drops any bucket under MIN_BUCKET_SAMPLES —
        # without this marker, a line like "0.3-0.4: 15" gives no hint
        # that it contributed nothing to the escalated sample.
        marker = "" if count >= MIN_BUCKET_SAMPLES else "  (below floor, not escalated)"
        print(f"  {bucket:10} {count:6d}  {bar}{marker}", file=stream)

    # Spec §9.4 concentration. This is a large part of what a human is
    # looking FOR when they read this histogram before authorizing spend:
    # a table built mostly from a handful of speakers or districts measures
    # those people, not the language.
    #
    # The two are reported differently because only one can be enforced.
    # stratify() guarantees speaker-disjointness silently, so the speaker
    # line is a statement of what WILL be escalated. District-disjointness
    # is impossible by pigeonhole -- IndicVoices spans 145 districts and a
    # run selects ~150 samples -- so concentration there is shown and left
    # to judgement rather than gated.
    speakers = {row.get("speaker_id") for row in scored if row.get("speaker_id")}
    spread = district_spread(scored)
    if speakers:
        print(f"  {len(speakers)} distinct speakers across {len(scored)} segments "
              f"(escalation takes at most one per speaker)", file=stream)
    if spread:
        top = sorted(spread.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        summary = ", ".join(f"{name} x{n}" for name, n in top)
        print(f"  {len(spread)} districts; most common: {summary}", file=stream)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dhvani-calibrate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="phase 1: transcribe and score locally (free)")
    c.add_argument("--db", default="calibration.db")
    c.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    c.add_argument("--per-lang", type=int, default=1000)
    c.add_argument("--scored-out", default="scored.json")
    c.add_argument("--pcm-cache", default=DEFAULT_PCM_CACHE,
                   help="directory phase 1 writes segment PCM to and phase 2 "
                        "reads it back from; both phases must be given the same one")
    # default=None, resolved against corpus.DEFAULT_DATASET inside the
    # branch below: dhvani.corpus must not be imported at module level (see
    # this module's docstring), so the constant is not available here to use
    # as an argparse default. The literal in the help text is pinned to the
    # constant by a test so the two cannot drift.
    c.add_argument("--dataset", default=None,
                   help="HuggingFace dataset id to stream "
                        "(default: ai4bharat/IndicVoices). Recorded into "
                        "each segment's source_id, so a second dataset "
                        "collected into the same --db stays distinguishable")

    e = sub.add_parser("escalate", help="phase 2: stratify and run Tier 1 (paid)")
    e.add_argument("--db", default="calibration.db")
    e.add_argument("--scored-in", default="scored.json")
    e.add_argument("--out", default="delta_table.json")
    e.add_argument("--pcm-cache", default=DEFAULT_PCM_CACHE,
                   help="directory phase 1 wrote segment PCM to; Tier 1 is sent "
                        "this audio inline, so it must be the same directory")
    # I5: escalate used to hardcode mode="live" and fixtures="fixtures",
    # so the spec's replay-mode test matrix was unreachable from the CLI and
    # no calibration run was reproducible without credentials and a bill.
    # Same shape as dhvani/cli.py, including the replay default: a command
    # that spends money must be asked for explicitly.
    e.add_argument("--mode", default=os.environ.get("DHVANI_MODE", "replay"),
                   choices=["replay", "record", "live"],
                   help="replay reads fixtures and never calls out; record "
                        "calls live and saves the response; live just calls")
    e.add_argument("--fixtures", default="fixtures")
    e.add_argument("--dry-run", action="store_true",
                   help="stratify, print the estimate, and exit without spending")
    e.add_argument("--confirm", action="store_true",
                   help="required before any paid call")

    r = sub.add_parser("repair",
                       help="phase 3: run Tier 2 over the sample (free tier)")
    r.add_argument("--db", default="calibration.db")
    r.add_argument("--scored-in", default="scored.json")
    r.add_argument("--out", default="delta_table_tier2.json",
                   help="written separately from the Tier 1 table by default: "
                        "merging them is a decision, not a side effect")
    r.add_argument("--mode", default=os.environ.get("DHVANI_MODE", "replay"),
                   choices=["replay", "record", "live"])
    r.add_argument("--fixtures", default="fixtures")
    r.add_argument("--quota", type=int, default=GEMINI_DAILY_QUOTA,
                   help="daily request ceiling for this run (local, not the "
                        "vendor's -- see the design, §8)")
    r.add_argument("--lang", default="hi-IN")
    r.add_argument("--tier1-variant", default="",
                   help="if set, a segment with a Tier 1 hypothesis under this "
                        "key is repaired FROM it, and scored against it")
    r.add_argument("--dry-run", action="store_true",
                   help="stratify, print the plan, and exit without calling")
    # Deliberately NO --confirm. Tier 1 has one because it bills; Tier 2 is
    # free, so the same ceremony would be cargo-cult rather than a guard.
    # The quota budget is printed instead, which is the resource at stake.

    args = ap.parse_args(argv)

    if args.cmd == "collect":
        from dhvani.backends.tier0_conformer import Tier0Conformer
        from dhvani.corpus import DEFAULT_DATASET, IndicVoicesCorpus

        corpus = IndicVoicesCorpus(args.dataset or DEFAULT_DATASET)
        with Store(args.db) as store:
            # I6: ONE BACKEND PER LANGUAGE. A single Tier0Conformer() takes
            # the default lang="hi", so reusing it across --langs decoded
            # Kannada and Malayalam audio as Hindi and gave all three runs
            # the same variant_key, collapsing them onto one cache entry.
            # Tier0Conformer speaks the bare code ("hi"), the corpus and the
            # scored records speak the full tag ("hi-IN") -- same split the
            # corpus loader does to pick its config.
            scored = collect(corpus,
                             lambda lang: Tier0Conformer(lang=lang.split("-")[0]),
                             store, args.langs, args.per_lang, args.pcm_cache)
        _print_histogram(scored)
        with open(args.scored_out, "w", encoding="utf-8") as fh:
            json.dump(scored, fh, indent=2)
        print(f"wrote {args.scored_out}", file=sys.stderr)
        return 0

    if args.cmd == "repair":
        return _run_repair(args)

    # escalate
    try:
        with open(args.scored_in, encoding="utf-8") as fh:
            scored = json.load(fh)
    except FileNotFoundError:
        print(f"missing {args.scored_in}; run `dhvani-calibrate collect` first",
              file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"{args.scored_in} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    selected = stratify(scored)
    estimate = estimate_cost(selected)
    _print_histogram(scored)
    print(f"stratified {len(selected)} segments; estimated cost ${estimate:.4f}",
          file=sys.stderr)

    if args.dry_run:
        print("dry run: nothing spent, no table written", file=sys.stderr)
        return 0

    if not args.confirm:
        print(f"refusing to spend ${estimate:.4f} without --confirm", file=sys.stderr)
        return 2

    from dhvani.backends.base import Recorded
    from dhvani.backends.tier1_chirp import Tier1Chirp

    # The audio comes from the PCM cache phase 1 wrote (--pcm-cache), because
    # Tier1Chirp.transcribe() sends segment.pcm INLINE via recognize_pcm --
    # there is no GCS object, and no other copy of the corpus survives the
    # gap between the two phases. A missing file is fatal, deliberately:
    # this used to pass np.zeros(1) under a comment claiming Tier 1 fetched
    # the audio itself, which would have billed a real transcription of two
    # bytes of silence for every selected segment and produced a uniformly
    # wrong table that still looked plausible.
    segments = LazySegments(selected, args.pcm_cache)
    langs = sorted({s["lang"] for s in selected})

    try:
        with Store(args.db) as store:
            before = store.total_spend()
            rows = []
            # I6, Tier 1 half: Tier1Chirp() also defaults to lang="hi-IN",
            # so one instance shared across the corpus asked Chirp to
            # transcribe Kannada and Malayalam as Hindi. dhvani/cli.py
            # already builds Tier1Chirp(lang=...) per run; this is the same
            # shape, once per language present in the selection.
            for lang in langs:
                subset = [s for s in selected if s["lang"] == lang]
                tier1 = Recorded(Tier1Chirp(lang=lang), args.mode,
                                 args.fixtures, store)
                rows.extend(escalate_selected(subset, tier1, store, segments))
            spent = store.total_spend() - before
    # PcmCacheError, not PcmCacheMiss: a corrupt cache entry is the
    # other way this can fail and must reach the operator as the same
    # clean message and exit 3, not as a traceback.
    except PcmCacheError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    # escalate_selected() silently skips any segment lacking a reference or
    # a Tier 0 hypothesis. Without this line, "3 legitimately lacked Tier
    # 0" and "the variant key was wrong so everything was skipped" are
    # indistinguishable from the outside.
    skipped = len(selected) - len(rows)
    print(f"skipped {skipped} of {len(selected)} selected segments "
          f"(no reference or Tier 0 hypothesis)", file=sys.stderr)

    # I7: a run that measured nothing must NOT write a table and must NOT
    # exit 0. delta_table.json is what the router trusts as measurement, so
    # {"tier1": {}} written on a fully-skipped run reads to it as "Tier 1
    # never helps" -- and the money for this run is already spent by now.
    # Refusing also leaves any previous, real table on disk untouched.
    try:
        write_table(rows, selected, args.out, spent, langs)
    except NoMeasuredBuckets as exc:
        print(str(exc), file=sys.stderr)
        print(f"spent ${spent:.4f} and wrote no table", file=sys.stderr)
        return 4

    print(f"wrote {args.out} from {len(rows)} rows; spent ${spent:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


class _TextSegment:
    """The minimum a text-only tier needs: an id and its timings.

    Deliberately not a Segment -- it carries no pcm, so an accidental
    attempt to send audio from the repair path fails loudly here rather
    than silently transmitting zeros.
    """

    __slots__ = ("segment_id", "t_start_ms", "t_end_ms")

    def __init__(self, scored_row):
        self.segment_id = scored_row["segment_id"]
        self.t_start_ms = 0
        self.t_end_ms = int(scored_row.get("duration_ms", 0))


def _run_repair(args) -> int:
    """Phase 3: measure a Tier 2 delta over the stratified sample.

    Deliberately unlike phase 2 in three ways, each following from Tier 2
    being free rather than paid:

      no --confirm      nothing is billed, so a spend gate would be ceremony.
                        The QUOTA is printed instead, because that is the
                        resource a run can actually exhaust.
      no cost estimate  there is no cost. The plan states request count
                        against the daily ceiling.
      separate --out    the Tier 2 table is written beside the Tier 1 one,
                        not merged into it. Merging two independently
                        measured tables is a decision a human should make,
                        not something a calibration command does quietly.
    """
    try:
        with open(args.scored_in, encoding="utf-8") as fh:
            scored = json.load(fh)
    except FileNotFoundError:
        print(f"missing {args.scored_in}; run `dhvani-calibrate collect` first",
              file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"{args.scored_in} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    selected = stratify(scored)
    _print_histogram(scored)
    print(f"stratified {len(selected)} segments; "
          f"needs at most {len(selected)} of {args.quota} daily quota "
          f"(Tier 2 is free -- quota is the only budget)", file=sys.stderr)

    if args.dry_run:
        print("dry run: nothing called, no table written", file=sys.stderr)
        return 0

    from dhvani.backends.base import Recorded
    from dhvani.backends.tier2_gemini import Tier2Gemini
    from dhvani.quota import GEMINI_RPM_DEFAULT, QuotaGate, TokenBucket

    hypotheses: dict = {}
    with Store(args.db) as store:
        tier2 = Recorded(
            Tier2Gemini(hypothesis_source=hypotheses.get, lang=args.lang),
            args.mode, args.fixtures, store,
        )
        gate = QuotaGate(store=store, tier=tier2.name, cap=args.quota,
                         bucket=TokenBucket(capacity=max(1, GEMINI_RPM_DEFAULT),
                                            refill_per_sec=GEMINI_RPM_DEFAULT / 60.0))
        # NOT LazySegments. Tier 2 is text-in, text-out: Tier2Gemini reads
        # segment.segment_id and nothing else, so loading phase 1's cached
        # PCM would demand audio this tier never looks at -- and would make
        # phase 3 impossible whenever that cache is gone, which is precisely
        # the situation this project is in. Tier 1 needs LazySegments because
        # Tier1Chirp sends the PCM inline; the requirement does not transfer.
        segments = {s["segment_id"]: _TextSegment(s) for s in selected}
        rows = repair_selected(selected, tier2, store, gate, hypotheses,
                               segments, tier1_variant=args.tier1_variant)
        used = store.quota_used(tier2.name, gate._today())

    if len(rows) < len(selected):
        print(f"measured {len(rows)} of {len(selected)} selected "
              f"({used} requests used) -- re-run to continue where this "
              f"stopped; cached repairs cost no quota", file=sys.stderr)

    write_table(rows, selected, args.out, 0.0,
                sorted({s.get("lang") for s in selected if s.get("lang")}),
                tier="tier2")
    print(f"wrote {args.out} from {len(rows)} rows; spent $0.00 "
          f"({used} free-tier requests)", file=sys.stderr)
    return 0
