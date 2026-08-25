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
    LazySegments,
    NoMeasuredBuckets,
    PcmCacheError,
    collect,
    escalate_selected,
    estimate_cost,
    histogram,
    stratify,
    write_table,
)
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
