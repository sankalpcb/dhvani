"""Entrypoint: dhvani <audio.wav> [--out track.json]

Usage note (fix round 2, I4): the audio path is a bare positional, NOT a
`transcribe` subcommand. This module's docstring and report_cli's error
message both used to name such a subcommand, which argparse rejects — it
parses the verb as the audio path and then errors on the real one. The
mismatch is resolved toward the bare positional because
Phase 1 has exactly one verb; a subcommand layer with nothing to
distinguish would be surface area for its own sake. Phase 2 can introduce
subcommands when it adds a second verb.

--out persists the track as JSON so `make bench` (dhvani.report_cli) has
an input; the same payload still goes to stdout.
"""

import argparse
import json
import os
import sys

import soundfile as sf

from dhvani.audio import normalize
from dhvani.backends.base import Recorded
from dhvani.backends.tier0_conformer import Tier0Conformer
from dhvani.config import POLICY_ID
from dhvani.metrics import summarize
from dhvani.pipeline import run
from dhvani.segmenter import segment as split
from dhvani.store import Store


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dhvani")
    ap.add_argument("audio")
    ap.add_argument("--db", default="dhvani.db")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--mode", default=os.environ.get("DHVANI_MODE", "replay"))
    ap.add_argument("--fixtures", default="fixtures")
    ap.add_argument("--delta-table", default="delta_table.json")
    ap.add_argument(
        "--out",
        help="also write the caption track here as JSON "
             "(dhvani.report_cli reads this file; see `make bench`)",
    )
    ap.add_argument(
        "--escalate", action="store_true",
        help="after transcribing, plan and submit a Tier 1 escalation batch",
    )
    ap.add_argument(
        "--reconcile", action="store_true",
        help="poll outstanding escalation jobs and merge any completed results",
    )
    args = ap.parse_args(argv)

    samples, rate = sf.read(args.audio)
    pcm = normalize(samples, rate)

    delta_table = {}
    if os.path.exists(args.delta_table):
        with open(args.delta_table) as fh:
            delta_table = json.load(fh)

    samples: dict = {}

    with Store(args.db) as store:
        tier0 = Recorded(
            Tier0Conformer(lang=args.lang), args.mode, args.fixtures, store
        )
        entries = run(pcm, os.path.basename(args.audio), tier0, store,
                      delta_table, args.budget, samples=samples)

        if args.escalate or args.reconcile:
            from dhvani.backends.async_base import SyncAsyncAdapter
            from dhvani.backends.tier1_chirp import Tier1Chirp
            from dhvani.escalate import escalate as do_escalate
            from dhvani.reconcile import reconcile as do_reconcile
            from dhvani.track import entries_to_json

            source = os.path.basename(args.audio)
            # Real Segments, not stubs: Tier1Chirp.transcribe() reads .pcm.
            segments = {s.segment_id: s for s in split(pcm)}
            # Recorded is the ONLY thing that enforces "replay never falls
            # back to live" (fix round 3, C2). Building the adapter over a
            # bare Tier1Chirp meant --mode replay was ignored entirely for
            # Tier 1: anyone who installed the `cloud` extra to record
            # fixtures got a real billed API call out of a replay-mode
            # command. Ordering matters -- Recorded wraps the sync backend,
            # SyncAsyncAdapter wraps Recorded -- so the mode check happens
            # on the way to the paid call, not around the async shell.
            tier1 = SyncAsyncAdapter(
                Recorded(Tier1Chirp(lang=f"{args.lang}-IN"),
                         args.mode, args.fixtures, store)
            )

            if store.latest_track_version(source) == 0:
                store.put_track(source, 1, POLICY_ID,
                                entries_to_json(entries), 0.0)

            if args.escalate:
                do_escalate(source, entries, segments, tier1, store,
                            delta_table, args.budget)
            if args.reconcile:
                # tier1's _jobs is in-memory and starts empty in a process
                # that did not also --escalate; jobs rows in the Store are
                # durable, so poll() would raise JobNotFound for a job that
                # is genuinely still outstanding. adopt() rebuilds it from
                # the durable record before reconcile() polls it.
                for job in store.open_jobs(source):
                    if job["tier"] != tier1.name or job["variant_key"] != tier1.variant_key:
                        continue
                    job_segments = [segments[sid] for sid in job["segment_ids"]
                                     if sid in segments]
                    if len(job_segments) != len(job["segment_ids"]):
                        continue
                    tier1.adopt(job["job_id"], job_segments)
                do_reconcile(source, tier1, store, samples=samples)

    payload = json.dumps([e.__dict__ for e in entries], ensure_ascii=False, indent=2)

    if args.out:
        # Same bytes that go to stdout, so a track consumed from --out and
        # one captured by redirecting stdout are interchangeable.
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload)

    print(payload)

    # Metrics go to stderr, never stdout: stdout is the caption track JSON
    # and --out asserts it matches stdout byte-for-byte (see the docstring
    # above), so anything printed here must not touch it.
    print(json.dumps(summarize(samples), ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
