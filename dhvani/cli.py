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

import soundfile as sf

from dhvani.audio import normalize
from dhvani.backends.base import Recorded
from dhvani.backends.tier0_conformer import Tier0Conformer
from dhvani.pipeline import run
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
    args = ap.parse_args(argv)

    samples, rate = sf.read(args.audio)
    pcm = normalize(samples, rate)

    delta_table = {}
    if os.path.exists(args.delta_table):
        with open(args.delta_table) as fh:
            delta_table = json.load(fh)

    with Store(args.db) as store:
        tier0 = Recorded(
            Tier0Conformer(lang=args.lang), args.mode, args.fixtures, store
        )
        entries = run(pcm, os.path.basename(args.audio), tier0, store,
                      delta_table, args.budget)

    payload = json.dumps([e.__dict__ for e in entries], ensure_ascii=False, indent=2)

    if args.out:
        # Same bytes that go to stdout, so a track consumed from --out and
        # one captured by redirecting stdout are interchangeable.
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload)

    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
