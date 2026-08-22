"""Entrypoint: dhvani transcribe <audio.wav>"""

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

    print(json.dumps([e.__dict__ for e in entries], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
