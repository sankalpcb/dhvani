"""make bench entrypoint: renders the cost/quality frontier to stdout."""

import json
import os
import sys

from dhvani.report import frontier, render_markdown

BUDGETS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 1.0]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    track_path = argv[0] if argv else "results/track.json"
    table_path = argv[1] if len(argv) > 1 else "delta_table.json"

    if not os.path.exists(track_path):
        print(f"missing {track_path}; run `dhvani transcribe` first", file=sys.stderr)
        return 1

    from dhvani.pipeline import TrackEntry

    with open(track_path, encoding="utf-8") as fh:
        try:
            track_rows = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"malformed JSON in {track_path}: {exc}", file=sys.stderr)
            return 1

    try:
        entries = [TrackEntry(**row) for row in track_rows]
    except TypeError as exc:
        print(f"malformed track row in {track_path}: {exc}", file=sys.stderr)
        return 1

    delta_table = {}
    if os.path.exists(table_path):
        with open(table_path, encoding="utf-8") as fh:
            try:
                delta_table = json.load(fh)
            except json.JSONDecodeError as exc:
                print(f"malformed JSON in {table_path}: {exc}", file=sys.stderr)
                return 1

    print("# Dhvani cost/quality frontier\n")
    print(render_markdown(frontier(entries, delta_table, BUDGETS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
