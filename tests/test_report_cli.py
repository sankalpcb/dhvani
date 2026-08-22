"""Covers dhvani.report_cli's clean-exit behavior on bad input (Task 11
review Finding 2): malformed JSON in either file, and a track row that
doesn't match TrackEntry's fields, must all exit 1 with a clear stderr
message -- never a raw traceback, since this is the user-facing entrypoint.
"""

import json

from dhvani.report_cli import main


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def _valid_row(**overrides):
    row = {
        "segment_id": "s0", "t_start_ms": 0, "t_end_ms": 3000,
        "text": "hi", "risk": 0.1, "band": "ship",
    }
    row.update(overrides)
    return row


def test_missing_track_file_exits_cleanly(capsys, tmp_path):
    missing = tmp_path / "no_such_track.json"
    rc = main([str(missing)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "missing" in err
    assert str(missing) in err


def test_malformed_track_json_exits_cleanly(capsys, tmp_path):
    track = tmp_path / "track.json"
    track.write_text("{not valid json", encoding="utf-8")
    rc = main([str(track)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "malformed" in err.lower()
    assert str(track) in err


def test_malformed_delta_table_json_exits_cleanly(capsys, tmp_path):
    track = tmp_path / "track.json"
    _write(track, [_valid_row()])
    table = tmp_path / "delta_table.json"
    table.write_text("{not valid json", encoding="utf-8")

    rc = main([str(track), str(table)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "malformed" in err.lower()
    assert str(table) in err


def test_track_row_missing_required_field_exits_cleanly(capsys, tmp_path):
    track = tmp_path / "track.json"
    _write(track, [{"segment_id": "s0", "risk": 0.1}])  # missing most fields
    rc = main([str(track)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "malformed track row" in err.lower()


def test_track_row_unexpected_field_exits_cleanly(capsys, tmp_path):
    track = tmp_path / "track.json"
    _write(track, [_valid_row(extra_field="nope")])
    rc = main([str(track)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "malformed track row" in err.lower()


def test_happy_path_still_exits_zero(capsys, tmp_path):
    track = tmp_path / "track.json"
    _write(track, [_valid_row()])
    rc = main([str(track)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cost/quality frontier" in out.lower()
