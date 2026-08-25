"""End-to-end CLI tests.

These specifically guard project goal G5: a stranger clones the repo and
runs the offline replay workflow with no cloud credentials and no ML
dependencies installed. Replay mode never calls the Tier 0 backend at all,
so dhvani.cli must not require torch/transformers to get there.
"""

import json
import pathlib

import numpy as np
import pytest
import soundfile as sf

import dhvani.cli
from dhvani.audio import normalize
from dhvani.backends.tier0_conformer import Tier0Conformer
from dhvani.cli import main
from dhvani.config import SAMPLE_RATE
from dhvani.ids import variant_slug
from dhvani.report_cli import main as report_main
from dhvani.segmenter import segment as split


def _write_wav(path, seconds=6.0, freq=200.0):
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), x, SAMPLE_RATE, subtype="PCM_16")


def test_cli_replay_mode_runs_end_to_end_without_torch(tmp_path, capsys):
    """G5 regression guard, verified live: `dhvani foo.wav` with
    the default --mode replay must exit 0 and emit a track. This project's
    venv genuinely has neither torch nor transformers installed, so this
    passing is real evidence the replay path never touches the model
    loader — not a simulated stand-in for that fact."""
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path)

    # Replicate exactly what dhvani.cli.main() does before it reaches the
    # backend, purely to learn which segment_id(s) it will look up — so we
    # can pre-write the fixture(s) replay mode needs. normalize/segment are
    # pure numpy/scipy; neither imports torch.
    samples, rate = sf.read(str(wav_path))
    pcm = normalize(samples, rate)
    segments = split(pcm)
    assert segments, "expected at least one voiced segment from the test tone"

    # Fixtures are keyed by tier AND backend variant AND POLICY_ID (fix
    # round 2, I2/I3), so build the same backend the CLI will build and
    # ask it where its fixtures live rather than hardcoding a layout.
    fixtures_dir = tmp_path / "fixtures"
    tier0_dir = (fixtures_dir / "tier0"
                 / variant_slug(Tier0Conformer(lang="hi").variant_key))
    tier0_dir.mkdir(parents=True)
    for seg in segments:
        (tier0_dir / f"{seg.segment_id}.json").write_text(json.dumps({
            "text": "नमस्ते world",
            "signals": {"ctc_rnnt_disagreement": 0.1},
        }))

    exit_code = main([
        str(wav_path),
        "--db", str(tmp_path / "t.db"),
        "--mode", "replay",
        "--fixtures", str(fixtures_dir),
        "--delta-table", str(tmp_path / "no-such-delta-table.json"),
    ])

    assert exit_code == 0
    track = json.loads(capsys.readouterr().out)
    assert len(track) == len(segments)
    assert all(entry["text"] for entry in track)
    assert all(entry["band"] in {"ship", "marked", "review"} for entry in track)


def test_cli_emits_metrics_summary_on_stderr_and_leaves_stdout_alone(tmp_path, capsys):
    """I7: the CLI must surface timing metrics, but ONLY on stderr -- stdout
    carries the caption track JSON and --out asserts it matches stdout
    byte-for-byte, so metrics leaking onto stdout would break that."""
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path)

    samples, rate = sf.read(str(wav_path))
    pcm = normalize(samples, rate)
    segments = split(pcm)

    fixtures_dir = tmp_path / "fixtures"
    tier0_dir = (fixtures_dir / "tier0"
                 / variant_slug(Tier0Conformer(lang="hi").variant_key))
    tier0_dir.mkdir(parents=True)
    for seg in segments:
        (tier0_dir / f"{seg.segment_id}.json").write_text(json.dumps({
            "text": "नमस्ते world",
            "signals": {"ctc_rnnt_disagreement": 0.1},
        }))

    exit_code = main([
        str(wav_path),
        "--db", str(tmp_path / "t.db"),
        "--mode", "replay",
        "--fixtures", str(fixtures_dir),
        "--delta-table", str(tmp_path / "no-such-delta-table.json"),
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    track = json.loads(captured.out)
    assert len(track) == len(segments)
    assert "tier0" in captured.err


def test_cli_help_runs_without_error():
    """Sanity check: --help must exit 0 and never construct a backend."""
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("argparse --help should raise SystemExit(0)")


# --- Fix round 2, I4: the CLI must be installable, and `make bench` runnable ---

def test_pyproject_declares_the_dhvani_console_script():
    """`dhvani` has to actually exist as a command. Every docstring and
    error message in the project names it, but nothing declared it."""
    import tomllib

    root = pathlib.Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)

    scripts = data["project"]["scripts"]
    assert scripts["dhvani"] == "dhvani.cli:main"
    assert scripts["dhvani-bench"] == "dhvani.report_cli:main"


def test_cli_takes_a_bare_audio_positional_not_a_transcribe_subcommand():
    """Resolves the documented CLI/doc mismatch in ONE direction: there is
    no `transcribe` subcommand, so nothing may document one. Phase 1 has a
    single verb; the word would be parsed as the audio path and fail
    confusingly."""
    with pytest.raises(SystemExit):
        main(["transcribe", "clip.wav"])  # two positionals: argparse rejects

    assert "transcribe <audio.wav>" not in (dhvani.cli.__doc__ or "")


def test_cli_persists_the_track_to_out_path(tmp_path, capsys):
    """`make bench` needs a track file to report on, and before this the
    CLI could only print to stdout. --out writes the same payload it
    prints, creating parent directories as needed."""
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path)

    samples, rate = sf.read(str(wav_path))
    segments = split(normalize(samples, rate))
    tier0_dir = (tmp_path / "fixtures" / "tier0"
                 / variant_slug(Tier0Conformer(lang="hi").variant_key))
    tier0_dir.mkdir(parents=True)
    for seg in segments:
        (tier0_dir / f"{seg.segment_id}.json").write_text(json.dumps({
            "text": "नमस्ते world", "signals": {"ctc_rnnt_disagreement": 0.1},
        }))

    out_path = tmp_path / "nested" / "results" / "track.json"
    exit_code = main([
        str(wav_path),
        "--db", str(tmp_path / "t.db"),
        "--mode", "replay",
        "--fixtures", str(tmp_path / "fixtures"),
        "--delta-table", str(tmp_path / "no-such-delta-table.json"),
        "--out", str(out_path),
    ])

    assert exit_code == 0
    assert out_path.exists(), "--out must create parent directories"
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == json.loads(capsys.readouterr().out)

    # The whole point: report_cli must accept exactly this file. This is
    # `make bench`'s input contract, end to end.
    assert report_main([str(out_path)]) == 0


# --- Fix round 3, C2/C3: the async path through the shipped CLI ---

ALL_BUCKETS = {f"{i/10:.1f}-{(i+1)/10:.1f}": 20.0 for i in range(10)}


def _prepare(tmp_path, wav_path, freq=200.0,
             tier1_text="escalated-by-chirp"):
    """Write the wav plus tier0 AND tier1 fixtures for every segment.

    freq and tier1_text exist so a test can build two DISTINGUISHABLE
    sources under one fixtures tree; fixtures are keyed by segment_id, so
    different audio cannot clash and the directories are shared.

    Returns (fixtures_dir, delta_table_path, segments).
    """
    _write_wav(wav_path, freq=freq)
    samples, rate = sf.read(str(wav_path))
    segments = split(normalize(samples, rate))

    fixtures = tmp_path / "fixtures"
    tier0_dir = fixtures / "tier0" / variant_slug(Tier0Conformer(lang="hi").variant_key)
    tier0_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        (tier0_dir / f"{seg.segment_id}.json").write_text(json.dumps({
            "text": "नमस्ते world", "signals": {"ctc_rnnt_disagreement": 0.1},
        }))

    from dhvani.backends.tier1_chirp import Tier1Chirp
    tier1_dir = fixtures / "tier1" / variant_slug(Tier1Chirp(lang="hi-IN").variant_key)
    tier1_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        (tier1_dir / f"{seg.segment_id}.json").write_text(json.dumps({
            "text": tier1_text, "signals": {},
        }))

    delta_path = tmp_path / "delta.json"
    delta_path.write_text(json.dumps({"tier1": ALL_BUCKETS}))
    return fixtures, delta_path, segments


def test_cli_escalate_in_replay_mode_never_makes_a_live_call(tmp_path, capsys,
                                                             monkeypatch):
    """C2: cli.py built SyncAsyncAdapter(Tier1Chirp(...)) with no Recorded
    wrapper, and Recorded is the only enforcer of "replay never falls back
    to live" -- so --mode replay was ignored entirely for Tier 1.

    This docstring used to claim the environment had no google-cloud-speech
    installed, and to lean on that as the reason a regression here would be
    caught rather than billed. It is installed -- it lives behind the
    `cloud` extra, and the extra is present -- and application default
    credentials exist on this machine too. So the fallback the claim
    described is gone in both halves: reaching the live path from here
    would be a real billed request, not a ModuleNotFoundError.

    The environment was never the right thing to depend on. Assert the
    property instead: _default_client() is the single door to the billed
    API, and a replay-mode command must never open it. Same tripwire the
    calibrate side already uses, so this stays a valid regression guard
    wherever it runs and whatever is installed.
    """
    import dhvani.backends.tier1_chirp as tier1_mod

    # The tripwire RECORDS rather than only raising, and is asserted after
    # the run. Raising alone is not enough here and quietly proves nothing:
    # reconcile() wraps each poll() in `except Exception` on purpose, so one
    # failing job cannot abandon the whole pass -- which means it swallows a
    # tripwire's AssertionError too, dead-letters the job, and lets main()
    # return 0 with the track still the right length. A raise-only version
    # of this test passes even with the C2 bug reintroduced. The flag
    # survives that handler; the exception does not.
    live_calls = []

    def live_call_tripwire(*args, **kwargs):
        live_calls.append(args)
        raise AssertionError(
            "replay mode reached _default_client() -- the next step is a "
            "real billed Chirp request"
        )

    monkeypatch.setattr(tier1_mod, "_default_client", live_call_tripwire)

    wav_path = tmp_path / "clip.wav"
    fixtures, delta_path, segments = _prepare(tmp_path, wav_path)

    exit_code = main([
        str(wav_path),
        "--db", str(tmp_path / "t.db"),
        "--mode", "replay",
        "--fixtures", str(fixtures),
        "--delta-table", str(delta_path),
        "--budget", "10.0",
        "--escalate", "--reconcile",
    ])

    assert not live_calls, (
        "replay mode opened the door to the billed Chirp API "
        f"({len(live_calls)} call(s) to _default_client)"
    )
    assert exit_code == 0
    track = json.loads(capsys.readouterr().out)
    assert len(track) == len(segments)
    # Anti-vacuity: "never reached the live client" is trivially true of a
    # run that escalated nothing, so prove the replay path really ran.
    # This used to have to consult the store, because stdout carried run()'s
    # Tier 0 entries and not the escalation; the CLI now prints the
    # reconciled track, so the evidence is right here in stdout.
    assert any(e["text"] == "escalated-by-chirp" for e in track), (
        "replay produced no escalated segments -- nothing was exercised"
    )


def test_cli_prints_the_reconciled_track_not_the_tier0_input(tmp_path, capsys):
    """The CLI printed run()'s Tier 0 entries even when --reconcile had just
    merged Tier 1 results into a new track version.

    So an operator who paid for Tier 1 got output that did not contain what
    they bought -- the escalated captions existed only inside the SQLite
    store, reachable by nothing the CLI offered. stdout is the product of
    this command; it has to show the track the run actually ended with.
    """
    from dhvani.store import Store

    wav_path = tmp_path / "clip.wav"
    fixtures, delta_path, segments = _prepare(tmp_path, wav_path)
    db = tmp_path / "t.db"

    exit_code = main([
        str(wav_path), "--db", str(db), "--mode", "replay",
        "--fixtures", str(fixtures), "--delta-table", str(delta_path),
        "--budget", "10.0", "--escalate", "--reconcile",
    ])

    assert exit_code == 0
    track = json.loads(capsys.readouterr().out)
    assert [e["text"] for e in track] == ["escalated-by-chirp"], (
        "stdout must carry the escalated text, not the Tier 0 text"
    )

    # And it must be the SAME track the store ended on, not a lookalike
    # rebuilt in the CLI. The store is keyed by the content-addressed
    # source id, not the bare basename -- see ids.source_id -- so derive it
    # the way the CLI does rather than hardcoding "clip.wav".
    from dhvani.ids import source_id as make_source_id
    samples, rate = sf.read(str(wav_path))
    source = make_source_id(wav_path.name, normalize(samples, rate))

    with Store(str(db)) as store:
        version = store.latest_track_version(source)
        assert version > 1, "reconcile() never advanced the track"
        stored = json.loads(store.get_track(source, version)["content_json"])
    assert track == stored


def test_reconciled_output_carries_the_recomputed_band(tmp_path, capsys):
    """merge_entries() recomputes band from the escalated risk so a segment
    can leave the review band. Printing the Tier 0 entries threw that away
    silently; printing the merged track must preserve it, band and risk
    agreeing on every entry."""
    from dhvani.pipeline import band_of

    wav_path = tmp_path / "clip.wav"
    fixtures, delta_path, _ = _prepare(tmp_path, wav_path)

    assert main([
        str(wav_path), "--db", str(tmp_path / "t.db"), "--mode", "replay",
        "--fixtures", str(fixtures), "--delta-table", str(delta_path),
        "--budget", "10.0", "--escalate", "--reconcile",
    ]) == 0

    track = json.loads(capsys.readouterr().out)
    assert track, "nothing was printed"
    for entry in track:
        assert entry["band"] == band_of(entry["risk"]), (
            f"band {entry['band']} disagrees with risk {entry['risk']}"
        )


def test_out_file_still_matches_stdout_for_a_reconciled_track(tmp_path, capsys):
    """--out's byte-for-byte contract with stdout predates escalation and
    must survive it: both sides have to move to the reconciled track
    together, or `make bench` reports on a different track than the one the
    operator was shown."""
    wav_path = tmp_path / "clip.wav"
    fixtures, delta_path, _ = _prepare(tmp_path, wav_path)
    out_path = tmp_path / "nested" / "track.json"

    assert main([
        str(wav_path), "--db", str(tmp_path / "t.db"), "--mode", "replay",
        "--fixtures", str(fixtures), "--delta-table", str(delta_path),
        "--budget", "10.0", "--escalate", "--reconcile",
        "--out", str(out_path),
    ]) == 0

    printed = capsys.readouterr().out
    # The file holds the payload; stdout holds the payload plus the newline
    # print() appends. That one-character difference is pre-existing and not
    # what this test is about -- the point is that both sides carry the SAME
    # track, so they must not be allowed to drift apart.
    assert printed == out_path.read_text(encoding="utf-8") + "\n", (
        "--out and stdout diverged"
    )
    assert "escalated-by-chirp" in printed
    # Still `make bench`'s input contract, now on a reconciled track.
    assert report_main([str(out_path)]) == 0


def test_a_run_with_no_escalation_flags_prints_the_tier0_track(tmp_path, capsys):
    """The plain path must not change. Without --escalate/--reconcile the
    CLI never touches tracks, and reading the store here would make a bare
    `dhvani clip.wav` start replaying a previous run's escalation."""
    wav_path = tmp_path / "clip.wav"
    fixtures, delta_path, _ = _prepare(tmp_path, wav_path)
    db = tmp_path / "t.db"

    # First, a run that escalates and leaves a reconciled v2 in the store.
    assert main([
        str(wav_path), "--db", str(db), "--mode", "replay",
        "--fixtures", str(fixtures), "--delta-table", str(delta_path),
        "--budget", "10.0", "--escalate", "--reconcile",
    ]) == 0
    assert "escalated-by-chirp" in capsys.readouterr().out

    # Then the same audio with no escalation flags at all.
    assert main([
        str(wav_path), "--db", str(db), "--mode", "replay",
        "--fixtures", str(fixtures), "--delta-table", str(delta_path),
    ]) == 0

    track = json.loads(capsys.readouterr().out)
    assert [e["text"] for e in track] == ["नमस्ते world"], (
        "a non-escalating run must print its own Tier 0 output"
    )


def test_two_files_with_the_same_basename_do_not_share_a_track(tmp_path, capsys):
    """The basename collision, end to end.

    source_id was os.path.basename(audio), so a/clip.wav and b/clip.wav
    were ONE source inside a single --db. The second run found the first's
    track already at v2 and so never wrote its own v1; reconcile() then
    merged the second file's results into the FIRST file's entries, where
    merge_entries() correctly ignored them as unknown segment_ids -- and
    the CLI printed the first file's captions as if they were the second
    file's. Two different videos, one of them silently answered with the
    other's transcript.
    """
    from dhvani.store import Store

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    wav_one = tmp_path / "a" / "clip.wav"
    wav_two = tmp_path / "b" / "clip.wav"

    fixtures, delta_path, _ = _prepare(tmp_path, wav_one, freq=200.0,
                                       tier1_text="escalated-A")
    _prepare(tmp_path, wav_two, freq=880.0, tier1_text="escalated-B")

    db = tmp_path / "t.db"

    def run_one(wav):
        assert main([
            str(wav), "--db", str(db), "--mode", "replay",
            "--fixtures", str(fixtures), "--delta-table", str(delta_path),
            "--budget", "10.0", "--escalate", "--reconcile",
        ]) == 0
        return json.loads(capsys.readouterr().out)

    first = run_one(wav_one)
    second = run_one(wav_two)

    assert [e["text"] for e in first] == ["escalated-A"]
    assert [e["text"] for e in second] == ["escalated-B"], (
        "the second file was answered with the first file's track"
    )
    # Different audio really is different segments, so this is not two
    # names for one thing.
    assert {e["segment_id"] for e in first} != {e["segment_id"] for e in second}

    # And the store holds them as two separate sources, each with its own
    # version history rather than one interleaved one.
    with Store(str(db)) as store:
        sources = [r["source_id"] for r in store.conn.execute(
            "SELECT DISTINCT source_id FROM tracks").fetchall()]
    assert len(sources) == 2, f"expected two distinct sources, got {sources}"
    assert all(s.startswith("clip.wav") for s in sources), (
        "the name must stay legible in the store"
    )


def test_cli_replay_escalation_reserves_no_money(tmp_path, capsys):
    """Replay makes no paid calls, so it must reserve nothing against the
    $20 ceiling -- escalate() used to price at the live rate regardless."""
    from dhvani.store import Store

    wav_path = tmp_path / "clip.wav"
    fixtures, delta_path, _ = _prepare(tmp_path, wav_path)
    db = tmp_path / "t.db"

    assert main([
        str(wav_path), "--db", str(db), "--mode", "replay",
        "--fixtures", str(fixtures), "--delta-table", str(delta_path),
        "--budget", "10.0", "--escalate", "--reconcile",
    ]) == 0
    capsys.readouterr()

    with Store(str(db)) as store:
        assert store.total_spend() == 0.0
