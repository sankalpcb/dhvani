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


def _write_wav(path, seconds=6.0):
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 200 * t)
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
