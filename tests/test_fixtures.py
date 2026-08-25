"""The committed replay fixtures must match the committed generator.

A stranger clones the repo, runs `make track`, and gets a caption track with
no model, no credentials and no network -- that is goal G5 made concrete
rather than merely tested. It works only because a fixture is named for the
SHA256 of the audio it was recorded from, and `scripts/make_sample_wav.py`
reproduces that audio byte for byte.

Nothing enforced that link. Editing the generator -- a different duration, a
different sweep, a different envelope -- silently renames every segment, and
the fixtures stop matching audio nobody can regenerate any more. The suite
would stay green, because every other test builds its own fixtures in
tmp_path; only a person running the CLI would find out.
"""

import importlib.util
import json
import pathlib

import soundfile as sf

from dhvani.audio import normalize
from dhvani.backends.tier0_conformer import Tier0Conformer
from dhvani.ids import variant_slug
from dhvani.segmenter import segment as split

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _generator():
    """Load scripts/make_sample_wav.py, which is not an importable package.

    Safe to exec because the module writes nothing at import; the write is
    behind __main__. That guard is load-bearing for this test.
    """
    path = ROOT / "scripts" / "make_sample_wav.py"
    spec = importlib.util.spec_from_file_location("make_sample_wav", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tier0_dir():
    # Constructing Tier0Conformer imports no ML dependency -- see
    # tests/test_tier0.py's G5 guard -- so this stays offline.
    return ROOT / "fixtures" / "tier0" / variant_slug(
        Tier0Conformer(lang="hi").variant_key)


def _regenerated_segments(tmp_path):
    wav = tmp_path / "sample.wav"
    _generator().write_sample(str(wav))
    samples, rate = sf.read(str(wav))
    return split(normalize(samples, rate))


def test_the_generator_writes_nothing_when_imported(tmp_path, monkeypatch):
    """_generator() execs the script. If the write ever escapes its __main__
    guard, this test would scribble sample.wav into whatever directory pytest
    ran from."""
    monkeypatch.chdir(tmp_path)
    _generator()
    assert list(tmp_path.iterdir()) == []


def test_every_segment_of_the_sample_has_a_committed_tier0_fixture(tmp_path):
    segments = _regenerated_segments(tmp_path)
    assert segments, "the generator produced no segments at all"

    missing = [seg.segment_id for seg in segments
               if not (_tier0_dir() / f"{seg.segment_id}.json").exists()]
    assert not missing, (
        "regenerating sample.wav produces segments with no committed fixture, "
        "so `make track` cannot run offline from a clean clone.\n"
        f"  fixture dir: {_tier0_dir().relative_to(ROOT)}\n"
        f"  unmatched:   {missing}\n"
        "Either scripts/make_sample_wav.py changed (re-record with "
        "`dhvani sample.wav --mode record`) or the fixtures were not committed."
    )


def test_the_committed_fixtures_are_shaped_like_backend_output(tmp_path):
    """A fixture Recorded cannot read is as good as a missing one, and the
    failure would only appear when someone ran the CLI."""
    for seg in _regenerated_segments(tmp_path):
        payload = json.loads(
            (_tier0_dir() / f"{seg.segment_id}.json").read_text(encoding="utf-8"))
        assert set(payload) == {"text", "signals"}, payload
        assert isinstance(payload["text"], str)
        assert isinstance(payload["signals"], dict)
        for name, value in payload["signals"].items():
            assert isinstance(value, (int, float)), (name, value)
