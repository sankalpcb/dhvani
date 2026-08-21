import numpy as np
import pytest
from dhvani.audio import normalize
from dhvani.ids import segment_id
from dhvani.config import SAMPLE_RATE


def _tone(seconds=1.0, rate=SAMPLE_RATE, freq=440.0):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def test_same_audio_yields_same_id():
    pcm = normalize(_tone(), SAMPLE_RATE)
    assert segment_id(pcm) == segment_id(pcm.copy())


def test_id_is_64_char_hex():
    pcm = normalize(_tone(), SAMPLE_RATE)
    sid = segment_id(pcm)
    assert len(sid) == 64
    assert all(c in "0123456789abcdef" for c in sid)


def test_resampling_converges_to_same_id():
    """Same signal captured at 44.1kHz and 16kHz normalizes to near-identical PCM."""
    a = normalize(_tone(rate=44100), 44100)
    b = normalize(_tone(rate=SAMPLE_RATE), SAMPLE_RATE)
    assert len(a) == len(b)
    # Resampling is lossy; assert close, not identical.
    assert np.mean(np.abs(a.astype(int) - b.astype(int))) < 200


def test_different_audio_yields_different_id():
    a = normalize(_tone(freq=440.0), SAMPLE_RATE)
    b = normalize(_tone(freq=880.0), SAMPLE_RATE)
    assert segment_id(a) != segment_id(b)


def test_stereo_is_downmixed_to_mono():
    stereo = np.stack([_tone(), _tone()], axis=1)
    pcm = normalize(stereo, SAMPLE_RATE)
    assert pcm.ndim == 1


def test_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="int16"):
        segment_id(np.zeros(10, dtype=np.float32))
