import numpy as np
from dhvani.config import SAMPLE_RATE
from dhvani.segmenter import segment


def _speech(seconds):
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float64)


def _silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds))


def _pcm(x):
    return (np.clip(x, -1, 1) * 32767).round().astype(np.int16)


def test_silence_only_yields_no_segments():
    assert segment(_pcm(_silence(5.0))) == []


def test_single_speech_burst_yields_one_segment():
    audio = np.concatenate([_silence(0.5), _speech(3.0), _silence(0.5)])
    segs = segment(_pcm(audio))
    assert len(segs) == 1


def test_two_bursts_separated_by_silence_yield_two_segments():
    audio = np.concatenate([_speech(3.0), _silence(1.5), _speech(3.0)])
    segs = segment(_pcm(audio))
    assert len(segs) == 2


def test_long_burst_is_split_at_max_duration():
    segs = segment(_pcm(_speech(20.0)), max_ms=8000)
    assert len(segs) >= 3
    assert all(s.t_end_ms - s.t_start_ms <= 8000 for s in segs)


def test_segments_are_time_ordered_and_non_overlapping():
    audio = np.concatenate([_speech(3.0), _silence(1.5), _speech(3.0)])
    segs = segment(_pcm(audio))
    for a, b in zip(segs, segs[1:]):
        assert a.t_end_ms <= b.t_start_ms


def test_segment_ids_are_populated_and_unique():
    audio = np.concatenate([_speech(3.0), _silence(1.5), _speech(4.0)])
    segs = segment(_pcm(audio))
    ids = [s.segment_id for s in segs]
    assert all(len(i) == 64 for i in ids)
    assert len(set(ids)) == len(ids)
