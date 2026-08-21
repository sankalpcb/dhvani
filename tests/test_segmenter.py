import numpy as np
from dhvani.config import SAMPLE_RATE
from dhvani.ids import segment_id
from dhvani.segmenter import FRAME_MS, segment


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


def test_trailing_subthreshold_silence_does_not_extend_segment():
    """Regression for Finding 1: `_runs` must not fold trailing silence shorter
    than MIN_SILENCE_MS into the final run. If it did, identical speech with
    different trailing silence would hash to different segment_ids, silently
    breaking the content-addressed dedup cache."""
    audio = np.concatenate([_speech(2.0), _silence(0.2)])
    segs = segment(_pcm(audio))
    assert len(segs) == 1
    # Speech ends at exactly 2000ms. VAD operates at FRAME_MS granularity, so the
    # boundary may round up by at most one frame (the speech/silence-mixed frame),
    # but must not absorb the full 200ms of trailing silence.
    assert 2000 <= segs[0].t_end_ms <= 2000 + FRAME_MS


def test_segment_pcm_matches_source_slice_and_rehashes_to_same_id():
    audio = np.concatenate([_silence(0.5), _speech(3.0), _silence(0.5)])
    pcm = _pcm(audio)
    segs = segment(pcm)
    assert len(segs) == 1
    seg = segs[0]
    start_sample = seg.t_start_ms * SAMPLE_RATE // 1000
    end_sample = seg.t_end_ms * SAMPLE_RATE // 1000
    assert np.array_equal(seg.pcm, pcm[start_sample:end_sample])
    assert segment_id(seg.pcm) == seg.segment_id


def test_segment_ids_are_populated_and_unique():
    audio = np.concatenate([_speech(3.0), _silence(1.5), _speech(4.0)])
    segs = segment(_pcm(audio))
    ids = [s.segment_id for s in segs]
    assert all(len(i) == 64 for i in ids)
    assert len(set(ids)) == len(ids)
