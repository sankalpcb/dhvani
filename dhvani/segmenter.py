"""Energy-based voice activity detection producing caption-sized segments."""

from dataclasses import dataclass, field

import numpy as np

from dhvani.config import SAMPLE_RATE
from dhvani.ids import segment_id as compute_id

FRAME_MS = 30
SILENCE_RATIO = 0.02   # frame RMS below this fraction of peak counts as silence
MIN_SILENCE_MS = 400   # silence shorter than this does not split a segment


@dataclass(frozen=True)
class Segment:
    segment_id: str
    t_start_ms: int
    t_end_ms: int
    pcm: np.ndarray = field(compare=False, repr=False)


def _voiced_frames(pcm: np.ndarray, frame_len: int) -> np.ndarray:
    n = len(pcm) // frame_len
    if n == 0:
        return np.zeros(0, dtype=bool)
    frames = pcm[: n * frame_len].reshape(n, frame_len).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    peak = rms.max()
    if peak == 0:
        return np.zeros(n, dtype=bool)
    return rms > (SILENCE_RATIO * peak)


def _runs(voiced: np.ndarray, gap_frames: int) -> list[tuple[int, int]]:
    """Contiguous voiced runs, bridging silences shorter than gap_frames."""
    runs, start, silence = [], None, 0
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= gap_frames:
                runs.append((start, i - silence + 1))
                start = None
    if start is not None:
        runs.append((start, len(voiced)))
    return runs


def segment(pcm: np.ndarray, min_ms: int = 2000, max_ms: int = 8000) -> list[Segment]:
    """Split int16 PCM into caption-sized segments. Deterministic."""
    frame_len = SAMPLE_RATE * FRAME_MS // 1000
    voiced = _voiced_frames(pcm, frame_len)
    gap_frames = MIN_SILENCE_MS // FRAME_MS

    max_frames = max_ms // FRAME_MS
    out: list[Segment] = []

    for run_start, run_end in _runs(voiced, gap_frames):
        cursor = run_start
        while cursor < run_end:
            chunk_end = min(cursor + max_frames, run_end)
            s0, s1 = cursor * frame_len, chunk_end * frame_len
            piece = pcm[s0:s1]
            if len(piece) == 0:
                break
            out.append(Segment(
                segment_id=compute_id(piece),
                t_start_ms=int(s0 * 1000 / SAMPLE_RATE),
                t_end_ms=int(s1 * 1000 / SAMPLE_RATE),
                pcm=piece,
            ))
            cursor = chunk_end

    return out
