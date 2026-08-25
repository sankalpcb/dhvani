"""Generate a short 16kHz mono WAV for the Chirp spike. No audio sourcing needed.

Also the audio the committed replay fixtures were recorded from. The fixture
filename is the SHA256 of this signal's normalized PCM, so the two only line
up while this function produces byte-identical output -- which is what
tests/test_fixtures.py pins.
"""

import numpy as np
import soundfile as sf

SR = 16000
SECONDS = 3.0


def sample_signal(seconds: float = SECONDS, rate: int = SR) -> np.ndarray:
    """A speech-shaped sweep with an amplitude envelope -- not real speech, but
    a valid non-silent LINEAR16 payload, which is all the spike needs."""
    n = int(rate * seconds)
    sig = 0.4 * np.sin(2 * np.pi * np.cumsum(np.linspace(120, 900, n)) / rate)
    sig *= 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))
    return sig.astype(np.float32)


def write_sample(path: str = "sample.wav") -> str:
    sf.write(path, sample_signal(), SR, subtype="PCM_16")
    return path


if __name__ == "__main__":
    write_sample()
    print(f"wrote sample.wav ({SECONDS:.1f}s, {SR}Hz mono PCM16)")
