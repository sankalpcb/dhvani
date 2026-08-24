"""Generate a short 16kHz mono WAV for the Chirp spike. No audio sourcing needed."""

import numpy as np
import soundfile as sf

SR = 16000
t = np.linspace(0, 3.0, SR * 3, endpoint=False)
# A speech-shaped sweep with an amplitude envelope — not real speech, but a
# valid non-silent LINEAR16 payload, which is all the spike needs.
sig = 0.4 * np.sin(2 * np.pi * np.cumsum(np.linspace(120, 900, SR * 3)) / SR)
sig *= 0.5 * (1 - np.cos(2 * np.pi * np.arange(len(sig)) / len(sig)))
sf.write("sample.wav", sig.astype(np.float32), SR, subtype="PCM_16")
print(f"wrote sample.wav ({len(sig)/SR:.1f}s, {SR}Hz mono PCM16)")
