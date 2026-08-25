# Committed demo audio

The audio in this directory is redistributed from a third-party corpus. It is
here so that a clean clone can run the offline replay workflow end to end:
replay fixtures are named for the SHA256 of a segment's normalized PCM, so
they only match audio reproduced byte for byte, and a generated substitute
would not do.

## `fleurs-hi-12091698556182716328.wav`

- **Source:** FLEURS (Few-shot Learning Evaluation of Universal
  Representations of Speech), `google/fleurs` on the Hugging Face Hub,
  configuration `hi_in`, split `dev`/`validation`, FLEURS id `1607`,
  original filename `12091698556182716328.wav`.
- **Reference transcript:** कुतूहल गाँव में आधे घंटे टहलना समय की हानि नहीं है।
  ("A half-hour stroll in Kutuhal village is not a waste of time.")
- **Duration:** 4.86 s, mono 16 kHz PCM16.
- **Licence:** Creative Commons Attribution 4.0 International (CC-BY-4.0).
- **Attribution:** © Google LLC, released under CC-BY-4.0. FLEURS is
  described in Conneau et al., "FLEURS: Few-shot Learning Evaluation of
  Universal Representations of Speech".
- **Modifications:** decoded and normalized to mono 16-bit PCM at 16 kHz by
  `dhvani.audio.normalize`, then written as a WAV. No other change; the
  samples are otherwise as published.

CC-BY-4.0 permits redistribution, including modified copies, provided the
source is credited and changes are indicated. Both are done above.

## Why not IndicVoices

The calibration corpus for this project is AI4Bharat IndicVoices, and the
demo clip was meant to come from it. `ai4bharat/IndicVoices` is a gated
dataset: its card is public, but the data returns HTTP 403 until an account
accepts the terms on the dataset page. FLEURS is CC-BY-4.0 and ungated, and
serves the same purpose for a demo clip. Nothing about the calibration
harness changed — `dhvani/corpus.py` still streams IndicVoices.
