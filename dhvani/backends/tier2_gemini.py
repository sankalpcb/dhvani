"""Tier 2: Gemini repair of the best available hypothesis (design §1.1).

NOT strictly downstream of Tier 1. Spec §3 draws repair_tier2 after
asr_tier1, but under the measured delta table nothing escalates to Tier 1,
so a Tier 2 wired that way would be permanently unreachable and
demonstrable only behind samples/demo-delta-table.json. This backend
repairs whichever hypothesis exists -- Tier 1's when present, Tier 0's
otherwise -- which is what the spec's own contract describes:
`(segment, [hypothesis]) -> text` takes a LIST.

Free tier, so cost_per_call() is 0.0 and the scarce resource is quota,
not money. See dhvani/quota.py; the spend ledger stays truthful about
dollars and gains nothing false.

Fixture safety: this exposes transcribe(segment) so backends/base.py's
Recorded wraps it unchanged. That is sound because the input -- the best
upstream hypothesis -- is itself content-addressed and keyed
(segment_id, tier), so under a fixed POLICY_ID the segment_id determines
the input. A hypothesis cannot vary independently of the key, so replay
fixtures cannot collide the way FIX ROUND 2 (I2/I3) describes.
"""

import os

# UNVERIFIED (design §8): the model id is read from the environment rather
# than hardcoded, exactly as SPEECH_MODEL is, because the correct Gemini
# model name and its free-tier RPM have NOT been confirmed against the
# vendor. A day-one spike settles both before the smoke run. Do not
# promote this default to a fact.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "")

# Bump with POLICY_ID. The prompt changes the output for byte-identical
# PCM, so it is part of variant_key and changing it invalidates fixtures
# rather than silently altering what is cached.
PROMPT_VERSION = "r1-2026-09-02"

PROMPT = """You are correcting an automatic speech-recognition transcript of \
{lang} speech that may mix in English words.

Rules:
- Preserve the speaker's words exactly. Do not translate, summarise or rephrase.
- Write numbers the way they were spoken, in words, not as digits.
- Use the script the word was spoken in; do not transliterate a genuine \
English word into the local script.
- Return only the corrected transcript, with no commentary.

Transcript:
{text}"""


class Tier2Gemini:
    name = "tier2"

    def __init__(self, hypothesis_source, client=None, lang: str = "hi-IN",
                 model: str = None):
        """Stores its inputs and builds nothing.

        Constructing this must import no SDK and touch no network, so a
        replay-mode caller can build one with `google-genai` absent
        (goal G4). The client is constructed lazily on first use, the
        same lazy-property shape Tier1Chirp and Tier0Conformer use.

        hypothesis_source is `segment_id -> str`: the best available
        upstream text. Injected rather than reaching into the store, so
        the backend stays testable without a database.
        """
        self._hypothesis_source = hypothesis_source
        self._injected_client = client
        self._built_client = None
        self.lang = lang
        self.model = GEMINI_MODEL if model is None else model

    @property
    def variant_key(self) -> str:
        """Everything that changes the output for byte-identical PCM."""
        return f"model={self.model};lang={self.lang};prompt={PROMPT_VERSION}"

    @property
    def _client(self):
        if self._injected_client is not None:
            return self._injected_client
        if self._built_client is None:
            self._built_client = _default_client(self.model)
        return self._built_client

    def cost_per_call(self, segment) -> float:
        """Always 0.0 -- the free tier costs no money.

        This is not a placeholder for a price to be filled in later. If
        the free tier is ever exhausted the answer is to wait for the
        reset, never to enable billing (design N4), so there is no paid
        path for this number to describe.
        """
        return 0.0

    def transcribe(self, segment) -> dict:
        text = self._hypothesis_source(segment.segment_id) or ""
        if not text.strip():
            # Nothing to repair. Returning early matters: quota is the
            # scarce resource here, and spending a request to have a model
            # confirm that "" is still "" is the one waste worth coding
            # around.
            return {"text": "", "signals": {"repair_skipped": "empty_input"}}
        repaired = self._client.repair(text, self.lang)
        return {"text": str(repaired), "signals": {}}


def _default_client(model: str):
    """Thin adapter over the Gemini SDK exposing repair(text, lang) -> str.

    Imported inside the function so the module imports cleanly with no
    `google-genai` installed (goal G4).
    """
    from google import genai

    class _Client:
        def __init__(self, model: str):
            if not model:
                raise RuntimeError(
                    "GEMINI_MODEL is unset. The model id is deliberately not "
                    "defaulted -- see design §8, it awaits the day-one spike."
                )
            self._model = model
            self._genai = genai.Client()

        def repair(self, text: str, lang: str) -> str:
            response = self._genai.models.generate_content(
                model=self._model,
                contents=PROMPT.format(lang=lang, text=text),
                config={"temperature": 0.0},
            )
            return (response.text or "").strip()

    return _Client(model)
