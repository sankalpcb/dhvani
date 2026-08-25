"""Tier 1: Google Cloud Speech-to-Text v2 (Chirp 3), synchronous.

Phase 1 calls Chirp synchronously purely to populate delta_table.json. Phase 2
replaces this with asynchronous dynamic-batch submission plus reconciliation
(spec §7), which is where the interesting distributed-systems work lives.

Spec §14 risk 2 is CLOSED. scripts/spike_chirp.py ran against a real GCP
project on 2026-08-24 (commit aeb09b5): DYNAMIC_BATCHING is ACCEPTED, on
europe-west4/chirp_2 and us-central1/chirp. The contingency this paragraph
used to describe -- fall back to USD_PER_MIN_STANDARD if the strategy is
rejected -- did not come to pass.

The pricing numbers themselves were settled separately, on 2026-08-25, and
NOT the way this file long claimed. Both constants below were wrong, and
both were checkable all along from Google's published pricing rather than
from any invoice: the per-minute rate was 25% low, and the billing
increment was the one belonging to Speech-to-Text ON-PREM rather than the
V2 API this module actually calls. See each constant's docstring.

This paragraph previously called the spike blocked on environment (no GCP
project, no credentials, google-cloud-speech not installed). That was true
when written; the run above superseded it, and the `cloud` extra is
installed in this venv today.

Fix round 1 (post-review): cost_per_call() rounds each segment's duration up
to BILLING_INCREMENT_SEC before pricing it -- see that constant's docstring.
"""

import os

USD_PER_MIN_DYNAMIC_BATCH = 0.004
"""Dynamic Batch: documented as 75% below the Standard tier, so 0.25 x 0.016.

Was 0.003, taken from the brief and never checked against Google's published
pricing. That under-priced every call by 25% -- the dangerous direction,
because cost_per_call() feeds the ceiling check that runs BEFORE each paid
call, and an understated cost lets real spend pass a ledger that still reads
under budget.
"""
USD_PER_MIN_STANDARD = 0.016
"""Standard real-time/batch tier, the published list rate."""

# Region and model, both settled by scripts/spike_chirp.py on 2026-08-24.
#
# `chirp_3` is NOT usable: asia-south1 returns
#   403 ... on model chirp_3 locale hi-IN. It is no longer generally available.
# and us-central1 / europe-west4 report it does not exist at all. `global`
# holds no chirp model of any generation, so the recognizer path must be
# regional and the client needs a regional endpoint.
#
# Verified working: europe-west4/chirp_2 and us-central1/chirp. asia-south1
# (Mumbai) offers neither, so Indian-language audio is transcribed outside
# India — a data-residency point worth stating rather than discovering later.
SPEECH_LOCATION = os.environ.get("DHVANI_SPEECH_LOCATION", "europe-west4")
SPEECH_MODEL = os.environ.get("DHVANI_SPEECH_MODEL", "chirp_2")

BILLING_INCREMENT_SEC = 1
"""Speech-to-Text V2 rounds each request up to the nearest 1 second.

This was 15, carried as an explicitly UNVERIFIED guess whose docstring
insisted only real Cloud Billing line items could settle it. That premise
was wrong twice over. Billing granularity is published pricing policy, not a
per-account fact, so no invoice was ever needed -- and 15 seconds is the
increment for Speech-to-Text ON-PREM, a different product with its own SKU.
Dhvani calls V2 (google.cloud.speech_v2.SpeechClient), so the code was
applying the wrong product's rounding.

Direction of the error, since this file cares about that: 15s rounding
OVER-reserved. At the 7.1s mean segment of the first real calibration run it
more than doubled every estimate -- 2010 billed seconds where V2 charges
939 -- so the USD 20 ceiling was quietly worth about USD 42 of real audio
and every --confirm estimate overstated by the same factor. Safe, but not
honest, and it partially masked the opposite error in
USD_PER_MIN_DYNAMIC_BATCH above.

Rounding up is still correct and still load-bearing: sub-second audio is
billed as a whole second, and Recorded (backends/base.py) checks this exact
number against the ceiling before every live call.
"""


def cost_for_duration_ms(duration_ms: int) -> float:
    """USD billed for one Tier 1 call on a segment of this duration.

    THE single Tier 1 cost model. Anything that needs to price a Tier 1
    call — the backend itself, and report.frontier(), which builds the
    cost/quality chart people choose a budget from — must call this, not
    re-derive a price from USD_PER_MIN_DYNAMIC_BATCH.

    Fix round 2 (I1): report.py had its own copy of the rate and priced
    candidates at exact wall-clock (rate * duration / 60000), while this
    module rounded up to a whole billing increment. The frontier therefore
    understated real spend by up to 7.5x on a 2000 ms segment ($0.000100
    charted vs $0.000750 billed), so a run planned from the chart would
    hit BudgetExceeded partway through. One function, one rate, one
    rounding rule.

    Rounding up to BILLING_INCREMENT_SEC is ceiling division in integer
    arithmetic — see that constant's docstring for why rounding up is
    required for the USD 20 ceiling to fail closed.
    """
    increment_ms = BILLING_INCREMENT_SEC * 1000
    billable_ms = -(-int(duration_ms) // increment_ms) * increment_ms
    return USD_PER_MIN_DYNAMIC_BATCH * (billable_ms / 60000.0)


class Tier1Chirp:
    name = "tier1"

    def __init__(self, client=None, lang: str = "hi-IN", recognizer: str = ""):
        """Stores the injected client (or None), lang, and recognizer.
        Does NOT call _default_client() -- constructing a Tier1Chirp must
        import nothing and touch no network, so replay-mode callers (e.g.
        dhvani.cli) can build one with zero cloud dependencies installed.
        The real client is constructed lazily on first use via the
        `_client` property, mirroring Tier0Conformer's `_model` property.
        """
        self._injected_client = client
        self._built_client = None
        self.lang = lang
        self.recognizer = recognizer

    @property
    def variant_key(self) -> str:
        """Everything about this backend that changes the output for
        byte-identical PCM. lang and recognizer both do — see FIX ROUND 2
        (I2/I3) in backends/base.py."""
        return f"lang={self.lang};recognizer={self.recognizer}"

    @property
    def _client(self):
        """The client to call, constructing and caching it on first access.

        An injected client (tests, or any caller that already has one) is
        used as-is. Otherwise _default_client() runs exactly once, on the
        first transcribe() call -- never at construction time.
        """
        if self._injected_client is not None:
            return self._injected_client
        if self._built_client is None:
            self._built_client = _default_client(self.recognizer)
        return self._built_client

    def cost_per_call(self, segment) -> float:
        return cost_for_duration_ms(segment.t_end_ms - segment.t_start_ms)

    def transcribe(self, segment) -> dict:
        text = self._client.recognize_pcm(segment.pcm, self.lang)
        return {"text": str(text), "signals": {}}


def _recognizer_path(recognizer: str) -> str:
    """Resolve the full Chirp recognizer resource path to call.

    A non-empty `recognizer` (e.g. a real Chirp recognizer resource ID a
    caller configured -- Phase 2's async path will need one) is used
    as-is. Empty (the default) falls back to the global wildcard
    recognizer built from GOOGLE_CLOUD_PROJECT via _project(), which fails
    closed if that env var is unset. This function has no google-cloud
    imports, so it is directly testable without the `cloud` extra
    installed.
    """
    if recognizer:
        return recognizer
    return f"projects/{_project()}/locations/{SPEECH_LOCATION}/recognizers/_"


def _default_client(recognizer: str = ""):
    """Thin adapter over SpeechClient exposing recognize_pcm(pcm, lang) -> str."""
    from google.api_core.client_options import ClientOptions
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech as cs

    class _Client:
        def __init__(self, recognizer: str):
            # Chirp models live in regional endpoints; the default global
            # endpoint serves none of them.
            self._inner = SpeechClient(client_options=ClientOptions(
                api_endpoint=f"{SPEECH_LOCATION}-speech.googleapis.com"))
            self._recognizer = recognizer

        def recognize_pcm(self, pcm, lang: str) -> str:
            req = cs.RecognizeRequest(
                recognizer=_recognizer_path(self._recognizer),
                config=cs.RecognitionConfig(
                    explicit_decoding_config=cs.ExplicitDecodingConfig(
                        encoding=cs.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=16000,
                        audio_channel_count=1,
                    ),
                    model=SPEECH_MODEL,
                    language_codes=[lang],
                ),
                content=pcm.tobytes(),
            )
            resp = self._inner.recognize(request=req)
            return " ".join(
                r.alternatives[0].transcript for r in resp.results if r.alternatives
            )

    return _Client(recognizer)


def _project() -> str:
    import os

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("set GOOGLE_CLOUD_PROJECT before using the live Chirp backend")
    return project
