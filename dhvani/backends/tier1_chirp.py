"""Tier 1: Google Cloud Speech-to-Text v2 (Chirp 3), synchronous.

Phase 1 calls Chirp synchronously purely to populate delta_table.json. Phase 2
replaces this with asynchronous dynamic-batch submission plus reconciliation
(spec §7), which is where the interesting distributed-systems work lives.

Day-one spike (spec §14 risk 2): could not confirm whether Chirp 3 accepts
the DYNAMIC_BATCHING processing strategy -- see scripts/spike_chirp.py and
the Task 12 commit message for the blocked-on-environment result (no GCP
project, no credentials, and google-cloud-speech not installed in this
environment). USD_PER_MIN_DYNAMIC_BATCH below is therefore the *unverified*
brief rate, not a confirmed one. If a future run of the spike shows
dynamic batching is unsupported for chirp_3, set
USD_PER_MIN_DYNAMIC_BATCH = USD_PER_MIN_STANDARD -- the benchmark budget
would then rise from roughly $2.70 to roughly $14.40, still inside the $20
ceiling but with no margin.

Fix round 1 (post-review): cost_per_call() rounds each segment's duration up
to BILLING_INCREMENT_SEC before pricing it -- see that constant's docstring.
"""

USD_PER_MIN_DYNAMIC_BATCH = 0.003
USD_PER_MIN_STANDARD = 0.016

BILLING_INCREMENT_SEC = 15
"""Google Cloud Speech-to-Text has historically billed synchronous/batch
recognize requests in whole increments of this many seconds, rounded up per
request -- a 2-second segment is billed the same as a 15-second one.

UNVERIFIED: the day-one spike that would confirm Chirp 3's actual billing
granularity (spec §14 risk 2) could not run in this environment -- no GCP
project, no credentials, google-cloud-speech not installed (see
scripts/spike_chirp.py and the Task 12 commit history). This value must be
confirmed against real Google Cloud billing before any conclusions are
drawn from measured cost.

Rounding up here is deliberate and conservative, not a guess dressed up as
a fact: cost_per_call() feeds directly into Recorded's USD 20 ceiling check
(dhvani/backends/base.py), which records spend BEFORE the paid call runs
for exactly this reason -- an overstated cost fails safe, an understated
one lets real spend breach the ceiling while the ledger still reads under
budget. segmenter.segment() can produce segments well under 15s, and each
segment is one paid Chirp call, so charging exact wall-clock duration would
understate cost by up to 7x on short segments. Same principle as the
record-spend-before-the-call ruling in base.py: when true granularity is
unknown, round toward the number that keeps the ceiling honest.
"""


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
        duration_ms = segment.t_end_ms - segment.t_start_ms
        increment_ms = BILLING_INCREMENT_SEC * 1000
        # Ceiling division in integer arithmetic: round the segment's
        # duration up to the next whole billing increment before pricing
        # it. See BILLING_INCREMENT_SEC's docstring for why rounding up
        # (rather than exact duration) is required for the USD 20 ceiling
        # to fail closed.
        billable_ms = -(-duration_ms // increment_ms) * increment_ms
        minutes = billable_ms / 60000.0
        return USD_PER_MIN_DYNAMIC_BATCH * minutes

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
    return f"projects/{_project()}/locations/global/recognizers/_"


def _default_client(recognizer: str = ""):
    """Thin adapter over SpeechClient exposing recognize_pcm(pcm, lang) -> str."""
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech as cs

    class _Client:
        def __init__(self, recognizer: str):
            self._inner = SpeechClient()
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
                    model="chirp_3",
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
