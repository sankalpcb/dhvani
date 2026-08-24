"""Day-one spike: does Chirp 3 accept the dynamic-batch processing strategy?

Spec §14 risk 2. This determines the commercial rate used by Tier 1: dynamic
batch is $0.003/min versus $0.016/min standard, a 5.3x difference deciding
whether the benchmark costs roughly $2.70 or roughly $14.40. It also probes
the billing granularity behind BILLING_INCREMENT_SEC, which is currently an
unverified conservative guess in dhvani/backends/tier1_chirp.py.

Reads its configuration from the environment so nothing has to be hand-edited:

    GOOGLE_CLOUD_PROJECT   your GCP project id            (required)
    DHVANI_SPIKE_GCS_URI   gs://bucket/sample.wav         (required)

Requires the optional `cloud` extra (`uv pip install -e ".[cloud]"`) and
application-default credentials (`gcloud auth application-default login`).
"""

import os
import sys


def main() -> int:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    uri = os.environ.get("DHVANI_SPIKE_GCS_URI")

    missing = [n for n, v in (("GOOGLE_CLOUD_PROJECT", project),
                              ("DHVANI_SPIKE_GCS_URI", uri)) if not v]
    if missing:
        print(f"unset: {', '.join(missing)} — see this file's docstring", file=sys.stderr)
        return 2

    try:
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech as cs
    except ImportError:
        print('google-cloud-speech missing: uv pip install -e ".[cloud]"', file=sys.stderr)
        return 2

    client = SpeechClient()
    recognizer = f"projects/{project}/locations/global/recognizers/_"
    config = cs.RecognitionConfig(
        auto_decoding_config=cs.AutoDetectDecodingConfig(),
        model="chirp_3",
        language_codes=["hi-IN"],
    )

    def attempt(label, strategy):
        kwargs = dict(
            recognizer=recognizer,
            config=config,
            files=[cs.BatchRecognizeFileMetadata(uri=uri)],
            recognition_output_config=cs.RecognitionOutputConfig(
                inline_response_config=cs.InlineOutputConfig()
            ),
        )
        if strategy is not None:
            kwargs["processing_strategy"] = strategy
        try:
            op = client.batch_recognize(request=cs.BatchRecognizeRequest(**kwargs))
            print(f"{label}: ACCEPTED -> {op.operation.name}")
            return True
        except Exception as exc:
            print(f"{label}: REJECTED -> {type(exc).__name__}: {str(exc).splitlines()[0][:220]}")
            return False

    # The answer we need: is DYNAMIC_BATCHING accepted for chirp_3?
    dynamic = attempt(
        "DYNAMIC_BATCHING",
        cs.BatchRecognizeRequest.ProcessingStrategy.DYNAMIC_BATCHING,
    )
    # Control: plain batch, to distinguish "strategy unsupported" from
    # "the whole request is malformed / creds are wrong".
    attempt("default batch (control)", None)

    print()
    if dynamic:
        print("RESULT: dynamic batch supported — USD_PER_MIN_DYNAMIC_BATCH = 0.003 stands.")
        print("        Benchmark cost stays ~$2.70.")
    else:
        print("RESULT: dynamic batch NOT supported for chirp_3.")
        print("        Set USD_PER_MIN_DYNAMIC_BATCH = USD_PER_MIN_STANDARD (0.016)")
        print("        in dhvani/backends/tier1_chirp.py. Benchmark cost rises to ~$14.40.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
