"""Day-one spike: does Chirp 3 accept the dynamic-batch processing strategy?

Spec §14 risk 2. This determines the commercial rate used by Tier 1: dynamic
batch is $0.003/min versus $0.016/min standard, a 5.3x difference deciding
whether the benchmark costs roughly $2.70 or roughly $14.40.

Requires: `google-cloud-speech` installed (the optional `cloud` extra),
GCP application-default credentials, a real GCP project ID, and a GCS
bucket holding a sample WAV file.

Task 12 (2026-08-22) could not run this: `GOOGLE_CLOUD_PROJECT` was unset,
no `gcloud` CLI was present, no application-default credentials were
configured, and `google-cloud-speech` was not installed in the venv (it
lives behind the optional `cloud` extra, deliberately left uninstalled so
the test suite runs without cloud dependencies per Goal G5). Per the task
brief, that means BLOCKED-ON-ENVIRONMENT: do not install the extra, do not
attempt network calls, do not create cloud resources, and do not guess
what the API would return. dhvani/backends/tier1_chirp.py therefore keeps
USD_PER_MIN_DYNAMIC_BATCH = 0.003 unverified pending a real run of this
script by someone with GCP access.
"""

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cs

PROJECT = "REPLACE_WITH_YOUR_GCP_PROJECT_ID"
client = SpeechClient()

req = cs.BatchRecognizeRequest(
    recognizer=f"projects/{PROJECT}/locations/global/recognizers/_",
    config=cs.RecognitionConfig(
        auto_decoding_config=cs.AutoDetectDecodingConfig(),
        model="chirp_3",
        language_codes=["hi-IN"],
    ),
    processing_strategy=cs.BatchRecognizeRequest.ProcessingStrategy.DYNAMIC_BATCHING,
    files=[cs.BatchRecognizeFileMetadata(uri="gs://YOUR_BUCKET/sample.wav")],
    recognition_output_config=cs.RecognitionOutputConfig(
        inline_response_config=cs.InlineOutputConfig()
    ),
)

try:
    op = client.batch_recognize(request=req)
    print("DYNAMIC_BATCHING: accepted ->", op.operation.name)
except Exception as exc:
    print(f"DYNAMIC_BATCHING: REJECTED -> {type(exc).__name__}: {exc}")
