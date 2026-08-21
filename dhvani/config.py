"""Central configuration. Bump POLICY_ID when normalization, weights, or
the delta table change — it invalidates the content-addressed cache."""

SAMPLE_RATE = 16000
POLICY_ID = "p1-2026-08-22"
MAX_SPEND_USD = 20.0

# Risk function weights. Fixed by offline grid search (Task 11), not trained.
RISK_WEIGHTS = {
    "ctc_rnnt_disagreement": 0.35,
    "mean_neg_logprob": 0.25,
    "script_mix_entropy": 0.20,
    "romanization_smell": 0.15,
    "short_segment": 0.05,
}

TAU_SHIP = 0.30
TAU_FLAG = 0.65
