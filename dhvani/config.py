"""Central configuration. Bump POLICY_ID when normalization, weights, or
the delta table change — it invalidates the content-addressed cache."""

SAMPLE_RATE = 16000
POLICY_ID = "p1-2026-08-22"
MAX_SPEND_USD = 20.0

# Risk function weights. Fixed by offline grid search (Task 11), not trained.
#
# FIX ROUND 2 (minor 6): mean_neg_logprob is weighted 0.0 because the
# signal does not exist yet. IndicConformer does not expose a per-token
# logprob (see backends/tier0_conformer.py and spec §14: the day-one spike
# that would confirm the decoder interface is blocked on a gated model
# repo), so Tier0Conformer hardcodes it to 0.0 for every segment.
#
# It previously carried 0.25. A signal permanently pinned at 0.0 holding a
# quarter of the weight capped the maximum risk any real segment could
# score at 0.75 -- only 0.10 above TAU_FLAG, and reachable only with all
# four remaining signals simultaneously at their absolute maximum. The
# `review` band was therefore effectively unreachable, silently defeating
# spec §6.2's primary output contract that nothing below tau_flag ships
# unexamined.
#
# The 0.25 is redistributed proportionally across the four live signals
# (each old weight divided by their 0.75 total), so their relative
# importance is unchanged and the total is exactly 1.0 -- the literals
# below are those quotients rounded to the nearest double that sums
# exactly. This is a renormalization, not a re-tuning: no new weighting
# opinion is expressed, and no model was changed or trained.
#
# When the blocked spike lands and a real logprob becomes available, the
# weights must be re-derived rather than reverted --
# tests/test_scorer.py::test_unavailable_signal_carries_no_weight fails
# loudly if the dead signal silently reacquires weight before then.
RISK_WEIGHTS = {
    "ctc_rnnt_disagreement": 0.4666666666666667,
    "mean_neg_logprob": 0.0,
    "script_mix_entropy": 0.26666666666666666,
    "romanization_smell": 0.2,
    "short_segment": 0.06666666666666667,
}

TAU_SHIP = 0.30
TAU_FLAG = 0.65

# Spec §11 recorded a free tier of "1,000 requests/day". The spike of
# 2026-09-02 could NOT confirm that, and the reason matters: Google no
# longer publishes free-tier RPD/RPM figures at all. ai.google.dev/
# gemini-api/docs/rate-limits now says limits "can be viewed in Google AI
# Studio" -- they are per-PROJECT account facts behind a login, not
# published policy. Reports also suggest free quotas were cut sharply in
# December 2025, so 1,000 may be off by an order of magnitude.
#
# This is therefore a LOCAL CEILING, not the vendor's limit. It is safe to
# be wrong high: repair() treats the vendor's own refusal as degradation
# rather than a crash, so an over-generous value costs a wasted call, not a
# failed run. Set --repair-quota to the real number from AI Studio when it
# is known; --repair-quota 0 demonstrates the exhaustion path.
GEMINI_DAILY_QUOTA = 1000

# Likewise unpublished. 15 is a conservative pacing figure, not a measured
# ceiling; the token bucket smooths bursts rather than enforcing a contract.
GEMINI_RPM = 15
