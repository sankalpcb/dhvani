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
