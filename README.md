# Dhvani

A confidence-gated caption cascade for code-mixed Indic speech.

Transcribe locally with a cheap model, score how much you distrust each segment, and spend
money on a better model **only** where the spend is measured to buy something. The last clause
is the whole project: the router refuses to escalate on a hunch, and a calibration harness
produces the evidence it consults.

```bash
git clone <this repo> && cd Youtube
uv sync --extra dev           # pytest only — no ML dependencies needed
make test                     # 433 tests, offline (4 skip without the data extra)
make track                    # captions a real Hindi clip, no model, no network
```

`make track` prints a caption track for committed audio using committed replay fixtures. No
GPU, no cloud credentials, no downloads. That property is the point, not a convenience — see
[Running it offline](#running-it-offline).

---

## The finding

The calibration ran for real on 2026-08-25: 150 IndicVoices utterances (50 each Hindi,
Kannada, Malayalam) scored locally, 124 escalated to Google Chirp 2 live, **$0.10 total**.

```json
"tier1": { "0.0-0.1": -17.91,  "0.1-0.2": -14.75 }
```

**Both deltas are negative.** Tier 1 transcribes this corpus *worse* than Tier 0, so the
shipped `delta_table.json` makes the router escalate nothing at all. `plan()` excludes non-positive delta by construction (invariant I3), so the table is
self-enforcing.

That is a result, not a broken run, and it is the one worth having. AI4Bharat's
IndicConformer 600M is purpose-built for Indic languages; Chirp 2 is general multilingual,
`chirp_3` is withdrawn for these locales, and `europe-west4` is the only region serving a
usable chirp model for them. The harness exists to ask *"is escalation worth paying for?"* and
for this pairing the answer is no — established for ten cents rather than assumed either way.

It was checked before being believed, and the check changed the metric. Three candidate
artifacts were measured rather than waved away: 0 of 124 Tier 1 transcripts were empty; digit
normalization (`3456` where the reference spells numbers out) affects only 10% of outputs; and
orthographic variance — `हाँ` against `हां`, chandrabindu against anusvara, the same word
scored as a total miss — was real. toWER now folds that class of difference away (see
`normalize_orthography`), which moved the deltas from −21.65 and −16.53 to the figures above.

So the numbers you see are *after* removing the obvious objection. **Roughly 4 points were
measurement noise; the remaining ~18 are genuine.**

### The digit gap

One measured defect is deliberately left in. Chirp writes `3456`; the IndicVoices references
spell the number out. toWER scores that as a total miss on every token involved, and it affects
about **10% of the calibration corpus**.

`normalize_orthography()` does not fold it, and the restraint is the point. Every rule in that
function changes how a word is *written* and never which word it is — NFC, candrabindu against
anusvara, zero-width joiners, punctuation. Mapping `3456` to साढ़े तीन हज़ार is a different kind
of operation: it needs a **per-language number lexicon**, including the grammar of how each
language says compound numbers, and a rule loose enough to do it is loose enough to collapse
नौ (nine) and नो, which are genuinely different words. A test pins that pair precisely so the
normalizer cannot drift into inflating the score it is supposed to measure.

**The direction matters.** This gap makes Tier 1 look *worse* than it is — the digit
mismatches are counted against Chirp, so closing it would move both deltas toward zero, not
further negative. The measured −17.91 and −14.75 are therefore a lower bound on Chirp's
quality, not an upper one. How much of the ~18 points is digits has not been quantified, and
the honest statement is that the finding "Tier 1 is worse here" survives the gap while its
*magnitude* does not.

It is also the clearest hypothesis for what Tier 2 is for. The repair prompt already carries
the instruction — *write numbers the way they were spoken, in words, not as digits* — so
measuring the `tier2` delta would partly be a test of whether a language model closes this gap
more cheaply than a hand-built lexicon per language.

### Risk skewed low, as predicted

Risk skewed hard low — 106 of 150 in the bottom bucket, median
0.0667, nothing at all above 0.6 — because `ctc_rnnt_disagreement` carries weight 0.4667 and
the two decoder heads agree on read speech. Only 2 of 10 buckets cleared the 20-sample floor,
and no larger sample would populate the top ones: this corpus contains no high-disagreement
audio.

## How it works

```
audio ─► segment ─► Tier 0 (local, free) ─► risk score ─► router ─┬─► ship / marked / review
                     IndicConformer 600M                          │
                                                                  ├─► Tier 1 (Chirp, paid)
                                                                  │    async, reconciled later
                                                                  │
                                                                  └─► Tier 2 (Gemini, free tier)
                                                                       repairs the best available
                                                                       hypothesis — Tier 1's if it
                                                                       ran, Tier 0's if it did not
```

The two upper tiers are gated the same way and are independent: Tier 2 does **not** sit
behind Tier 1. Wiring it there, as the original spec drew it, would make it unreachable —
nothing escalates to Tier 1 under the measured table, so nothing would ever arrive.

**Risk** is a weighted sum of decoder-level signals, not a confidence number the model hands
over:

| signal | weight | what it catches |
|---|---|---|
| `ctc_rnnt_disagreement` | 0.4667 | the two decoder heads disagreeing |
| `script_mix_entropy` | 0.2667 | Devanagari/Latin churn within a segment |
| `romanization_smell` | 0.2000 | Hindi written in Latin script |
| `short_segment` | 0.0667 | too little audio to trust |
| `mean_neg_logprob` | 0.0000 | *unavailable from this backend; carries no weight* |

Segments band into **ship** (< 0.30), **marked** (< 0.65) and **review**. Nothing ships
silently above the flag threshold.

**Routing** is a budget-constrained selection over measured deltas — a segment is escalated
only when `delta_table.json` records that *that tier* improved *that* risk bucket, and only
while budget remains. The rule is per tier, so a tier with no measured entry is simply never
selected. Both upper tiers are subject to it, including the free one — the question the
table answers is "does this help?", not "can we afford it?".

**Escalation is asynchronous.** Tier 1 results can arrive up to 24 hours later, out of order,
partially, or twice. `reconcile()` folds them into a new immutable track version through a
pure, idempotent merge. Six invariants are asserted under injected faults (partial delivery,
reordering, duplication, transient errors):

- **I1** no loss · **I2** idempotent merge · **I3** no negative-value escalation
- **I4** budget respected · **I5** determinism · **I6** convergence

**Quota** is guarded the same way money is, because it is the same problem in another
currency. `Store.reserve_quota()` claims a request against the daily free-tier cap in one
atomic statement before the call. The per-minute rate is handled separately by an in-process
token bucket: over-running the daily cap is unrecoverable until midnight Pacific, while going too fast
merely returns a retryable 429 — different failure semantics, different mechanisms. Pacing is
checked first, so being rate-limited never burns quota on a call that was not made.

**Money** is guarded by a single atomic statement. `Store.reserve_spend()` checks the ceiling
and records the spend in one SQL statement, always *before* the paid call — never after, so a
crash cannot under-count and let the $20 ceiling be breached on restart.

## Running it offline

```bash
make test    # 433 tests: no torch, no cloud SDK, no credentials, no network
make track   # captions samples/fleurs-hi-*.wav from committed fixtures
make bench   # cost/quality frontier from the measured delta table
```

Identity is content-addressed throughout: a segment's id is the SHA256 of its normalized PCM,
so a recorded fixture matches exactly the audio it was recorded from. That is why the demo
audio is committed rather than generated — a locally regenerated substitute would hash
differently and match nothing.

Replay mode never falls back to live. A missing fixture is a hard `FixtureMissing`, so an
offline run cannot silently become a billed one.

To see the async escalation machinery actually run:

```bash
dhvani samples/fleurs-hi-*.wav --delta-table samples/demo-delta-table.json \
       --budget 1.0 --escalate --reconcile
```

That uses a **deliberately fake** delta table, labelled as such in the file and asserted by a
test. It exists because the real table's negative deltas mean nothing ever escalates, so the
Phase 2 code would otherwise be undemonstrable. Never point a real run at it.

## Repairing what shipped

Tier 2 sends the best available hypothesis to Gemini for script, code-mix and digit repair on
its free tier. It is **not** wired strictly downstream of Tier 1 as the original spec drew it:
under the measured delta table nothing escalates to Tier 1, so a Tier 2 placed behind it would
be permanently unreachable. It repairs Tier 1's text when Tier 1 ran and Tier 0's when it did
not.

```bash
export GEMINI_MODEL=gemini-2.5-flash-lite     # required; never defaulted, see below
dhvani clip.wav --repair                      # repair what the delta table says is worth it
dhvani clip.wav --repair --repair-quota 0     # watch it degrade
```

The free tier imposes two limits, and they are handled differently on purpose. The **daily cap
is a consumable** — over-running it is unrecoverable until the reset — so it fails closed
through `Store.reserve_quota()`, one atomic statement taken before the call, the same
discipline the USD ceiling uses. The **per-minute rate is pacing** — exceeding it returns a
retryable 429 and consumes nothing — so it fails soft through an in-process token bucket.
Pacing is checked first, because reserving before pacing would burn a day's quota on calls
never made.

Exhaustion is a normal condition, not an error. The run completes, the caption ships with its
unrepaired text marked `repair_unavailable`, and the next run picks it up — no job record
needed, because "still wants repair" is already derivable from the content-addressed cache.

That degradation path is demonstrable offline with **no credentials at all**, because
degrading means not calling:

```
tier2: repaired 0, deferred 1 (quota_exhausted); deferred captions ship
       unrepaired and retry on a later run
```

### What the spike found

Run 2026-09-02 against published documentation. It settled two questions without a key, and
one of the answers was a live bug.

**The quota day rolled over eight hours early.** Google resets requests-per-day at midnight
**Pacific**, per project rather than per key. The code keyed the day by UTC, on the recorded
reasoning that a wrong boundary could only ever be *conservative*. That was backwards: UTC
midnight is late afternoon Pacific the previous day, so the gate reset a counter Google had
not, and would authorize up to **twice the cap** inside one Google day — the over-spending
direction the whole reservation discipline exists to prevent. Now keyed with `ZoneInfo`, not a
fixed offset, because Pacific moves an hour with daylight saving.

**A guessed model id would have been retired.** `gemini-2.0-flash` is shut down; the current
line is `gemini-3.7-flash`, `3.5-flash`, `2.5-flash-lite` and siblings. `GEMINI_MODEL` is read
from the environment and has no default, which is why the guess was never made.

**The daily limit is not knowable from documentation**, which inverts what the spec recorded.
Google no longer publishes free-tier figures — they are per-project facts behind an AI Studio
login — so the configured cap is a *local ceiling*, not the vendor's. Being wrong high is safe:
a refusal from Google defers that segment and the run continues, rather than crashing at
exactly the moment graceful degradation is supposed to happen.

The pattern is worth noting twice over. This project already found that a question assumed to
need account data — the billing increment — was published policy all along. This spike found
the mirror image: a number recorded as published turned out to be an account fact. Neither
direction was safe to assume, and both were cheap to check.

**Still unmeasured:** there is no `tier2` entry in `delta_table.json`, so `--repair` currently
changes nothing on real data. Measuring it costs nothing but needs a key. Full detail in
[the design](docs/superpowers/specs/2026-09-02-dhvani-tier2-repair-design.md) §8.

## Calibrating it yourself

Two decoupled phases, because one is slow and free and the other is fast and paid, and a human
should look at the histogram in between.

```bash
uv sync --extra models --extra data --extra cloud                   # only this path needs them
dhvani-calibrate collect  --langs hi-IN kn-IN ml-IN --per-lang 50   # local, free
dhvani-calibrate escalate --mode live --confirm                     # paid, ~$0.10
```

`collect` streams AI4Bharat IndicVoices (gated: accept its terms on the Hub first), transcribes
locally, and is resumable — a killed multi-hour run restarts without losing work. `escalate`
prints a cost estimate and refuses to spend without `--confirm`; `--dry-run` shows the estimate
and stops. A bucket with fewer than 20 samples is omitted from the table rather than published
as a noisy average.

## What is measured and what is not

| | status |
|---|---|
| Chirp 2 vs IndicConformer delta | **measured** — 124 live calls, 2026-08-25 |
| Dynamic-batch rate, $0.004/min | **published pricing** — 75% below Standard's $0.016 |
| Billing increment, 1 second | **published pricing** — the 15s increment is Speech-to-Text *On-Prem*, a different product |
| Risk weights | **from the spec**, not refit — the spike confirmed the heads are exposed and the weights stand |
| Buckets above 0.2 | **never measured** — this corpus produces no high-disagreement audio |
| Digit/number mismatch | **measured, deliberately unfixed** — ~10% of the corpus; needs a per-language number lexicon, so it is counted against Tier 1 rather than folded away. Its direction inflates Tier 1's measured deficit |
| Speaker-disjointness of the shipped table | **permanently unverifiable** — enforcement was wired in on 2026-09-02, after the run. The speaker metadata lived only in the calibration DB, which is gitignored and no longer exists; `results/scored.json` never carried the field and the committed fixtures cover only the demo audio. The guard protects future tables, not this one |
| Gemini repair delta | **never measured** — the tier is built and gated, but no `tier2` entry exists, so `--repair` changes nothing on real data |
| Gemini quota reset boundary | **published policy** — midnight Pacific, per project, confirmed 2026-09-02 |
| Gemini free-tier requests/day | **not knowable from docs** — Google no longer publishes it; it is a per-project fact behind an AI Studio login, so the configured cap is a local ceiling, not the vendor's |
| Gemini model id | **namespace confirmed, choice open** — `gemini-2.0-flash` is retired; a Flash-class 2.5/3.x id is read from `GEMINI_MODEL` rather than guessed |

Two honest limits sit in that table. The router has **no evidence about high-risk segments**,
because none occurred in the corpus — that is the ceiling on what the Tier 1 table can claim.
And it has **no evidence about Tier 2 at all**: the machinery is complete and the measurement
is not, which is the same order this project ran for Tier 1. Measuring Tier 2 is free, so what
is missing there is an API key rather than a budget.

The three Gemini rows below the delta also record which *kind* of fact each one is, because
that turned out to matter: the reset boundary was published policy the code had wrongly
assumed needed an account, and the requests-per-day figure was the reverse.

## At production scale

[The scaling sketch](docs/superpowers/specs/2026-09-02-dhvani-production-scaling.md) does the
capacity and cost math at 720,000 audio-hours/day — written, not built, as spec §12 asks.

At that rate the system sees **4,207 segments/second**. Tier 0 compute dominates at roughly
**$27,100 per million audio-hours**, against **$256,800** for routing everything to Chirp, so
the cascade is ~9.3x cheaper at the measured 0% escalation rate and stops paying for itself
somewhere near 90%. `segment_id` being a SHA256 turns out to be a near-ideal Bigtable row key
by accident — uniform by construction, immutable, so hot rows are cacheable with no
invalidation — at the cost of scan locality, which is why `tracks` is keyed separately.

Every figure is labelled measured, published or assumed. The single largest source of error is
named up front: Tier 0's real-time factor came off one laptop with unpinned thread count, and
the dominant cost line inherits its error.

## Layout

```
dhvani/            29 modules — segmenter, scorer, router, escalate, reconcile,
                   repair, quota, store, CLIs
tests/             35 files, 433 tests
fixtures/          committed replay fixtures (Tier 0 and Tier 1)
samples/           committed demo audio + attribution; the illustrative delta table
docs/superpowers/  design specs, the three implementation plans, the scaling sketch
delta_table.json   the measured routing table
results/scored.json  phase 1 output, so buckets can be re-derived without paying again
```

Three console scripts: `dhvani` (transcribe), `dhvani-bench` (frontier report),
`dhvani-calibrate` (measure).

Optional extras, all genuinely optional: `models` (torch/transformers/onnxruntime), `cloud`
(google-cloud-speech), `data` (datasets), `repair` (google-genai). The test suite passes with
none of them installed — 4 of the 433 skip without `data`, and the skip is deliberate rather
than incidental.

## Attribution

Demo audio is from [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0, © Google
LLC), decoded and normalized to mono 16 kHz PCM16. Calibration data is
[AI4Bharat IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) (CC-BY-4.0).
Full details in [`samples/ATTRIBUTION.md`](samples/ATTRIBUTION.md).

Indic-language audio is transcribed in `europe-west4`: `asia-south1` offers no usable chirp
model for these locales, which is a data-residency consideration worth stating rather than
discovering later.
