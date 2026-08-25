# Dhvani

A confidence-gated caption cascade for code-mixed Indic speech.

Transcribe locally with a cheap model, score how much you distrust each segment, and spend
money on a better model **only** where the spend is measured to buy something. The last clause
is the whole project: the router refuses to escalate on a hunch, and a calibration harness
produces the evidence it consults.

```bash
git clone <this repo> && cd Youtube
uv sync --extra dev           # pytest only — no ML dependencies needed
make test                     # 397 tests, offline
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
"tier1": { "0.0-0.1": -21.65,  "0.1-0.2": -16.53 }
```

**Both deltas are negative.** Tier 1 transcribes this corpus *worse* than Tier 0 — mean toWER
45.6 against 25.0 — so the shipped `delta_table.json` makes the router escalate nothing at
all. `plan()` excludes non-positive delta by construction (invariant I3), so the table is
self-enforcing.

That is a result, not a broken run, and it is the one worth having. AI4Bharat's
IndicConformer 600M is purpose-built for Indic languages; Chirp 2 is general multilingual,
`chirp_3` is withdrawn for these locales, and `europe-west4` is the only region serving a
usable chirp model for them. The harness exists to ask *"is escalation worth paying for?"* and
for this pairing the answer is no — established for ten cents rather than assumed either way.

It was checked before being believed. Three candidate artifacts were measured, not waved away:
0 of 124 Tier 1 transcripts were empty; digit normalization (`3456` where the reference spells
numbers out) affects only 10% of outputs; and orthographic variance — `हाँ` against `हां`,
chandrabindu against anusvara, the same word scored as 100% WER — is real but small.
Normalizing NFC, chandrabindu, zero-width joiners and punctuation moves the delta from −20.7
to −17.3. **Roughly 3 points is measurement noise and roughly 17 is genuine.**

A second prediction held. Risk skewed hard low — 106 of 150 in the bottom bucket, median
0.0667, nothing at all above 0.6 — because `ctc_rnnt_disagreement` carries weight 0.4667 and
the two decoder heads agree on read speech. Only 2 of 10 buckets cleared the 20-sample floor,
and no larger sample would populate the top ones: this corpus contains no high-disagreement
audio.

## How it works

```
audio ─► segment ─► Tier 0 (local, free) ─► risk score ─► router ─┬─► ship / marked / review
                     IndicConformer 600M                          │
                                                                  └─► Tier 1 (Chirp, paid)
                                                                       async, reconciled later
```

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
only when `delta_table.json` records that Tier 1 improved that risk bucket, and only while
budget remains.

**Escalation is asynchronous.** Tier 1 results can arrive up to 24 hours later, out of order,
partially, or twice. `reconcile()` folds them into a new immutable track version through a
pure, idempotent merge. Six invariants are asserted under injected faults (partial delivery,
reordering, duplication, transient errors):

- **I1** no loss · **I2** idempotent merge · **I3** no negative-value escalation
- **I4** budget respected · **I5** determinism · **I6** convergence

**Money** is guarded by a single atomic statement. `Store.reserve_spend()` checks the ceiling
and records the spend in one SQL statement, always *before* the paid call — never after, so a
crash cannot under-count and let the $20 ceiling be breached on restart.

## Running it offline

```bash
make test    # 397 tests: no torch, no cloud SDK, no credentials, no network
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

The last row is the honest limit of the current table: the router has no evidence about
high-risk segments, because none occurred.

## Layout

```
dhvani/            28 modules — segmenter, scorer, router, escalate, reconcile, store, CLIs
tests/             31 files, 397 tests
fixtures/          committed replay fixtures (Tier 0 and Tier 1)
samples/           committed demo audio + attribution; the illustrative delta table
docs/superpowers/  design spec and the three implementation plans
delta_table.json   the measured routing table
results/scored.json  phase 1 output, so buckets can be re-derived without paying again
```

Three console scripts: `dhvani` (transcribe), `dhvani-bench` (frontier report),
`dhvani-calibrate` (measure).

Optional extras, all genuinely optional: `models` (torch/transformers/onnxruntime), `cloud`
(google-cloud-speech), `data` (datasets). The test suite passes with none of them installed.

## Attribution

Demo audio is from [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0, © Google
LLC), decoded and normalized to mono 16 kHz PCM16. Calibration data is
[AI4Bharat IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) (CC-BY-4.0).
Full details in [`samples/ATTRIBUTION.md`](samples/ATTRIBUTION.md).

Indic-language audio is transcribed in `europe-west4`: `asia-south1` offers no usable chirp
model for these locales, which is a data-residency consideration worth stating rather than
discovering later.
