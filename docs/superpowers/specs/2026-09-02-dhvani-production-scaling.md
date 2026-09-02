# Dhvani — Production Scaling Sketch (M6)

**Status: written, not built.** Spec §12 asks for an architecture document with capacity and
cost math at 720,000 audio-hours/day. Nothing here is deployed, and no number below was
produced by running the system at that scale. What the POC does supply is a set of *measured*
per-unit constants, and the whole point of this document is to be explicit about which
constants those are and which figures are extrapolation on top of them.

Target: **720,000 audio-hours/day**.

---

## 1. Where the numbers come from

Extrapolation is only as good as its inputs, so these are separated by provenance. The
"assumed" rows are the ones to attack first if a figure here looks wrong.

| input | value | provenance |
|---|---|---|
| Mean segment duration | 7.13 s | **measured** — 884.1 s over 124 segments, calibration run 2026-08-25 |
| Tier 1 audio actually sent | 884.5 s | **measured** — Google `api/request_sizes`, 28,304,734 B ÷ 32 kB/s, read 2026-09-02 |
| Tier 1 latency | mean 2.34 s, p50 ∈ [1.05, 2.10) s, p99 ∈ [4.19, 8.39) s | **measured** — Google `api/request_latencies`, 124 samples, server-side |
| Tier 1 success rate | 124/125 attempts reached the service | **measured** — `api/request_count`, zero 5xx for `Recognize` |
| Tier 0 transcribe time | 1.85 s per utterance | **measured** — 2026-08-25, on an 8-core Apple M1 |
| Tier 0 real-time factor | ≈ 0.26 core-s per audio-s | **derived** — 1.85 / 7.13; see the caveat below |
| Tier 1 rate | $0.004 / audio-minute | **published** — Dynamic Batch, 75% below Standard's $0.016 |
| Tier 1 billing increment | 1 s | **published** — Speech-to-Text V2; not yet corroborated against an invoice |
| Cloud Run vCPU | $0.000024 / vCPU-s | **published**, region-dependent, dated 2026-09 |
| Cloud Run memory | $0.0000025 / GiB-s | **published**, region-dependent, dated 2026-09 |
| Bigtable SSD node | $0.65 / node-hour | **published**, region-dependent, dated 2026-09 |
| Bigtable SSD storage | $0.17 / GB-month | **published**, region-dependent, dated 2026-09 |
| Pub/Sub | $40 / TiB delivered | **published**, dated 2026-09 |
| Dedup cache hit rate | **unknown** | **assumed 0** throughout — see §7 |

**The RTF caveat is the load-bearing one.** 0.26 is one process on one developer laptop, and
the thread count of that process was never pinned. If IndicConformer was quietly using four M1
cores, the true per-core figure is nearer 1.04 and every compute number in §5 is ~4× optimistic.
Server-class inference, batching, and GPU/TPU serving all push the other way. **This single
constant should be re-measured under controlled parallelism before anyone quotes §5 to a
stranger.** It is the largest source of error in this document, larger than every pricing
assumption combined.

---

## 2. Decomposing the target

```
720,000 audio-hours/day
  = 2.592 × 10⁹ audio-seconds/day
  = 30,000 audio-seconds per wall-clock second      (30,000× real time)
  = 363.5 million segments/day at 7.13 s/segment
  =   4,207 segments/second
```

Everything downstream is a function of **4,207 segments/second**.

---

## 3. POC → production mapping

Spec §12 fixes the target column. The third column is why, which is the part worth arguing
about.

| POC | Production | why this one |
|---|---|---|
| SQLite `segments` / `hypotheses` | Bigtable keyed by `segment_id` | key is already a SHA256 — see §4 |
| In-process queue | Pub/Sub | escalation is already fire-and-forget; at-least-once matches I2 |
| `jobs` table | Cloud Tasks | per-task retry/backoff already modelled by `MAX_JOB_ATTEMPTS` |
| Local invocation | Cloud Run (scale-to-zero) | request-shaped, bursty, no local state to keep |
| Local files | GCS | audio must not travel through the queue — see §6 |
| `make bench` | Dataflow | re-scoring a corpus is embarrassingly parallel over `segment_id` |

The migration is unusually mechanical for one reason: **the POC's invariants are already the
ones a distributed system needs.** I2 (idempotent merge) is not a nicety that becomes
important later — it is precisely what makes Pub/Sub's at-least-once delivery safe. I1 (no
loss) and I6 (convergence) are what make Cloud Tasks retries safe. I5 (determinism) is what
makes Dataflow reprocessing safe. The chaos suite was written against injected faults on one
machine, and those faults are the everyday behaviour of the target architecture.

---

## 4. Shard-key analysis: `segment_id`

`segment_id = SHA256(normalized PCM)`.

**Distribution is uniform by construction.** Bigtable shards by contiguous row-key range into
tablets, so the failure mode to avoid is a key whose high-order bytes correlate with arrival
time — a timestamp or monotonic id prefix sends every write to the last tablet and pins one
node at 100% while the rest idle. A SHA256 has no such correlation: the leading byte is
uniform over 0x00–0xFF, so 4,207 writes/second spread evenly across tablets with no
pre-splitting, no salting, and no key design work. This is a real and slightly lucky property —
the hash was chosen for content-addressed dedup and cache correctness, and even key
distribution came free.

**The cost of that uniformity is locality.** Consecutive segments of one video land in
unrelated tablets, so "fetch this video's caption track in order" degrades into hundreds of
scattered point reads. Content-addressing and range-scan locality are directly opposed and no
single key gives both. The resolution is two tables, which the POC schema already anticipates:

- `segments`, keyed by `segment_id` — content-addressed, immutable, the dedup and cache layer.
- `tracks`, keyed by `(video_id, t_start_ms)` — ordered, scannable, storing *references* to
  `segment_id` rather than transcripts.

A track read becomes one range scan plus a batched multi-get, and `video_id` prefixes carry
their own mild hotspotting risk (a single viral video) which is handled the same way as §4.1.

### 4.1 Hot keys on popular deduplicated segments

Dedup means a segment appearing in many videos is *one row*. A channel intro reused across
100,000 videos/day is one key read ~1.16 times/second — unremarkable. The tail is what to
reason about: a segment shared across 10 million videos/day is ~116 reads/second on a single
row, and a single Bigtable row lives on exactly one tablet server, so a sufficiently viral
segment concentrates load no amount of node-adding will spread.

Two properties make this bounded rather than dangerous:

1. **Reads are trivially cacheable, and cache invalidation does not exist here.** The value at
   a content-addressed key can never change — a different transcript implies different audio
   implies a different key. So a read-through cache (Memorystore, or in-process LRU per Cloud
   Run instance) takes infinite TTL with no coherence protocol. The hottest keys are exactly
   the ones a cache serves best, so hotness and cacheability rise together.
2. **Writes cannot conflict.** Every writer of a given key writes byte-identical content, so
   the write is an idempotent blind put. No read-modify-write, no check-and-mutate, no
   contention — concurrent writers racing on a hot key produce the same row.

The residual risk is a cold cache after a mass restart, where the top-N keys hit Bigtable
simultaneously. Standard mitigation: staggered instance rollout, plus request coalescing so N
concurrent misses on one key produce one backend read.

---

## 5. Capacity and cost

### 5.1 Tier 0 compute — the dominant cost

```
2.592 × 10⁹ audio-s/day × 0.26 core-s/audio-s = 6.74 × 10⁸ core-s/day
                                              ÷ 86,400 s
                                              ≈ 7,800 cores steady-state
```

Per million audio-hours: `10⁶ × 3600 × 0.26 = 9.36 × 10⁸ core-seconds = 260,000 core-hours`.

| | per million audio-hours |
|---|---|
| vCPU @ $0.000024/vCPU-s | $22,464 |
| Memory @ 2 GiB/worker | $4,680 |
| **Tier 0 total** | **≈ $27,100** |

That is the on-demand Cloud Run upper bound. A fleet with a 7,800-core floor is not an
on-demand workload — committed-use discounts or GKE spot capacity would cut it substantially,
and scale-to-zero earns nothing at a constant floor. **Cloud Run is the wrong runtime for
steady-state Tier 0**; it is the right one for the request-shaped API tier in front of it. The
spec's mapping table is correct about the API and over-general about the workers.

### 5.2 Tier 1 — the escalation bill

At $0.004/audio-minute, one million audio-hours = 60 million audio-minutes = **$240,000** at
100% escalation, before rounding.

The 1-second increment adds a predictable premium: rounding a 7.13 s mean segment up costs
~0.5 s, i.e. **+7.0%**. Note what governs that — the premium is roughly
`increment / (2 × mean_segment_duration)`, so it is a *segmentation* design parameter, not
just a pricing footnote. Short segments are punished disproportionately: at a 1 s mean segment
the same increment would cost ~50%. It also explains the POC's over-reservation cleanly — the
old, wrong 15 s increment implies `15 / 7.13 ≈ 2.1×`, matching the observed over-reserve.

So Tier 1 ≈ **$256,800 × escalation rate** per million audio-hours.

### 5.3 Storage and queue

Per million audio-hours: `10⁶ × 3600 / 7.13 ≈ 505 million segments`. At ~600 B/row
(`segment_id`, transcript, risk, timings, variant) that is **≈ 303 GB**, or ~$52/month at
$0.17/GB-month — and it accumulates, so this is the one line that grows with corpus age
rather than with daily rate.

Throughput: 4,207 writes/s for segments, a similar rate for hypotheses, plus 4,207 dedup
lookups/s ≈ **12,600 ops/s**. At ~10,000 QPS per SSD node that is 2 nodes of raw capacity;
call it **4 nodes** for p99 headroom and replication — ~$62/day, ~$87 per million audio-hours.

Queue traffic is negligible **provided audio never enters it** (§6): 4,207 msg/s × 250 B ≈
91 GB/day ≈ $3.31/day even at 100% escalation.

### 5.4 Cost per million audio-hours at several budget settings

The budget knob is the escalation rate the router's delta table and budget ceiling jointly
permit.

| escalation rate | Tier 1 | Tier 0 | storage + queue | **total** | vs Chirp-only |
|---|---|---|---|---|---|
| **0%** — the currently measured table | $0 | $27,100 | ~$500 | **$27,600** | **9.3× cheaper** |
| 1% | $2,568 | $27,100 | ~$500 | **$30,200** | 8.5× |
| 5% | $12,840 | $27,100 | ~$500 | **$40,400** | 6.4× |
| 10% | $25,680 | $27,100 | ~$500 | **$53,300** | 4.8× |
| 25% | $64,200 | $27,100 | ~$500 | **$91,800** | 2.8× |
| 100% | $256,800 | $27,100 | ~$500 | **$284,400** | 0.9× (worse) |
| *Chirp-only baseline* | $256,800 | — | ~$500 | **$257,300** | 1.0× |

Two things this table says that are easy to miss:

**The cascade stops paying for itself somewhere near 90% escalation** — past that, Tier 0
compute is pure overhead on top of a bill you were going to pay anyway. The architecture is
only justified by escalating *selectively*, which is the router's entire job.

**At the measured escalation rate of 0%, the honest headline is not "matches Chirp quality
9.3× cheaper" — it is better and worse than that.** Better, because on the calibration corpus
Tier 0 *beat* Chirp (deltas −17.91 and −14.75), so this is not a quality-for-cost trade at
all. Worse, because that was 150 utterances of IndicVoices read speech, and this document's
target is YouTube-scale audio, which is spontaneous, noisy, code-mixed and musical. **The
delta table has no evidence about that audio.** A production deployment must re-run
calibration on real traffic before trusting a 0% escalation rate — the frontier could invert
entirely.

---

## 6. Why audio never enters the queue

Measured mean Tier 1 request payload: **228 KB** (`api/request_sizes`, 124 samples). Two
designs for the escalation message:

| message | volume at 4,207 msg/s | Pub/Sub cost |
|---|---|---|
| GCS URI + metadata, ~250 B | 91 GB/day | **$3.31/day** |
| Inline audio, 228 KB | 82.9 TB/day | **~$3,015/day** |

**~900× more expensive**, before considering that Pub/Sub's 10 MB message cap makes long
segments unrepresentable and that redelivery re-ships the payload every time. Audio goes to
GCS; the queue carries the `segment_id` and a URI. This is the one place where the POC's
in-process queue is actively misleading as a model — passing PCM by reference costs nothing on
one machine and is the difference between $3/day and $3,000/day across a network.

---

## 7. The dedup lever, and why it is unquantified

Every figure above assumes a **0% cache hit rate** — every segment transcribed as if never
seen. Content-addressing makes the dedup mechanism free and exact, and at scale it is
plausibly the single largest cost lever: Tier 0 cost scales directly with `(1 − hit_rate)`, so
a 30% duplicate rate removes ~$8,100 per million audio-hours, more than every storage and
queue line combined.

YouTube-shaped corpora should carry real duplication — re-uploads, Shorts remixes of the same
source, channel intros and outros, licensed music beds, and the same clip quoted across many
commentary videos. **This is not measured and is not estimated here.** A guessed hit rate
would silently become the headline number, and spec §9.1 already lists cache hit rate as a
metric to report rather than assume. The correct next step is to measure it on a real sample,
not to model it.

Note the interaction with §4.1: high dedup and hot keys are the *same phenomenon*. A corpus
with enough duplication to save real money is precisely one with hot rows, so the cache in
§4.1 is not just a latency optimization — it is how the savings are realized.

---

## 8. Latency and the async payoff

| path | latency | source |
|---|---|---|
| Tier 0 (serving path) | 1.85 s / utterance | measured, M1 laptop |
| Tier 1 p50 | 1.05 – 2.10 s | measured, server-side |
| Tier 1 p99 | 4.19 – 8.39 s | measured, server-side |
| Tier 1 worst observed | 8.39 – 16.78 s | measured, 1 of 124 |

Tier 1's p99 is 4–8× its p50 — a long tail, on read speech, at trivial concurrency. Under
load, with quota pressure and the transient 503s the retry path already handles, it would be
worse.

**The async design is what makes that tail irrelevant.** Because escalation is fire-and-forget
with a 24-hour reconciliation window, Tier 1 latency never sits on the serving path: captions
ship from Tier 0 immediately and are revised later into a new immutable track version. A
synchronous cascade would inherit Tier 1's p99 *and* its failure rate on every request. This
is the strongest systems argument the project has, and it is worth stating plainly: the
reconciliation machinery is not overhead justifying itself, it is what decouples user-visible
latency from the slowest, least reliable, most expensive component.

---

## 9. What this sketch does not establish

- **The RTF constant (§1).** Measured on one laptop with unpinned thread count. Everything in
  §5.1 inherits its error, and §5.1 is the dominant cost line.
- **The cache hit rate (§7).** Assumed zero. Could move total cost by tens of percent.
- **Escalation rate on real audio (§5.4).** The 0% row comes from 150 read-speech utterances.
  YouTube audio is a different distribution and the delta table has no evidence about it.
- **Anything above risk bucket 0.2.** The corpus produced no high-disagreement audio, so the
  router has never been evaluated where it would matter most.
- **Bigtable QPS per node.** Published guidance for ~1 KB rows; real throughput depends on
  row size, read/write mix and locality, none of which were benchmarked.
- **Every price.** Published rates as of 2026-09, region-dependent, and cloud pricing moves.
- **Tier 2.** M5 (Gemini repair with a quota-aware rate limiter) is unbuilt, so no Tier 2 cost
  or quota behaviour appears anywhere above.
