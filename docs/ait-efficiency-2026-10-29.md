# Exact AIT inference with less repeated work and allocation

The frozen comparison completed **4,464 episodes** with identical predictions,
full decision traces/metrics and final model states. Across the six controller/
language cases, measured inference speed improves **1.39–1.54×** and median
per-episode traced peak allocation decreases **3.45–23.11%**. Whole-worker RSS
does not consistently decrease. This is a local resource improvement at the
same task behavior, with the measurement scopes below.

This follows the [observation-hypothesis comparison](observation-world-2026-10-29.md).
That new controller reduced waiting by sacrificing some Japanese/Chinese F1 and
added CPU cost. This step improves execution efficiency while preserving each
controller's own choices and outputs. It does not change the trained checkpoint,
decoding rule, observation hypotheses, acquisition costs, or floating precision.

## Changes

### Keep inference free of backward work

Fresh inference sequences previously computed phase features during construction
and recomputed them immediately during forward execution. They now defer that
work until the forward pass. Cached training and validation sequences still
refresh against the current phase basis on every forward.

Inference no longer retains per-boundary backward tensors for each score.
Observation calls keep only the context summary they need. CRF marginal/loss
evaluation skips the transition-gradient loop and gradient arrays when callers
do not request gradients. Training retains its original caches and derivatives.

### Reuse the attention-score workspace

The NumPy encoder now adds attention bias, subtracts row maxima, exponentiates,
and normalizes within its private score buffer. The sequence of arithmetic
operations and precision are preserved. Caller inputs and masks remain untouched;
attention diagnostics and requested backward tapes keep their original values.
This removes overlapping full attention arrays without changing attention's
quadratic complexity or retained diagnostic matrix size.

### Stage observation transactions instead of copying replay vectors

Each stock delivered observation previously deep-copied all latent-world weights
and every prior replay array, even when fitting was disabled. Stock recording
appends new arrays, and stock fitting reads old replay rows. A staged replay
container and small phase snapshot therefore suffice for normal arrivals.
Weights are copied only before an actual fitting attempt. On failure, replay,
weights, phase history, counters, and the model handle recover consistently.

Customized class or instance dynamics/phase hooks retain conservative deep
rollback. Tests inject failures after optimizer updates and replay eviction,
including arbitrary in-place mutations from custom hooks.

### Plan once for an immediate stock step

`step()` previously computed a plan and called `execute()`, which recomputed the
same plan to validate it. The stock immediate path now dispatches the freshly
computed plan. Separately supplied public decisions still receive full stale/
forged-plan validation. Overridden class, instance, and subclass execution hooks
keep their existing dispatch and validation behavior.

Planning and committing also obtain only the first pending token, without
allocating a list of every later token. No new stream-global feature cache or
unbounded retained history is introduced.

## Reproduce the comparison

The runner requires the previously completed pinned local artifacts in
`.cache/active-context-pilot-v1` and the pinned PUD source cache. It downloads
nothing and performs no fitting. It exports the exact previous commit
`ce446e34f941ae25970652f2450e5d2b68d8b23e` into an isolated baseline directory,
checks the exported package fingerprint, and selects the requested source before
importing the package in each worker.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/benchmark_ait_efficiency.py \
  --output .cache/ait-efficiency-pilot-v1 --prepare-only
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/benchmark_ait_efficiency.py \
  --output .cache/ait-efficiency-pilot-v1
```

The frozen plan binds both implementations, the worker and supporting scripts,
checkpoint/channel/corpus hashes, all input identities, runtime, and execution
order before measurement. Sixteen fresh workers each process the same 279
sentences, with the same warmed checkpoint restored for every sentence:

- Baseline and optimized sources.
- Fixed observation channel and Bayesian observation-regime controller.
- Three untraced timing repetitions and one separate allocation pass per pair.

That is **4,464 measured episodes**, with one unmeasured same-input warmup per
language in each worker. Both controllers use character cost 0.002; phase/error
drives and latent-weight fitting are disabled in both variants. Those settings
are frozen to match the previous study's operating point, without selecting a
new policy using test scores.

The equality gate requires identical tokens, boundary and nonwhitespace F1,
complete metrics and retained trace/candidate plans, and full final model-state
hashes across implementations and all timing/memory repetitions. Truncated
traces are rejected. This also checks that an optimization does not silently
reduce observation counts, erase information diagnostics, or skip world replay.

### Timing and memory mean different things

Speed uses only untraced episode wall time, including session construction,
planning, feature inference and observation processing. Shared checkpoint
restoration, explicit GC, scoring, hashes and serialization occur outside that
timer. All three repetitions are retained; no fastest run is selected.

Separate memory passes use per-episode `tracemalloc` peaks after restoring the
model. These include NumPy buffers registered with Python's tracing mechanism
and exclude preloaded weights, restoration and untracked native allocations.
Their elapsed times do not enter speed estimates.

Whole-worker process RSS high-water is also retained for all untraced timing
workers and separately for allocation workers. Darwin reports `ru_maxrss` in
bytes; Linux reports KiB. The runner converts explicitly and records a pre-loop
high-water floor. These RSS figures include startup, input/checkpoint parsing,
warmup, restoration, verification and scoring. They are not per-episode live RSS
or model-weight memory. Lower transient allocations need not lower this process
high-water mark.

### Synthetic replay resource probe

After the baseline export, compare observation bookkeeping in isolation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/probe_observation_transactions.py \
  --package-root .cache/ait-efficiency-baseline-ce446e3 \
  --label baseline --output .cache/ait-efficiency-pilot-v1/transaction-baseline.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/probe_observation_transactions.py \
  --package-root . --label optimized \
  --output .cache/ait-efficiency-pilot-v1/transaction-optimized.json
```

This uses explicit synthetic numeric fixtures with 150 or 2,048 initial replay
entries, five untraced repetitions of 64 arrivals, and a separate fresh-fixture
allocation pass. The first buffer grows to 214 entries; the second stays at its
2,048-entry capacity. Setup, encoder inference and fitting are excluded.
Initial/final model states and all six frozen world-weight signatures must match
across variants. These measurements describe bookkeeping, not end-to-end speed,
real-world observations, task accuracy or learning efficacy.

The completed synthetic probe preserved both initial/final states and frozen
weights across implementations:

| Initial replay entries | Median arrival ms, baseline → optimized | Bookkeeping speedup | Peak traced bytes, baseline → optimized |
| --- | ---: | ---: | ---: |
| 150 | 0.292812 → 0.021100 | 13.88× | 344,872 → 55,224 |
| 2,048 | 2.842248 → 0.031986 | 88.86× | 2,340,200 → 100,120 |

Peak allocation reductions are **83.99%** and **95.72%** within this isolated
bookkeeping scope. End-to-end inference has additional work and is measured
separately. Both five timing repetitions and the separate allocation pass remain
in the [raw baseline](../reports/ait-efficiency-2026-10-29/transaction-baseline.json)
and [raw optimized](../reports/ait-efficiency-2026-10-29/transaction-optimized.json)
receipts; their [verified comparison](../reports/ait-efficiency-2026-10-29/transaction_comparison.json)
binds them to the main frozen contract.

## Evidence limits

The checkpoint, supervised observation channel, and translated PUD test texts
were previously inspected. This is a controlled preservation and local CPU
resource comparison, not independent-corpus quality validation or a platform
performance guarantee. The new implementation preserves the previous Bayesian
controller's quality tradeoff; it does not repair that controller's lower F1.
The normal segmenter and training receive exact-equivalence regression checks,
while the repeated resource comparison uses the two specified active controllers.

## Completed results

Frozen contract:
`4411ead087ebf46d04f671acb3e8f43e7aff479d09b51bbf9f5ee6d549c9d381`.
All 16 fresh workers completed their 279 cases. All per-row prediction,
behavior/candidate/metric and final-state hashes match within each controller
across both implementations and every timing/allocation repetition. Boundary
and nonwhitespace F1 deltas are exactly zero.

| Controller | Language | Boundary F1, both versions | Speedup | Median traced peak, baseline → optimized (bytes) | Median reduction | Max peak reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fixed channel | EN | 99.7918% | 1.426× | 702,009 → 677,459 | 3.50% | 0.72% |
| Fixed channel | JA | 95.4072% | 1.394× | 486,421 → 418,956 | 13.87% | 14.80% |
| Fixed channel | ZH | 89.1269% | 1.417× | 374,520 → 287,964 | 23.11% | 14.49% |
| Bayesian regime | EN | 99.7938% | 1.411× | 866,717 → 836,814 | 3.45% | 0.67% |
| Bayesian regime | JA | 94.6421% | 1.492× | 662,446 → 567,570 | 14.32% | 1.05% |
| Bayesian regime | ZH | 88.5374% | 1.537× | 580,749 → 476,003 | 18.04% | 0.36% |

Speedup divides the median of three baseline language-total inference times by
the corresponding optimized median. Allocation percentages compare the same
per-language summaries across separate memory workers. Largest peaks barely
change in English and in the Bayesian controller; lower medians do not imply
uniformly lower worst-case memory. Attention diagnostics and bounded returned
traces remain available, and all model weights retain their original precision.

All timing repetitions, in seconds for the same 93 sentences per language:

| Controller / language | Baseline repeats | Optimized repeats |
| --- | --- | --- |
| Fixed / EN | 11.423, 11.703, 11.363 | 7.917, 8.012, 8.979 |
| Fixed / JA | 4.968, 5.121, 5.023 | 3.507, 3.604, 3.935 |
| Fixed / ZH | 2.382, 2.487, 2.392 | 1.649, 1.688, 1.811 |
| Bayesian / EN | 13.620, 13.807, 13.945 | 9.783, 9.525, 9.920 |
| Bayesian / JA | 7.261, 7.287, 7.335 | 4.884, 4.702, 4.991 |
| Bayesian / ZH | 4.417, 4.455, 4.450 | 2.896, 2.813, 2.930 |

Untraced whole-worker RSS high-water ranges are **67.05–76.88 MiB →
69.17–74.28 MiB** for the fixed controller and **69.58–69.91 MiB →
68.98–76.77 MiB** for the Bayesian controller. The ranges overlap, and the
Bayesian optimized maximum is higher. These lifecycle measurements do not
establish a process-wide resident-memory reduction. Every individual worker's
floor/high-water values, including separately labeled traced workers, are
retained in the summary.

## Validation and receipts

**552 tests passed, one skipped, nine subtests passed.** Exact reference checks
cover emissions, marginals, context vectors, encoder diagnostics, every backward
gradient and tape, training-phase refresh, empty inputs, zero layers, varied
heads/dtypes, read-only inputs and masks. Transaction checks cover actual fitting
cadence, capacity eviction, partially updated failed fits, phase failures,
custom hooks and retrying the same observation ID. Immediate steps match the
separately validated execution path, and overridden execution hooks still run.

The full completed study resumes without recomputing workers or changing saved
evidence. Result hashes survive JSON round trips; incomplete rows and runtime or
source changes reject. Public results contain identities, counts and hashes;
raw text, weights, worker inputs and detailed execution data stay local.

- [Frozen plan and provenance](../reports/ait-efficiency-2026-10-29/plan.json)
- [All metrics, repetitions and worker receipt hashes](../reports/ait-efficiency-2026-10-29/summary.json)
- [558 exact-matched controller/sentence score and behavior records](../reports/ait-efficiency-2026-10-29/paired_scores.jsonl)
- [Synthetic transaction resource comparison](../reports/ait-efficiency-2026-10-29/transaction_comparison.json)
