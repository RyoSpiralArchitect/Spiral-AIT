# AIT observation hypotheses: infer the boundary and the world together

The [previous grounded loop](active-context-2026-10-27.md) learned an encoder-
feature world, but its forecasts reached acquisition planning only through a
scalar prediction-error drive. Online fitting changed no final predictions in
that comparison. This step adds a **task-outcome world** directly to planning:
uncertainty about which observation hypothesis generated the next raw boundary
state, alongside uncertainty about the boundary label.

It does not establish a benefit from the older latent-feature weights or phase
drives. Every condition in this comparison freezes those weights and disables
the two drives. The new online state is Bayesian inference over fixed observation
hypotheses; it does not train the encoder or invent runtime gold labels.

## Joint observation model

For each uncommitted boundary independently, retain the four probabilities
`q(R,Y) = (null/Y0, null/Y1, informative/Y0, informative/Y1)`. Here **Y** is a
boundary label and **R** is a persistent hypothesis for observations of that
same boundary. Initial marginals are independent, with a declared informative-
hypothesis prior of one half in the exploratory study.

The informative likelihood is the earlier frozen, labeled-prefix channel
`L_y(z|s,a)`. The comparison hypothesis is label-independent:
`Q(z|s,a) = (L_0(z|s,a) + L_1(z|s,a))/2`. It is a balanced null hypothesis,
not an empirical detector of source reliability or correctness. State **s**
contains the previous raw CRF bin and observed age; action **a** requests more
characters; outcome **z** is the newly observed raw CRF bin.

For every candidate action, compute from the same joint distribution:

```
p(z)        = sum_{r,y} q(r,y) L_r(z|y,s,a)
q(r,y | z)  = q(r,y) L_r(z|y,s,a) / p(z)
expected risk = sum_z p(z) min(q(Y=0|z), q(Y=1|z))
label information = H(Y) - sum_z p(z) H(Y|z)
model information = H(R) - sum_z p(z) H(R|z)
```

Both the hypothetical tree and actual filtering carry the **complete joint
posterior**. Reconstructing independent marginals after each observation loses
the correlation between label and hypothesis. The `fixed_mixture` ablation
deliberately makes that reset, retaining the label marginal while forgetting
regime evidence and correlation; it is a separate memoryless model assumption.

Risk and label information influence acquisition. Model information is recorded
separately; its optional reward coefficient is **zero** throughout the study.
At a fresh label prior of exactly one half, the balanced hypotheses predict the
same first outcome distribution. One acquisition then cannot identify the
hypothesis, even though it can correlate the two hidden variables. Subsequent
observations of the same boundary can distinguish their sequential predictions.

## Causal execution and bounds

`plan()` sees only the current observed frame. It neither consumes text nor
updates beliefs, phase history, replay, or weights. After a complete acquisition,
each supported uncommitted boundary uses its actually observed outcome. New,
unsupported, EOF-truncated, and reframed boundaries initialize from the current
raw marginal and declared hypothesis prior. No stream-global hypothesis is
learned by pooling correlated neighboring boundary outcomes.

The trace records a prequential outcome probability, log loss, multiclass Brier
score, and before/after joint beliefs for **one predeclared focused boundary**
per complete supported acquisition. Scores are computed from the belief before
the delivery. Signed realized label and hypothesis entropy changes may be
negative. `observation_world_updates` counts per-boundary assimilation operations;
they are correlated, need not change a posterior numerically, and are not
independent training samples.

Acquired receipts survive later processing failures; retry cannot reread or
apply Bayesian evidence twice. Existing window, trace, and acquisition bounds
remain. A pre-existing planning error is also repaired: hypothetical depth is
now capped by the **remaining actual acquisition budget**, so a read cannot be
valued only for a second read that execution would forbid. This repair applies
to all comparison modes, including the fixed channel.

## API and authored replay

```python
from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import (
    ActiveContextConfig, StringTextSource,
)

session = model.active_text_session(fitted_channel, config=ActiveContextConfig(
    observation_model="bayesian_regime",
    informative_prior=0.5,
    learn_observation_regime=True,
    model_epistemic_weight=0.0,
    learn_dynamics=False,
))
result = session.run(StringTextSource(provided_text))
```

Ordinary segmentation and the fixed-channel session remain defaults. The new
authored replay loads verified historical weights but explicitly fingerprints
the current implementation separately:

```bash
python scripts/demo_observation_world.py --study .cache/active-context-pilot-v1
```

`--mode fixed_mixture`, `--mode fixed_channel`, and `--mode null_channel` expose
the controls. `--output` creates a new receipt and refuses to overwrite one.
The previous `demo_active_ait.py` intentionally enforces its historical code
fingerprints; run it from the earlier revision to reproduce that exact study.
Provided-text replay is not live network or human interaction. Complete session
and source-cursor checkpointing across processes remains unsupported.

## Frozen comparison

Run the new study with the earlier completed local artifacts and pinned PUD
source cache; it performs no downloads or new offline fitting:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/benchmark_observation_world.py \
  --output .cache/observation-world-pilot-v2 --prepare-only
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/benchmark_observation_world.py \
  --output .cache/observation-world-pilot-v2
```

The plan freezes current and historical code identities separately, all source
artifact hashes, partitions, mode order, both costs, and bootstrap settings
before evaluation. Modes are fixed channel, resetting mixture, persistent joint
inference, and a zero-informative-prior null. Both predeclared character costs
**0.0005 and 0.002** are evaluated without selecting a test winner: four modes
×two costs ×279 sentences = **2,232 episodes**.

Closed-loop outcomes differ across policies, so their average prediction scores
cannot establish better forecasting on equal targets. A separate fixed 4/16-
character prefix probe chooses one boundary using initial observed uncertainty
and supplies exactly the same actual outcomes to every filter. Forecasts precede
each new prefix inspection. Gold labels score the probe afterward and never
enter any filter. The shared probe tests a specified acquisition schedule, not
the controller's chosen schedule or general task quality.

Paired document-group bootstrap intervals use 2,000 draws and resample the same
groups across parallel languages. Local sequential episode timings include
planning and observation processing after a common checkpoint restoration;
they exclude restoration, scoring, serialization, and the separate probe.
The full comparison is exploratory: it reuses previously inspected translated
PUD and a checkpoint whose learning rate was chosen on the original development
labels. Intervals describe sampling variation on this corpus, not independent-
corpus generalization, semantic information, or production readiness.

## Completed comparison: changed decisions, no accuracy gain from regime memory

All **2,232 episodes** completed under corrected contract
`337af206d6456bf411f682295662a421fde85dfeda7524652a287ea3a3d9d79e`.
The new mechanism changes actual decisions and predictions. It is not promoted
to the default: persistent hypothesis inference did not improve task accuracy
over resetting the mixture, and the mixture is more conservative than the
original fitted channel.

Sentence-macro boundary F1 percentages and token-pooled mean waiting characters:

| Cost / mode | EN F1 | JA F1 | ZH F1 | EN wait | JA wait | ZH wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0005 / fixed channel | 99.8151 | 95.5277 | 89.2998 | 3.415 | 23.850 | 18.030 |
| 0.0005 / resetting mixture | 99.7938 | 95.0820 | 88.6829 | 2.601 | 10.045 | 9.357 |
| 0.0005 / persistent joint | 99.7938 | 95.0690 | 88.6419 | 2.600 | 10.214 | 8.866 |
| 0.0005 / null | 99.8115 | 93.8504 | 86.8375 | 2.476 | 2.442 | 2.453 |
| 0.002 / fixed channel | 99.7918 | 95.4072 | 89.1269 | 2.652 | 14.068 | 12.878 |
| 0.002 / resetting mixture | 99.7938 | 94.7016 | 88.5374 | 2.504 | 3.925 | 4.619 |
| 0.002 / persistent joint | 99.7938 | 94.6421 | 88.5374 | 2.503 | 3.247 | 4.605 |
| 0.002 / null | 99.8115 | 93.8504 | 86.8375 | 2.476 | 2.442 | 2.453 |

At cost 0.0005, persistent inference changes predictions relative to resetting
in **one Japanese and four Chinese sentences**; at cost 0.002, in **five Japanese
sentences**. English predictions are identical between those two modes. The
equal-language F1 deltas in percentage points are **−0.0180** (95% document
bootstrap interval **[−0.0601, +0.0147]**) and **−0.0198**
(**[−0.0515, +0.0088]**). Neither supports an additional accuracy benefit from
retaining hypothesis memory.

Relative to the fixed channel, persistent inference reduces Japanese/Chinese
waiting but also lowers their F1 at both costs. Equal-language F1 drops
**0.3793 percentage points** at cost 0.0005 (interval **[−0.5798, −0.1744]**)
and **0.4508** at cost 0.002 (**[−0.7094, −0.1597]**).
Lower waiting is not a quality-preserving acceleration. These comparisons mix
changes in observation filtering and action schedules; they do not isolate
scheduling alone. Nonwhitespace F1 is separately retained in the receipts.

Local median episode wall time also increases; fewer waiting characters do not
mean lower execution cost:

| Cost / mode | EN ms | JA ms | ZH ms |
| --- | ---: | ---: | ---: |
| 0.0005 / fixed channel | 107.93 | 35.75 | 10.40 |
| 0.0005 / persistent joint | 126.64 | 55.12 | 26.47 |
| 0.002 / fixed channel | 105.31 | 42.56 | 14.96 |
| 0.002 / persistent joint | 136.21 | 66.23 | 37.49 |

These are one sequential local timing per episode under the frozen condition
order, not a platform benchmark or a general latency estimate.

### Forecasting on identical observations

The shared prefix probe supplies **1,399** matched outcomes per mode:
740 English, 395 Japanese, and 264 Chinese. Lower mean outcome log loss is better.

| Language | Fixed channel | Resetting mixture | Persistent joint | Null |
| --- | ---: | ---: | ---: | ---: |
| EN | 0.002441 | 0.057013 | 0.045815 | 0.116363 |
| JA | 0.460832 | 0.455878 | 0.453977 | 0.464053 |
| ZH | 0.749387 | 0.731668 | 0.733616 | 0.733159 |

Persistent inference improves English outcome prediction relative to resetting:
delta **−0.011198** (interval **[−0.011835, −0.010468]**). English still performs
much better under the original fixed channel. The persistent-versus-reset
Japanese and Chinese intervals include zero; the Chinese point estimate is
slightly worse. Relative to the fixed channel, persistent Chinese log loss
improves **−0.015771** (**[−0.032857, −0.001294]**), while English worsens and
Japanese is inconclusive. Predicting raw CRF states better does not establish
better boundary labels or semantic information. Both outcome Brier and posterior
label Brier scores are also published.

### Evidence retention and validation

The first run executed all episodes, but its six non-null condition receipts
failed result-hash verification after JSON reload: integer waiting-histogram
keys sorted numerically before writing and lexically as strings afterward.
All eight raw file hashes matched that run's summary. Its local files and
[failure manifest](../reports/observation-world-2026-10-29/initial_run_integrity_failure.json)
are retained; it is not accepted as completed replayable evidence.

The corrected runner normalizes JSON trees **before** hashing and saving. A
regression test uses histogram keys 2 and 10, verifies every stored result hash,
and resumes with recomputation forbidden. The corrected plan discloses the
failure and repeats every unchanged mode, cost, probe, and sentence. No parameter
was selected from first-run outcomes. Resumed probe receipts additionally require
all four complete mode records, matching row identities and outcome counts.

Every corrected condition's raw file hash and reloaded result hash verifies.
All **2,232 prediction, metric, and trace records** match the retained first run
exactly after excluding elapsed timing; the shared-probe records and all paired
intervals also match. A full corrected-run resume completes without recomputing
episodes or changing stored evidence. The default authored replay reproduces
its saved JSON exactly.

Local validation: **484 tests passed, one skipped, nine subtests passed**;
distributions built; a fresh wheel installation outside the source tree passed
joint Bayesian continuation and three-language source-preserving replay with
unchanged segmentation weights. Dedicated tests also cover analytical multi-step
joint posteriors, label/regime martingales, the null, the original-channel limit,
planning purity, acquisition-budget feasibility, receipt retries, and partial EOF.
CI also exposed an existing test's process-wide `hash` replacement leaking into
pytest 9.2's report hashing. The test now restores the builtin immediately after
the guarded call; its check that seeded features avoid Python's salted hash
remains. This changes test isolation only, not experiment code or predictions.

Evidence: [frozen plan](../reports/observation-world-2026-10-29/plan.json),
[all aggregate results and intervals](../reports/observation-world-2026-10-29/summary.json),
[paired sentence scores](../reports/observation-world-2026-10-29/paired_scores.jsonl),
and [authored replay](../reports/observation-world-2026-10-29/demo_trace.json).
The authored 211-character replay preserves all input, records 139 events and
86 correlated boundary assimilations, and shows an informative-hypothesis
probability falling from 0.50 to 0.263 after an unexpected outcome. It has no
gold task labels and establishes no accuracy claim.
