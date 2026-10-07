# Repairing policy scoring and learning, then comparing budgets

This follows [the first trilingual pilot](trilingual-context-2026-10-08.md).
Its negative transformer result is preserved. This study repairs two mechanisms
before comparing training budgets and initialization seeds. It uses an already
inspected PUD partition and is exploratory, not independent-corpus confirmation.

## Completed results and decision

All 45 training runs and 27 selected-model evaluations completed. Each cell
below is the mean sentence-level internal-boundary F1 across three seeds on
the same 93 test sentences for that language. These are 93 unique sentences,
not 279 independent observations. The transformer-learning column uses only
the learning rate selected on development data, with no test-based reranking.

| Training sentences per language | Language | Historical fixed encoder | Corrected fixed encoder | Corrected trained encoder |
| ---: | --- | ---: | ---: | ---: |
| 82 | English | 99.5668% | 99.6541% | 99.6516% |
| 82 | Japanese | 84.2530% | 90.3258% | 90.2864% |
| 82 | Chinese | 80.5369% | 83.7981% | 83.2792% |
| 308 | English | 99.7373% | 99.8128% | 99.8084% |
| 308 | Japanese | 92.4288% | 94.4823% | 94.4412% |
| 308 | Chinese | 85.5368% | 87.4385% | 87.4232% |
| 804 | English | 99.8105% | 99.8295% | 99.8451% |
| 804 | Japanese | 94.6534% | 95.3941% | 95.8233% |
| 804 | Chinese | 88.2451% | 89.1120% | 89.9498% |

The optimizer repair improves the mean in every language/budget cell. The
largest changes are in the smallest budget: +6.073 percentage points for
Japanese and +3.261 for Chinese, with the encoder still frozen. Increasing
training data then improves the corrected fixed encoder further. This separates
the repair effect from the effect of adding documents. It does not separate
the averaging, clipping, and regularization parts of the repair from each other.

**Supervised transformer learning helps at the largest budget in this run,
but does not improve the mean at the two smaller budgets.** At 804 training
sentences per language, its increment over the corrected fixed encoder is:

| Language | Trained minus fixed (percentage points) | Conditional document 95% interval (points) |
| --- | ---: | ---: |
| English | +0.0157 | [-0.0042, +0.0472] |
| Japanese | +0.4291 | [+0.1538, +0.6806] |
| Chinese | +0.8378 | [+0.1416, +1.5892] |

The Japanese and Chinese intervals are positive conditional on these fitted
models and this split. English includes zero. The full-budget trained-model
seed ranges are 99.8346–99.8637% (English), 95.5503–96.0936% (Japanese), and
89.5943–90.4866% (Chinese). At that budget, English F1 excluding boundaries
touching whitespace is 97.8608% → 98.0827% → 98.2246%, so its high aggregate
score must still be read with the easy whitespace boundaries in mind.

The selected encoder rates by seed 5042 / 101 / 2026 are 0.05 / 0.05 / 0.001
at cap 30, 0.01 / 0.001 / 0.001 at cap 120, and 0.01 / 0.05 / 0.05 at full
budget. Development prefers the trained encoder in eight of nine comparisons;
cap 120 / seed 2026 prefers the fixed encoder. The trained arm is still reported
there as an ablation, not described as a selected deployment. A development
advantage at the smaller budgets also failed to become a mean test gain.

### Policy scoring outcomes

Legacy EFE chose `SeekEvidence` in all 7,533 model/text evaluations. Shared
posterior risk chooses among candidates using the input, and retains zero bias
on exact ties. At full budget with the trained encoder, zero bias is retained
in 277/279 English, 225/279 Japanese, and 194/279 Chinese model/text evaluations.
These counts include the three repeated seeds.

| Full-budget arm | Language | Direct F1 | Legacy EFE F1 | Shared-posterior risk F1 |
| --- | --- | ---: | ---: | ---: |
| Corrected fixed | English | 99.8295% | 99.8257% | 99.8392% |
| Corrected fixed | Japanese | 95.3941% | 95.4499% | 95.4972% |
| Corrected fixed | Chinese | 89.1120% | 89.1937% | 89.2272% |
| Corrected trained | English | 99.8451% | 99.8451% | 99.8423% |
| Corrected trained | Japanese | 95.8233% | 95.8404% | 95.8229% |
| Corrected trained | Chinese | 89.9498% | 89.9608% | 89.9090% |

Policy scoring is no longer dominated by fictitious information gain, but
**its quality effect remains small and mixed**. It improves the full-budget
fixed encoder in this split and is slightly worse or effectively flat with
the trained encoder. Marginal-temperature calibration changes only one of the
7,533 posterior-risk predictions; it provides no material incremental effect
on this selector here. These policy comparisons are descriptive point estimates.
The prior streaming-calibration result is a different intervention and is not
replaced by this finding.

Keep both transformer learning and AIF opt-in. The next quality gate is an
independent corpus, especially native Japanese/Chinese and unseen English
domains, using a separately frozen plan. Do not tune these inspected PUD test
results into a new claim of independent generalization.

Evidence: [prepared plan](../reports/policy-learning-2026-10-08/plan.json),
[complete summary and seed results](../reports/policy-learning-2026-10-08/summary.json),
[per-document score sums](../reports/policy-learning-2026-10-08/document_scores.jsonl).
The public summary binds all 27 detailed local evaluation receipts by hash.
All 45 checkpoint hashes, frozen source hashes, disjoint nested partitions,
and 3,240 public document-score rows were checked. Source text and model
weights are not published.

## Mechanism repairs

### Mean objective before clipping

The NumPy optimizer previously clipped the **sum** of example gradients and
then divided by batch size. Once clipping was active, duplicating an otherwise
identical batch approximately halved its update. Head/lexical regularization was
also divided by batch size, while the supervised encoder's regularization was
not. The CRF transition decay was absent from the reported L2 objective.

The corrected update forms the mean NLL gradient plus one L2 gradient, computes
one global clipping factor for that objective, then applies each parameter
group's learning rate. The reported regularized loss and update use the same
parameter set. This changes future NumPy training; loading an existing checkpoint
does not change its direct predictions. Native optimizer kernels are outside this
repair. Heuristic phase/gate updates remain separate; both are frozen in the
comparison's fixed-encoder arms.

On a small deterministic regression example with active clipping, the old
single-example output-head step had L2 size 0.00004851, while a duplicated batch
produced 0.00002444. The corrected steps are exactly equal in the regression
test, including the context encoder and lexical parameters. Finite differences
also verify the mean regularized objective across all parameter families.
Epoch history now reports objective-gradient norms, clipping factors, clipped
update counts, and encoder-gradient norms.

### One shared belief for policy choices

The old segmentation EFE proxy assigned observation noise independently by
policy, although all policies only re-decoded the same logits. With its default
three-dimensional Gaussian prior, its entropy reduction is
`1.5 * log(1 + prior_sigma**2 / obs_sigma**2)`: a constant for each policy before
the text is considered. It awards `SeekEvidence` 4.504 units versus 3.378 for
the unbiased `ProbeMotivation`, without collecting an observation. This
unearned advantage explains the collapse observed in the earlier PUD run.

The new default when `use_aif=True` is `selection_mode="posterior_risk"`. It
evaluates every candidate's decoded boundaries under the **same unchanged CRF
marginals**, with expected false-positive and false-negative costs. No new
observation is claimed, and the epistemic term is zero. A zero-bias candidate is
required, and exact risk ties prefer that candidate. Tests enumerate the expected
boundary errors, demonstrate different input-dependent choices, and verify that
the old observation-noise settings cannot affect the new score.

This guarantees only that the selected candidate's estimated boundary risk is
no greater than the unchanged candidate's under that belief. It does **not**
guarantee real error or F1 improvement: raw marginals may be miscalibrated, and
boundary error and F1 are different objectives. Policy diversity is not a goal.
This remains a decoder-selection scaffold, not an evidence-acquisition agent.

Use `SegmentationAIFConfig(selection_mode="legacy_efe")` to reproduce the old
scoring rule. Its Gaussian noise, preference targets, and EFE weights apply only
in that explicit mode. `use_aif` itself remains off by default. Optional
`marginal_temperature` transforms the shared belief; this study fits it using
development data only, and does not persist it as a production calibration.

## Frozen comparison

- The same pinned PUD r2.18 sources and exact alignment rules as the first pilot.
  Raw text, attribution, licenses, and trained checkpoints remain local. Dataset
  terms and the upstream source-text ownership caveat remain as documented in
  [the earlier attribution](trilingual-context-2026-10-08.md#attribution-and-source-terms).
- Ten document/translation/normalized-duplicate folds retain seed 1739. This
  experiment fixes fold 0 as test and fold 1 as development. Every condition sees
  the same **93 test sentences per language** and 100 development sentences per
  language. It does not repeat the earlier ten-fold test sweep.
- Nested caps of 30 documents, 120 documents, and all remaining 317 documents
  give **82, 308, and 804 training sentences per language**. More data also means
  more optimizer updates because epochs, rather than compute, are held fixed.
- Initialization seeds 5042, 101, and 2026; eight epochs, identical per-seed
  initialization/example order, fixed phase, same lexical/context head, and a
  16-dimensional one-layer/two-head NumPy transformer.
- Historical fixed-encoder optimizer, corrected fixed-encoder optimizer, and
  corrected supervised encoders at rates 0.001, 0.01, and 0.05: **45 training runs**.
  The historical package is copied from immutable commit
  `76ac1ebc8a3af0da42a7537baab0e021b612a0c9` and imported in separate processes.
- Choose the supervised learning rate separately for each budget/seed using
  development macro F1, equally weighted across languages; ties prefer the lower
  rate. Record whether this beats the corrected fixed encoder on development.
  Complete and save all choices before running the common test set. The chosen
  learning rate is shared across languages for each multilingual model.
- Evaluate 27 fitted models using direct decoding, explicit legacy EFE, shared
  posterior risk, and development-calibrated posterior risk. Candidate biases
  and error costs are fixed in advance. Three seeds are repeated model runs on
  the same sentences, not three times as many independent test examples.

Configuration: [policy-learning-comparison.json](../configs/policy-learning-comparison.json).
Frozen contract: `39859d6998a3b7d392674fef9af3b92bc0e553051343c2565500ea04a3110057`.
Implementation, worker, corpus, reference-package, partition, and runtime hashes
are bound before fitting. A modified contract requires a new output directory.
Resume verifies completed checkpoint hashes and the saved development decision;
published evidence cannot be overwritten.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/benchmark_policy_learning.py --prepare-only \
  --output .cache/my-policy-learning-study
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/benchmark_policy_learning.py --workers 3 \
  --output .cache/my-policy-learning-study
```

The pinned source cache must already exist; otherwise explicitly add `--download`
to preparation. This comparison also requires the reference Git commit locally.
Use a checkout containing that commit, rather than a wheel alone, to reproduce it.

## Interpretation limits

PUD is translated news/Wikipedia text, and these test-fold aggregate results
were already included in the first pilot. Development-only selection within
this frozen run avoids further test-driven tuning; it cannot erase that prior
exposure. Any adoption claim needs an independent corpus. No native informal
language, speech, large pretrained model, or native training kernel is evaluated.

Seed ranges describe three particular initializations. Document bootstrap
intervals condition on these fitted models and one split; they do not estimate
population training-seed uncertainty. Larger data budgets also spend more
training compute. No general speed or compute-efficiency claim is made.

## Validation

- 335 tests passed, one skipped, and nine subtests passed locally. Regression
  checks cover clipped/regularized batch invariance, finite-difference objective
  gradients, posterior-risk enumeration and ties, checkpoint/source corruption,
  development-only selection, and an offline three-language study with resume.
- The historical reuse benchmark explicitly selects legacy EFE and NumPy on both
  paths. Its regression check verifies identical outputs/scores and five encoder
  passes versus one; it does not measure the posterior-risk selector's speed.
- Source distribution and wheel built. A fresh installed-wheel environment
  outside the checkout passed batch-invariant updates, three-language checkpoint
  inference, text preservation, default posterior risk, and explicit legacy EFE.
- The repeated historical run at cap 30 / seed 5042 produced a checkpoint
  **byte-identical** to the first pilot's fold 0 fixed-encoder checkpoint:
  `ba6f2ea3375ac00ed74b01ebfa6cbee22459b851938c3bae30f41173a59d0a98`.
  This anchors the optimizer comparison to the actual earlier implementation.
