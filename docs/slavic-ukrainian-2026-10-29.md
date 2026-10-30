# Ukrainian and Slavic segmentation adaptation pilot

Ukrainian boundary segmentation improves sharply after continuing a pinned
English/Japanese/Chinese checkpoint on Ukrainian, Polish and Czech examples.
Continuing on Slavic data alone also causes substantial Japanese and Chinese
forgetting. Rehearsing the original languages prevents most of that loss,
but Chinese still regresses. Preserve both outcomes; these checkpoints remain
experimental and local.

## Completed results

All four declared training jobs and all 3,315 evaluations completed. Each model
was evaluated on the same 663 cases: 192 Slavic development, 192 Slavic test,
and 279 old-language retention cases. The final fourth epoch is reported for
every job; development/test results never selected a checkpoint or learning rate.

The primary table excludes boundaries touching whitespace. Micro F1 pools
boundary counts within a language, then averages the two shuffle-seed scores.
This avoids treating easy whitespace boundaries as evidence of language skill.
Each Slavic language has 64 test sentences; each retained language has 93.

| Test language | Before adaptation | Slavic only | Slavic + old-language rehearsal |
| --- | ---: | ---: | ---: |
| Ukrainian | 7.3458% | 98.2384% | 98.2384% |
| Polish | 95.8824% | 97.8227% | 96.7742% |
| Czech | 97.2477% | 98.7952% | 97.7167% |
| English retention | 97.5000% | 96.3656% | 96.6942% |
| Japanese retention | 95.8678% | 73.8301% | 96.0843% |
| Chinese retention | 89.4217% | 78.0052% | 88.7817% |

Sentence-macro non-whitespace F1 for Ukrainian is 10.9308% → 97.8348% /
97.7902%. The pre-adaptation model severely oversegments Cyrillic text: 3,229
false positive non-whitespace boundaries across these 64 Ukrainian cases,
versus two in each continued model. This is a surface segmentation repair,
not evidence of general Ukrainian fluency or semantic understanding. No new
Cyrillic character category or checkpoint format was introduced.

For comparison with earlier studies, full-boundary sentence-macro F1 follows.
The intervals concern the **change from before adaptation**, in percentage
points, using 2,000 paired document/date bootstrap resamples. They do not cover
non-whitespace F1 or a direct statistical comparison between training arms.

| Test language | Before | Slavic only | Change, conditional 95% interval | With rehearsal | Change, conditional 95% interval |
| --- | ---: | ---: | --- | ---: | --- |
| Ukrainian | 47.3102% | 99.7807% | +52.4705 [51.6564, 52.9923] | 99.7514% | +52.4411 [51.5813, 52.9923] |
| Polish | 99.6767% | 99.8673% | +0.1906 [0.0488, 0.3710] | 99.7878% | +0.1111 [-0.0068, 0.2614] |
| Czech | 99.7436% | 99.8047% | +0.0611 [-0.1633, 0.3046] | 99.7438% | +0.0002 [-0.0942, 0.0693] |
| English retention | 99.8372% | 99.7427% | -0.0944 [-0.1836, -0.0179] | 99.7788% | -0.0584 [-0.1221, -0.0097] |
| Japanese retention | 95.5503% | 72.4459% | -23.1044 [-25.7844, -20.9071] | 95.8204% | +0.2700 [-0.1408, 0.6616] |
| Chinese retention | 89.5943% | 78.2028% | -11.3915 [-13.2797, -9.3888] | 88.7943% | -0.8000 [-1.5379, -0.0835] |

Rehearsal is the more useful continuation candidate here because it retains
most old-language performance with similar Ukrainian segmentation. It is not
a deployment selection: English and Chinese still lose accuracy, and Polish
and Czech gains are smaller. The next comparison should freeze a new training
and retention intervention before collecting fresh evaluation documents; these
inspected test cases should not become a tuning target.

### Raw boundary probability scores

Scores below average the two seed-level per-position aggregates; lower is
better. Each cell is Brier / NLL / ECE. ECE uses ten fixed bins. These are raw
CRF marginals without post-hoc calibration; low scores do not establish
calibrated correctness outside this fixed sample.

| Test language | Before | Slavic only | With rehearsal |
| --- | --- | --- | --- |
| Ukrainian | .671484 / 3.668805 / .683715 | .000797 / .005010 / .002257 | .000826 / .005922 / .003728 |
| Polish | .001820 / .007409 / .001477 | .000645 / .002079 / .001219 | .000929 / .002973 / .001321 |
| Czech | .001292 / .005126 / .001892 | .000589 / .002338 / .000929 | .000765 / .002670 / .001110 |
| English | .000631 / .002503 / .001047 | .001063 / .003463 / .001054 | .000860 / .002780 / .000661 |
| Japanese | .037574 / .137513 / .023116 | .230314 / 1.120463 / .217288 | .037784 / .138603 / .025457 |
| Chinese | .089510 / .287566 / .011247 | .287173 / 1.126360 / .306110 | .092650 / .295628 / .021345 |

## Frozen comparison and actual continuation

The [configuration](../configs/slavic-learning-pilot.json) declares two arms,
each with shuffle seeds 5042 and 101. Both start from the **same pretrained
parameters**, not independently pretrained initializations:

- **Slavic only:** 128 training sentences each for Ukrainian, Polish and Czech;
  384 per epoch, 1,536 processed over four epochs.
- **With rehearsal:** the same 384 sentences plus 96 old training sentences each
  for English, Japanese and Chinese; 672 per epoch, 2,688 processed in total.
  This adds 75% more example exposures and additional computation.

The starting checkpoint SHA256 is
`9c73420a3c7fe4e9d73cc463b473b0b45c01a7111f058bbf100943136a81e783`.
It is the earlier full-budget seed-5042 supervised-encoder checkpoint, with
encoder learning rate .01. Architecture is derived from the checkpoint,
including float64, context hidden dimension 16 and 4,096 lexical buckets.
Head/CRF/context/lexical/encoder learning rates are .01/.005/.01/.02/.001,
batch size four, regularization .0001, and global gradient clip ten.
Phase is frozen, validation split is zero, and all four epochs are fixed.

`StudentTrainingConfig(warm_start=True)` validates Python ownership, dimensions,
dtype, lexical/context configuration and existing parameter integrity before
continuation. Normal training keeps its previous fresh-head default. Each real
worker verifies exact zero-epoch parameter preservation, reseeds the shuffle
generator, then verifies changed head and encoder hashes with unchanged phase.
The saved checkpoint is reloaded and checked against those hashes. Training
random-generator state is not checkpointed; arbitrary shuffled or validation
resume is not claimed identical to uninterrupted training.

Evaluation uses direct full-text Viterbi segmentation and raw marginals from one
shared contextual observation per case. It does not use the previously fitted
active-acquisition channel after changing the boundary weights. This pilot
therefore adds no evidence for acquisition policy or world-model benefits.

## Sources and partition integrity

The [source manifest](../configs/slavic-v2.18-sources.json) pins revisions,
byte counts, SHA256, source origins, README and LICENSE files. Eleven files
totalling 14,680,982 bytes were acquired once. All 7,142 Ukrainian, 1,000 Polish
and 1,000 Czech sentences aligned exactly; no alignment exclusions occurred.
Surface multiword tokens replace component tokens, empty nodes are ignored,
and original Unicode whitespace, apostrophes and combining accents are kept.

Ukrainian ParlaMint contains **six actual source dates**, pooled across the
provider's train/dev/test files and assigned by a fixed date hash order:

| Role | Source dates |
| --- | --- |
| Train | Rada 2003-10-14, 2022-01-25, 2022-02-24 |
| Development | Rada 2004-11-27 |
| Test | Rada 2014-03-04 and NSDC 2014-02-28 |

All sessions of a date stay together. After assigning dates, 769 cross-role
normalized duplicate rows and 256 within-role duplicate rows are excluded.
There are 3,327/1,502/1,288 native train/dev/test rows before subsequent length
selection. This is a custom split, not an official UD test score.

Polish/Czech PUD are translated auxiliaries. Actual parallel sentence IDs map
to the original EN/JA/ZH document groups and roles, preventing translated test
content from entering training. The guard also checks all 3,000 old source
rows before filtering and caps. It excludes ambiguous exact-surface labels and
normalized cross-role duplicates before selecting examples up to 192 characters.
Six auxiliary rows lack an eligible inherited parallel role and are excluded.
The prior 279-case retention set remains unchanged.

Selection uses fixed hashes, not observed quality. Each Slavic language then
has 128/64/64 train/dev/test sentences. Ukrainian test has only **two dates**;
Polish/Czech test each span 34 inherited document groups, and retention spans
40 groups per language. Intervals remain conditional on these documents,
two shuffle orders and one initial checkpoint. Existing PUD content and model
selection were already inspected in earlier work. The PUD protocol uses one
held-out fold of the inherited ten-fold assignment, not a complete ten-fold
cross-validation sweep or independent-corpus confirmation.

## Time and memory

Fresh sequential workers used Python 3.12.6, NumPy 2.5.3 on macOS arm64 with
OMP/OpenBLAS/Accelerate thread variables set to one. Training wall time excludes
loading and scoring. RSS is the whole worker's high-water mark, including its
loading/training/evaluation lifetime. JSON bytes describe serialized checkpoints,
not compressed weights or resident model memory.

| Arm / shuffle seed | Training seconds | Whole-worker peak MiB | Checkpoint bytes |
| --- | ---: | ---: | ---: |
| Slavic only / 5042 | 11.285 | 109.88 | 496,118 |
| Slavic only / 101 | 13.016 | 116.17 | 496,208 |
| With rehearsal / 5042 | 15.950 | 128.45 | 496,131 |
| With rehearsal / 101 | 15.785 | 127.06 | 496,278 |

Per-case inference medians are recorded in the summary, but these sequential
training workers are not a controlled speed or memory optimization study.
The prior [exact inference comparison](ait-efficiency-2026-10-29.md) remains
separate, with its own frozen source revision and measured scope.

## Reproduction and receipts

The [prepared plan](../reports/slavic-ukrainian-2026-10-29/plan.json),
[complete summary](../reports/slavic-ukrainian-2026-10-29/summary.json), and
[663 paired score rows](../reports/slavic-ukrainian-2026-10-29/paired_scores.jsonl)
bind every source revision, case identity, label hash, source fingerprint,
checkpoint/receipt hash, seed and result. The plan was written before training.
Contract SHA256:
`cb5172467793757601b266bc45e04332002e1da8ee903e92c9eed5d759d75cb8`.
Public rows contain identities, hashes, counts and metrics. Raw corpora,
predicted tokens, task files and learned checkpoints remain in the local cache.

From the repository, with the pinned starting checkpoint restored to the
configuration's local path:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python scripts/benchmark_slavic_learning.py \
  --output .cache/slavic-learning-pilot-v1 --prepare-only --download
python scripts/benchmark_slavic_learning.py \
  --output .cache/slavic-learning-pilot-v1 \
  --publish reports/slavic-ukrainian-2026-10-29
```

Use a new publication directory when reproducing: existing published evidence
is never overwritten. A completed local resume validates all frozen inputs,
source, checkpoint, continuation, score and resource receipts without retraining.
An incomplete attempt is retained and blocks implicit retries. Exact contract
identity also includes Python/NumPy/platform; a different runtime creates a
different study identity rather than replacing this evidence. The starting
checkpoint itself is local; public receipts alone are insufficient to recreate
its weights. Its training protocol is documented in
[the earlier policy/learning study](policy-learning-2026-10-08.md).

Local validation completed with 612 tests passed, one skipped and nine subtests
passed. A freshly built wheel was installed into an isolated environment and
tested outside the source tree, including Slavic Unicode reconstruction,
checkpoint roundtrip, exact zero-epoch continuation, and one actual warm update
to head/encoder with phase preserved. An actual completed-study resume passed
without retraining. All 663 public paired rows, five conditions, 3,315 total
evaluations, source fingerprints and public-file privacy checks passed.

## Attribution and scope

The new UD sources are **CC BY-SA 4.0**. Credit their contributors and use the
complete pinned README/LICENSE files, retained alongside the local corpora:

- [UD Ukrainian ParlaMint](https://github.com/UniversalDependencies/UD_Ukrainian-ParlaMint/tree/dfa85dba5c2254f5e1a71b2fa8a7316b76bf18b9):
  Maria Shvedova, Arsenii Lukashevskyi, ParlaMint-UA/UD contributors, with the
  upstream acknowledgments to VESUM and UD Ukrainian IU.
- [UD Polish PUD](https://github.com/UniversalDependencies/UD_Polish-PUD/tree/888f7ff15422a70be1adbf5ea7c0c66f5d8b444d):
  Alina Wróblewska and the Polish PUD/PDB contributors at the Institute of
  Computer Science, Polish Academy of Sciences.
- [UD Czech PUD](https://github.com/UniversalDependencies/UD_Czech-PUD/tree/dfab1926110b82c2e01278c292d9c01b0abbc0ff):
  Charles University, the PUD translation/annotation teams and UD contributors.
- [CC BY-SA 4.0 terms](https://creativecommons.org/licenses/by-sa/4.0/).

The original EN/JA/ZH PUD sources retain their separate **CC BY-SA 3.0** terms
and underlying-source ownership caveat, as documented in
[their attribution](trilingual-context-2026-10-08.md#attribution-and-source-terms).
The nine new authored demo fixtures (five Ukrainian, two Polish, two Czech)
remain under the existing **CC BY 4.0** fixture license. They test Unicode and
API behavior and are not the study's quality evidence. Repository code remains
Apache-2.0; it does not replace external data terms.

This pilot measures labeled surface segmentation within a bounded sample. It
does not establish general Slavic proficiency, Ukrainian fluency, broad domain
generalization, independent initialization robustness or production readiness.
