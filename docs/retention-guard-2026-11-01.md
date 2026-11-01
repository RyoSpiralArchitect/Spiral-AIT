# Chinese retention and fresh-corpus Ukrainian evaluation

Increasing Chinese rehearsal within the same training-example budget reduces
the observed Chinese loss while preserving Ukrainian segmentation. A fixed
weight-space interpolation improves Chinese retention further but substantially
damages Ukrainian. **Neither candidate passes the predeclared joint gate.**
The study keeps both results and does not change inference or training defaults.

This follows [the Slavic continuation pilot](slavic-ukrainian-2026-10-29.md),
merged as PR #90. Its inspected test results motivated this prospective protocol;
they are not reused as evidence of fresh-corpus generalization. New evaluation
data was acquired and partitioned before scoring, with no candidate tuning on
its outcomes.

## Completed comparison

Two new training jobs and all **8,225 evaluations** completed: seven model
conditions on the same 1,175 cases each. These comprise the previous 663 cases
and 512 additional official-test cases, 128 per language from separate Ukrainian,
Chinese, Japanese and English treebanks. No new evaluation text enters training.

Each adapted column averages two shuffle orders from **one common pretrained
initialization**. The pre-adaptation column evaluates that single original model.
Historical rehearsal uses the exact two saved checkpoints from the prior study,
rescored here on every case; it is not retrained or selected again.

### Fresh official-test samples

The first table reports **full-boundary sentence-macro F1**. These are custom
128-case samples, not full official treebank scores.

| Language / new source | Pre-adaptation | Historical rehearsal | Chinese-priority rehearsal | Priority + fixed weight blend |
| --- | ---: | ---: | ---: | ---: |
| Ukrainian / IU | 48.9314% | 99.0922% | 99.0866% | 97.2152% |
| Chinese / GSDSimp | 82.9017% | 81.8601% | 82.4543% | 82.8285% |
| Japanese / GSD | 92.8939% | 92.5756% | 92.8618% | 92.2104% |
| English / EWT | 94.6741% | 94.8784% | 94.8416% | 94.9514% |

Chinese-priority rehearsal reduces the fresh Chinese point-estimate loss from
**1.0415 to 0.4474 percentage points** relative to pre-adaptation, improving
0.5942 points over historical rehearsal. Chinese original document IDs are
unavailable, so this is a descriptive comparison on the fixed set; no document
interval or population significance claim is supplied.

To expose easy whitespace boundaries, the next table uses **non-whitespace
boundary micro F1**: pooled TP/FP/FN within each language and model, then mean
across the two shuffle seeds. It excludes every boundary touching whitespace.

| Language | Pre-adaptation | Historical rehearsal | Chinese-priority rehearsal | Priority + fixed weight blend |
| --- | ---: | ---: | ---: | ---: |
| Ukrainian | 9.0485% | 91.9359% | 91.8564% | 80.0587% |
| Chinese | 82.5638% | 81.9208% | 82.4475% | 82.5707% |
| Japanese | 93.2593% | 92.9021% | 93.2512% | 92.4975% |
| English | 73.6626% | 74.5263% | 73.9266% | 75.3218% |

Fresh Ukrainian remains much stronger than pre-adaptation, but its non-whitespace
F1 is below the prior parliamentary test result. Increasing Chinese replay changes
it by -0.0795 points against historical rehearsal here. The fixed blend loses
11.8772 points. High full-boundary F1 must not hide this difference.

### Previously inspected test sets

These remain diagnostic retention/adaptation controls. Full-boundary macro F1:

| Language | Pre-adaptation | Historical rehearsal | Chinese-priority rehearsal | Priority + fixed weight blend |
| --- | ---: | ---: | ---: | ---: |
| Ukrainian | 47.3102% | 99.7514% | 99.8599% | 98.0616% |
| Polish | 99.6767% | 99.7878% | 99.8080% | 99.6929% |
| Czech | 99.7436% | 99.7438% | 99.7292% | 99.7579% |
| English | 99.8372% | 99.7788% | 99.7818% | 99.7793% |
| Japanese | 95.5503% | 95.8204% | 95.6231% | 95.1698% |
| Chinese | 89.5943% | 88.7943% | 89.0846% | 89.3220% |

On the old Ukrainian test, non-whitespace micro F1 is 98.2384% for historical
rehearsal, **98.8327%** for priority rehearsal, and **81.6855%** for the fixed
blend. On old Chinese, priority rehearsal reduces the full-macro point loss from
0.8000 to 0.5097 points. The paired conditional document-bootstrap change
against historical rehearsal is +0.2903 points, interval [-0.2161, +0.7950];
it includes zero. These results do not establish a population-level benefit.

### Predeclared gates, including failures

Before any scoring, the [frozen configuration](../configs/retention-guard-pilot.json)
required all of the following for a joint pass:

- Old Ukrainian non-whitespace micro F1 at least 97%.
- Fresh Ukrainian non-whitespace micro F1 within 0.5 points of historical rehearsal.
- Fresh Chinese full-boundary macro F1 at least historical rehearsal.
- Both old and fresh English/Japanese full-boundary macro F1 within 0.5 points
  of pre-adaptation; Chinese within **0.25 points**.

| Gate family | Priority rehearsal | Priority + fixed blend |
| --- | --- | --- |
| Old/fresh Ukrainian preservation | Pass / pass | **Fail / fail** |
| Fresh Chinese vs historical rehearsal | Pass | Pass |
| Old/fresh English tolerance | Pass / pass | Pass / pass |
| Old/fresh Japanese tolerance | Pass / pass | Pass / **fail** |
| Old/fresh Chinese tolerance | **Fail / fail** | **Fail** / pass |
| Joint result | **Fail** | **Fail** |

The blend's old Chinese loss is 0.2723 points, just outside the declared 0.25;
the threshold is not relaxed. Priority rehearsal improves the observed tradeoff
but is not promoted to a default or a deployment selection. No second ratio,
mixing weight, training retry, or alternative checkpoint was tried after reading
these outcomes. The next intervention needs a new protocol and uninspected
evaluation material, rather than another fit to these 512 cases.

## Interventions and implementation

Both new fits use the **exact same 384 Slavic training cases** as the prior
study: 128 Ukrainian, 128 Polish, 128 Czech. Rehearsal still adds 288 old training
cases, with nested deterministic hash selections:

| Replay language | Historical rehearsal | Chinese priority |
| --- | ---: | ---: |
| English | 96 | 64 |
| Japanese | 96 | 64 |
| Chinese | 96 | 160 |

Both have 672 examples per epoch, four fixed epochs and **2,688 processed
examples** per model. The number of characters and tokens need not match;
equal example counts are not a guarantee of equal work. Learning rates, batch
size four, regularization, clipping, architecture and supervised encoder training
are unchanged from the prior experiment. Phase is frozen. Seeds 5042 and 101
control shuffle order, not separate pretraining. No development or test metric
selects an epoch, rate, allocation or weight.

The fixed interpolation is
`0.25 * pre_adaptation_parameters + 0.75 * priority_parameters`. The new opt-in
`integrated.checkpoint_blend.blend_checkpoints()` validates explicit Python
ownership, matching architecture/precision/lexical layout, finite tensors,
consistent encoder copies and unchanged phase before mixing all head and
encoder parameters. Inputs are copied. At weights zero and one it returns exact
independent endpoint copies after validation. Other model state, including AIF
dynamics, comes from the adapted checkpoint and is not averaged.

The result is one ordinary checkpoint. Inference retains no extra model and
uses no oracle language label, routing rule or prediction ensemble. Weight
averaging does not guarantee interpolated predictions or retained accuracy;
the Ukrainian failure here demonstrates that limit. All scores use direct
full-text Viterbi segmentation and one contextual observation per case. This
study supplies no new evidence for AIF acquisition or world-model benefits.

## Fresh sources, exclusions and attribution

The [manifest](../configs/retention-ood-v2.18-sources.json) pins twelve files:
four official test files with their README and LICENSE, **8,722,461 bytes**.
All 4,018 source sentences aligned exactly with their surface labels. No raw
source text, predictions or checkpoint weights are published.

| Language | Source revision | Terms | Aligned sentences | Selected / supplied document IDs |
| --- | --- | --- | ---: | --- |
| Ukrainian | [IU `0d5ed088`](https://github.com/UniversalDependencies/UD_Ukrainian-IU/tree/0d5ed088e959ccfbd933d768114d8503c32c9b37) | CC BY-NC-SA 4.0 | 898 | 128 / 55 |
| English | [EWT `b7711cce`](https://github.com/UniversalDependencies/UD_English-EWT/tree/b7711cce01cdd4f5fcc0a8199b8a50d951b16c0c) | CC BY-SA 4.0 | 2,077 | 128 / 87 |
| Japanese | [GSD `33e7310b`](https://github.com/UniversalDependencies/UD_Japanese-GSD/tree/33e7310b58308e85fd2b33a2fc3ef3e434f821c7) | CC BY-SA 4.0 | 543 | 128 / 128 provider units |
| Chinese | [GSDSimp `7b61ed47`](https://github.com/UniversalDependencies/UD_Chinese-GSDSimp/tree/7b61ed473f963e911788efdf1f478154bc1053e4) | CC BY-SA 4.0 | 500 | 128 / **unavailable** |

Credit Institute for Ukrainian and the IU contributors Natalia Kotsyba, Bohdan
Moskalevskyi, Mykhailo Romanenko and the upstream annotation team. IU spans
fiction, news, Wikipedia, legal and other genres; this study does not verify
every text's original-language or translation provenance. Its noncommercial
data is used **only for evaluation**, never candidate training.

Credit the Stanford/UD EWT team and preserve its separate underlying-text
ownership caveat; its annotation/database license does not replace the original
source owners' rights. Japanese GSD credits its UD Japanese contributors and
the Google UDT source; its supplied document IDs equal sentence IDs, without
proving independent original documents. Chinese GSDSimp credits Peng Qi and
Koichi Yasuoka; it derives from traditional Chinese GSD via simplified-character
conversion and manual corrections. Complete original README/LICENSE files stay
in the local pinned cache. These terms remain separate from Apache-2.0 code,
the earlier source licenses and the CC BY 4.0 authored fixtures.

The overlap guard checks **all 12,184 previous source/fixture rows**, including
unselected source rows, before length filters and caps. Across the fresh sources,
it excludes 28 old-source overlaps, 95 normalized duplicates, 31 rows with
ambiguous normalized supervision, and 208 rows exceeding length bounds.
Original text, whitespace, combining accents and apostrophes are never repaired.
Selection is a fixed source/sentence hash up to 192 characters, with 128 cases
per language. No gold metric influences selection. Normalized exact matching
does not rule out semantic near duplicates or translation relationships.

Chinese lacks original document IDs. Japanese's provided units are effectively
sentences. **Neither receives a document-bootstrap interval.** Other full-macro
changes use 2,000 paired resamples of supplied document/date units, conditional
on these cases, two shuffle orders and the common starting checkpoint. Those
intervals do not cover non-whitespace micro F1, initialization uncertainty,
document independence or general population performance. Raw Brier/NLL/ECE and
all per-case counts remain available in the summary and paired records; no
post-hoc calibration is fitted.

## Cost, validation and reproduction

Training uses fresh sequential workers with Python 3.12.6 / NumPy 2.5.3 on
macOS arm64; OMP/OpenBLAS/Accelerate thread variables are one. Wall time covers
the training call. RSS covers the whole training worker, including loading and
training, and excludes later parent-process evaluation. It is not resident
model memory. JSON checkpoint bytes are serialization size, not compression.

| Shuffle seed | Training seconds | Whole-training-worker peak MiB | Checkpoint bytes | Blend construction seconds / bytes |
| --- | ---: | ---: | ---: | --- |
| 5042 | 11.718 | 114.30 | 496,060 | .0141 / 496,265 |
| 101 | 13.687 | 115.25 | 496,009 | .0274 / 496,195 |

These measurements are descriptive. The historical reference weights were
reused rather than freshly trained here; no matched training-speed or memory
improvement is claimed. Per-case inference timings are stored separately.

Local tests passed **631 cases, one skipped and nine subtests**. Fresh isolated
sdist/wheel builds and the complete installed-wheel smoke ran outside the
checkout, including checkpoint blending and Unicode reconstruction. The first
non-isolated build lacked `setuptools.build_meta`; an isolated build resolved
that packaging environment issue without changing the numerical environment.
The completed-study resume validated all records without retraining or rescoring.
Separate recomputation from public rows matched all 91 aggregate cells and
32 reported intervals. All 1,175 identities, 8,225 results, continuation/phase
hashes, 46 source fingerprints and public privacy checks passed.

Evidence: [prepared plan](../reports/retention-guard-2026-11-01/plan.json),
[complete summary and failed gates](../reports/retention-guard-2026-11-01/summary.json),
[paired case scores](../reports/retention-guard-2026-11-01/paired_scores.jsonl).
Frozen contract:
`cfeca986d1c1cd82efd4a513716c50d417bc3cc6fb64516aa4faab006322cfaf`.
The four fitted checkpoint files used here remain local (the two historical
rehearsal and two new priority fits), as do the starting and blended checkpoints.

With those historical local artifacts restored, from the repository:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python scripts/benchmark_retention_guard.py \
  --output .cache/retention-guard-pilot-v1 --prepare-only --download
python scripts/benchmark_retention_guard.py \
  --output .cache/retention-guard-pilot-v1 \
  --publish reports/retention-guard-2026-11-01
```

Use a new output/publication directory for a different runtime or new study.
Existing evidence is immutable. Resume with the same output and no `--publish`
validates completed records. An incomplete attempt blocks implicit retries.
Public receipts alone cannot reconstruct the unpublished historical weights.

This is bounded segmentation and retention evidence. It does not establish
Ukrainian fluency, broad Chinese retention, calibrated correctness or production
readiness. The joint gate failures remain part of the result.
