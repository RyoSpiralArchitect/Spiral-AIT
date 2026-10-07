# SpiralReality AIT Benchmark

## Evaluation Protocol

- Unique selected samples: 33; train pool: 23; held-out test: 10.
- Training and internal validation use only the train pool. Headline F1 and perturbation scores use the held-out test.
- Split seed: 5042; exact duplicate rows removed: 0.
- Languages omitted by the sample cap: none.
- Text and label manifest hashes are recorded in the JSON report.
- This small synthetic corpus does not establish external-corpus generalization.

## Dataset License

- **id**: CC-BY-4.0
- **name**: Creative Commons Attribution 4.0 International
- **url**: https://creativecommons.org/licenses/by/4.0/
- **attribution**: SpiralReality AIT authors
- **notes**: Synthetic reflective and multilingual narratives curated for the SpiralReality one-pass AIT demo. The texts were authored for this repository and may be redistributed under CC-BY-4.0.

## Held-out Student Metrics

- Mean F1: 0.9039
- Encode latency (ms): mean=11.167, p95=14.704, max=15.946

| Language | Test samples | Mean F1 |
| --- | --- | --- |
| de | 1 | 1.0000 |
| en | 5 | 0.9741 |
| es | 1 | 1.0000 |
| fr | 1 | 0.9565 |
| ja | 1 | 0.5455 |
| zh | 1 | 0.6667 |

Per-language counts are small; these scores are descriptive, not general accuracy estimates.

## Independent Rule Baseline

- whitespace_punctuation: held-out mean F1=0.8111. Text-covering whitespace/punctuation rule, with no curated-label lookup.
Curated teacher lookup defines gold labels and is not an independent prediction baseline.

## Inference Ablations (Context / AIF)

One trained student; context-off is an inference ablation, not a retrained architecture comparison.

| Setting | Mean F1 | Segment latency mean (ms) | p95 (ms) | Policy hist |
| --- | --- | --- | --- | --- |
| context_on_aif_off | 0.9039 | 7.084 | 8.887 |  |
| context_on_aif_on | 0.8972 | 8.032 | 9.834 | SeekEvidence:10 |
| context_off_aif_off | 0.8572 | 1.499 | 1.838 |  |
| context_off_aif_on | 0.8463 | 2.639 | 3.346 | SeekEvidence:10 |

## Perturbation Variants

| Variant | Mean F1 | Degradation (%) | Samples | Changed texts |
| --- | --- | --- | --- | --- |
| noise | 0.9039 | 0.00 | 10 | 4 |
| dialect | 0.9071 | 0.83 | 10 | 5 |
| tempo_slow | 0.8950 | 1.56 | 10 | 10 |
| tempo_fast | 0.8890 | 1.54 | 10 | 8 |

## NumPy vs np_stub
- L_inf error: 0.000000
- MSE: 0.000000
