# English, Japanese, and Chinese context-learning pilot

This experiment adds a differentiable NumPy context transformer and tests it
against the same model with frozen transformer features. Character vectors and
phase inputs remain fixed. Neither arm is a pretrained language model.

## Completed results and decision

All ten folds and twenty training runs completed. Each language has 997 held-out
sentences, each evaluated once, across 397 document groups. The frozen contract
is `da11014e4e72c492f818cda0d304b7ca21180e6d2babd7b20358d159441418b5`.
See the [prepared plan](../reports/trilingual-context-2026-10-08/plan.json),
[complete summary](../reports/trilingual-context-2026-10-08/summary.json), and
[paired sentence scores](../reports/trilingual-context-2026-10-08/paired_scores.jsonl).
The preparation commit identifies the base checkout; the plan's per-file source
hashes identify the implementation actually evaluated, including uncommitted
implementation files at preparation time.

**The supervised transformer did not improve segmentation under this budget.**
Keep it opt-in and leave the existing default unchanged. The new capability is
verified by gradient checks and actual parameter changes, not by a quality gain.

| Language | Frozen encoder macro F1 | Trained encoder macro F1 | Difference (percentage points) | Conditional document 95% interval (points) |
| --- | ---: | ---: | ---: | ---: |
| English | 99.426% | 99.419% | -0.0066 | [-0.0188, +0.0041] |
| Japanese | 83.456% | 83.409% | -0.0474 | [-0.1538, +0.0619] |
| Chinese | 79.399% | 79.192% | -0.2067 | [-0.3876, -0.0231] |

The English and Japanese intervals include zero; the Chinese interval is
negative conditional on these fitted models. None establishes a universal
effect across training seeds or budgets. Encoder parameter change was exactly
zero in every frozen run and L2 0.2650–0.3240 in the trained runs.

Macro F1 averages sentence-level internal-boundary F1; a sentence with no gold
or predicted boundaries scores one. Whitespace gaps are explicit spans, so
English gets many easy boundaries. Excluding positions touching whitespace,
macro F1 is 93.977% → 93.925% for English, 83.415% → 83.368% for Japanese, and
79.249% → 79.044% for Chinese. The summary also contains pooled micro F1.
The punctuation/whitespace rule baseline scores 68.600%, 30.700%, and 30.709%
respectively; it is only an internal reference, not a competitive tokenizer.

### Calibration and streaming

Development-fitted marginal calibration reduced held-out NLL and Brier error in
all three languages in both arms. ECE improved in English and Japanese but
**worsened in Chinese**. For the frozen encoder, pooled metrics are:

| Language | NLL raw → calibrated | Brier raw → calibrated | ECE raw → calibrated |
| --- | ---: | ---: | ---: |
| English | 0.01326 → 0.01253 | 0.003030 → 0.003019 | 0.002925 → 0.001220 |
| Japanese | 0.42127 → 0.39169 | 0.130888 → 0.119774 | 0.100926 → 0.030749 |
| Chinese | 0.50770 → 0.50356 | 0.172016 → 0.169221 | 0.045558 → 0.054361 |

Using those temperatures in the confidence-gated streaming wrapper, still with
the frozen encoder:

| Language | Streaming macro F1 raw → calibrated | Mean wait (characters) | p95 wait (characters) | Capacity-forced cuts |
| --- | ---: | ---: | ---: | ---: |
| English | 95.854% → 96.722% | 21.78 → 21.56 | 31 → 31 | 90 → 68 |
| Japanese | 83.011% → 83.066% | 24.88 → 24.09 | 59 → 57 | 9 → 8 |
| Chinese | 79.256% → 79.258% | 18.96 → 18.97 | 46 → 46 | 3 → 3 |

This is most promising for English streaming, with little segmentation change
in Japanese or Chinese. These are descriptive results without a paired interval
for calibration/streaming. Mean and p95 waiting do not bound every output:
English maximum wait increased from 102 to 106 characters. Wait counts characters
received after a token's end, including the fixed-size feed that triggers output;
it is not a latency guarantee. Calibration remains an evaluation wrapper and
has not been promoted to runtime defaults or checkpoint semantics.

### AIF and next experiment

The existing AIF settings chose `SeekEvidence` for **every one of the 997 texts
in every language and both arms**. With the frozen encoder, AIF macro F1 was
99.417%, 83.053%, and 79.347%, below the corresponding direct decoder in each
language. This run provides no evidence of useful policy diversity or a quality
gain from AIF; making it the default is not supported.

A subsequent, separately frozen experiment should test larger whole-document
training budgets and multiple initialization seeds, with learning-rate choices
made using development data only. Trainable character representations and an
AIF policy ablation are further hypotheses, not conclusions from this run.
Confirm any selected setting on an independent corpus before claiming a broader
quality gain: these PUD results have now been inspected. Preserve this negative
result rather than replacing it with a tuned rerun.

### Validation

- 316 tests passed, one skipped, and nine subtests passed locally. New checks
  cover finite-difference gradients through every transformer parameter family
  and the full CRF path, batch averaging, checkpoint reconstruction, source
  integrity, document isolation, calibration aggregation, and streaming rollback.
- Source distribution and wheel built successfully. A fresh environment outside
  the checkout verified three-language supervised updates, checkpoint round
  trips, AIF text preservation, and evaluation-module imports from the wheel.
- Verified the frozen contract against current source-file hashes, every local
  fold receipt and checkpoint, disjoint train/dev/test groups, and all 2,991
  paired scores. Published evidence contains no source text or learned weights.
- Native gradient kernels were not implemented or evaluated; supervised context
  learning requires the explicit NumPy encoder.

## Frozen protocol

- PUD r2.18, 1,000 source sentences in each language; exact commits, file lengths,
  data hashes, README hashes, and license hashes are pinned in
  [the source manifest](../configs/pud-v2.18-sources.json).
- CoNLL-U surface token forms are aligned exactly with the original sentence.
  Multiword-token rows replace their components; empty nodes are excluded;
  whitespace gaps are retained as separate spans. No generated labels or silent
  normalization are used. All 3,000 sentences align successfully.
- Apply the 256-character limit to the whole parallel group. Three parallel
  groups are excluded, leaving 997 sentences per language and 397 document groups.
- All sentences from a source document and their three translations stay in one
  fold. Documents sharing Unicode-normalized exact text are conservatively
  merged. Ten hash-ordered folds are fixed before fitting. Semantic near-duplicate
  detection beyond those identifiers and normalized exact text is not claimed.
- Each fold uses the next fold for development and the other eight for training.
  Training is capped at 30 whole document groups (79–88 sentences per language).
  The same document cap, initialization seed, example order, lexical features,
  and eight-epoch budget apply to both arms. There is no sentence-level internal
  validation split and no early stopping in this experiment.
- Transformer: 16 features, one layer, two attention heads. Both arms train the
  category/context/lexical/CRF head. Phase updates are disabled. The frozen arm
  also disables heuristic gate tuning; the other arm uses supervised gradients
  for every transformer parameter family, including gates and normalization.
- Fit one marginal log-odds temperature per language using development NLL only,
  from a fixed grid. Full-text Viterbi decoding is unchanged. Apply the selected
  temperatures to held-out confidence and to a streaming wrapper's probabilities.
  Calibration is an experiment-stage transformation, not a new default runtime
  setting or a calibrated checkpoint format.
- Streaming uses 16-character feeds, a 96-character maximum window, 16 characters
  of lookahead, 32 characters of context, and confidence threshold 0.8. Report
  boundary F1, waiting in received characters, and actual forced-cut counts.
  AIF is evaluated with its existing fixed settings, without test-set tuning.

The complete settings are in [the experiment config](../configs/trilingual-context-pilot.json).
The prepared plan fingerprints implementation files as well as the source and
partition manifests. Local detailed fold receipts and checkpoints are retained
under the chosen output directory. Compact published evidence includes hashes
of those receipts, model checkpoints, labels, and predictions; raw text and
learned model weights are not redistributed by this change.

## Interpretation limits

This is a deliberately small-training-budget, custom document cross-validation
pilot. PUD's upstream partition is entirely named `test`; the maintainers suggest
ten-fold cross-validation when using it for training. These scores are not an
official CoNLL test result and should not be compared with systems trained on
unrelated full treebanks as if training conditions were matched.

PUD is parallel news/Wikipedia text: 750 source sentences originate in English,
and 250 in German, French, Italian, or Spanish. Japanese and Chinese are
translations. Performance on native informal Japanese/Chinese, speech, or other
domains remains unmeasured. Report language-specific scores and non-whitespace
boundary scores so easy English spaces do not hide harder segmentation errors.

Document bootstrap intervals resample paired documents conditional on the fitted
fold models. They do not capture initialization uncertainty or dependencies
among overlapping CV training sets. There is one initialization seed per fold.
Waiting is measured in received characters, not real-world input time. CPU
latencies are local descriptive measurements, not a platform speed claim.

## Attribution and source terms

Data and translations are distributed upstream under **CC BY-SA 3.0**. The
provider's README preserves its caveat about ownership of underlying source
text; that caveat is retained in the source manifest. Source material stays in
the local cache with its complete original README and LICENSE. Repository code
remains under this repository's Apache-2.0 license; that license does not replace
the dataset's terms.

- [UD English PUD](https://github.com/UniversalDependencies/UD_English-PUD/tree/e173a1be1b442faf34e7d5a502189ad5d9d1e197)
- [UD Japanese PUD](https://github.com/UniversalDependencies/UD_Japanese-PUD/tree/4abd575c57bfa125dd4bc564f2ceb8973bbbf422)
- [UD Chinese PUD](https://github.com/UniversalDependencies/UD_Chinese-PUD/tree/7b54411fb8cec041c4f8794413e08878605792f3)
- [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

Credit the PUD translators and annotation teams (DFKI, Google, and the Universal
Dependencies contributors) and each upstream README's contributor list when
using the material. The pinned READMEs are the complete attribution source.
