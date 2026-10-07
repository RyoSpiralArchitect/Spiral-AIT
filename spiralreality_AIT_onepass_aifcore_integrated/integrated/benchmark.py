"""Benchmarking utilities integrating perturbations and metrics reporting."""

from __future__ import annotations

import json
import hashlib
import math
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from .augmentation import PerturbationGenerator
from .corpus import (
    _materialize_segments,
    corpus_catalog,
    corpus_license,
    naive_segments,
)
from .datasets import iter_samples
from .multilingual import build_multilingual_corpus, language_histogram
from .onepass_ait import OnePassAIT, StudentTrainingConfig

import numpy as real_numpy

from . import np_stub


def segmentation_f1(text: str, gold_segments: Sequence[str], predicted_segments: Sequence[str]) -> float:
    """Compute boundary F1 only for segmentations that reconstruct the text."""

    def cuts(segments: Sequence[str]) -> List[int]:
        if any(not isinstance(tok, str) or not tok for tok in segments) or "".join(segments) != text:
            raise ValueError("segments must be nonempty strings that reconstruct the input text")
        idx = 0
        out: List[int] = []
        for tok in segments:
            idx += len(tok)
            out.append(idx)
        if out and out[-1] == len(text):
            out.pop()
        return out

    gold = set(cuts(gold_segments))
    pred = set(cuts(predicted_segments))
    if not gold and not pred:
        return 1.0
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return 2.0 * tp / (2 * tp + fp + fn)


_TEXT_LANG_MAP: Dict[str, str] = {sample.text: sample.language for sample in iter_samples()}


@dataclass(frozen=True)
class CorpusSplit:
    """Unique examples; the test partition never enters student training."""

    train_texts: Tuple[str, ...]
    train_segments: Tuple[Tuple[str, ...], ...]
    train_languages: Tuple[str, ...]
    test_texts: Tuple[str, ...]
    test_segments: Tuple[Tuple[str, ...], ...]
    test_languages: Tuple[str, ...]
    receipt: Dict[str, object]


def split_corpus(
    texts: Sequence[str],
    segments: Sequence[Sequence[str]],
    languages: Sequence[str],
    *,
    test_fraction: float = 0.25,
    max_samples: int | None = None,
    seed: int = 5042,
) -> CorpusSplit:
    """Deduplicate exact texts, cap across languages, and split within languages.

    A cap selects at least two examples per represented language where possible.
    Languages omitted by a small cap are listed in the receipt. Singleton strata
    are training-only; at least one language must support a disjoint test example.
    Sorting seeded hashes makes membership independent of source record order.
    Duplicate rows are collapsed before *both* the external and internal splits.
    Conflicting labels or language tags for duplicate text are rejected.
    """
    if not (len(texts) == len(segments) == len(languages)):
        raise ValueError("texts, segments, and languages must have equal lengths")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between zero and one")
    if max_samples is not None and (isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 2):
        raise ValueError("max_samples must be None or an integer of at least two")

    records: Dict[str, Tuple[str, Tuple[str, ...], str]] = {}
    source_hashes: List[str] = []
    for text, seg, lang in zip(texts, segments, languages):
        if not isinstance(text, str) or not text or not isinstance(lang, str) or not lang:
            raise ValueError("each example must have nonempty text and language")
        seg = tuple(seg)
        segmentation_f1(text, seg, seg)  # Validate exact label alignment.
        record = (text, seg, lang)
        if text in records and records[text] != record:
            raise ValueError("duplicate text has conflicting segments or language")
        records[text] = record
        source_hashes.append(hashlib.sha256(json.dumps(record, ensure_ascii=False).encode("utf-8")).hexdigest())

    def rank(value: str) -> str:
        return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()

    strata: Dict[str, List[Tuple[str, Tuple[str, ...], str]]] = {}
    for record in records.values():
        strata.setdefault(record[2], []).append(record)
    for rows in strata.values():
        rows.sort(key=lambda row: rank(row[0]))
    eligible = sorted((lang for lang, rows in strata.items() if len(rows) >= 2), key=rank)
    if not eligible:
        raise ValueError("held-out evaluation needs at least one language with two distinct texts")

    limit = len(records) if max_samples is None else min(max_samples, len(records))
    if limit == len(records):
        selected = {lang: list(rows) for lang, rows in strata.items()}
    else:
        selected: Dict[str, List[Tuple[str, Tuple[str, ...], str]]] = {}
        remaining = limit
        for lang in eligible:
            if remaining < 2:
                break
            selected[lang] = strata[lang][:2]
            remaining -= 2
        # Fill remaining capacity round-robin, preserving the paired strata.
        fill_order = sorted(selected, key=rank) + sorted((lang for lang, rows in strata.items() if len(rows) == 1), key=rank)
        while remaining:
            progressed = False
            for lang in fill_order:
                count = len(selected.get(lang, []))
                if count < len(strata[lang]):
                    selected.setdefault(lang, []).append(strata[lang][count])
                    remaining -= 1
                    progressed = True
                    if not remaining:
                        break
            if not progressed:
                break  # An odd cap may leave one slot rather than split a pair.

    train: List[Tuple[str, Tuple[str, ...], str]] = []
    test: List[Tuple[str, Tuple[str, ...], str]] = []
    for lang in sorted(selected):
        rows = selected[lang]
        test_count = min(len(rows) - 1, max(1, int(len(rows) * test_fraction + 0.5))) if len(rows) > 1 else 0
        test.extend(rows[:test_count])
        train.extend(rows[test_count:])
    train.sort(key=lambda row: rank("train\0" + row[0]))
    test.sort(key=lambda row: rank("test\0" + row[0]))
    if not train or not test:
        raise ValueError("held-out evaluation requires nonempty training and test partitions")

    def receipt_for(rows: Sequence[Tuple[str, Tuple[str, ...], str]]) -> Dict[str, object]:
        manifest = json.dumps(sorted(rows), ensure_ascii=False, separators=(",", ":"))
        return {
            "samples": len(rows),
            "languages": language_histogram([row[2] for row in rows]),
            "text_sha256": sorted(hashlib.sha256(row[0].encode("utf-8")).hexdigest() for row in rows),
            "manifest_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
        }

    receipt: Dict[str, object] = {
        "method": "seeded_exact_text_groups_language_balanced_cap_stratified_holdout_v1",
        "seed": seed,
        "test_fraction": test_fraction,
        "max_samples": max_samples,
        "source_samples": len(texts),
        "unique_samples": len(records),
        "duplicate_samples_removed": len(texts) - len(records),
        "source_manifest_sha256": hashlib.sha256("\n".join(sorted(source_hashes)).encode("ascii")).hexdigest(),
        "source_languages": language_histogram(list(languages)),
        "selected_samples": len(train) + len(test),
        "omitted_languages": sorted(set(strata) - set(selected)),
        "training_only_languages": sorted(set(row[2] for row in train) - set(row[2] for row in test)),
        "train_pool": receipt_for(train),
        "test": receipt_for(test),
        "disjoint_text_hashes": True,
        "internal_validation": "Created only from train_pool; never from the held-out test partition.",
        "duplicate_policy": "Exact text duplicates collapsed; conflicting labels/tags rejected; near-duplicates are not detected.",
    }
    return CorpusSplit(
        train_texts=tuple(row[0] for row in train),
        train_segments=tuple(row[1] for row in train),
        train_languages=tuple(row[2] for row in train),
        test_texts=tuple(row[0] for row in test),
        test_segments=tuple(row[1] for row in test),
        test_languages=tuple(row[2] for row in test),
        receipt=receipt,
    )


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _language_for_text(text: str, fallback: str) -> str:
    return _TEXT_LANG_MAP.get(text, fallback)


def load_benchmark_split(
    *,
    languages: Sequence[str] | None = None,
    include_reflective: bool = True,
    max_samples: int | None = None,
    test_fraction: float = 0.25,
    seed: int = 5042,
) -> CorpusSplit:
    """Load curated text-covering labels and make an auditable held-out split."""
    texts, segments, tags = build_multilingual_corpus(
        languages=languages,
        include_reflective=include_reflective,
        shuffle=False,
        seed=seed,
    )
    return split_corpus(
        texts, segments, [_language_for_text(text, tag) for text, tag in zip(texts, tags)],
        max_samples=max_samples, test_fraction=test_fraction, seed=seed,
    )


def training_report(split: CorpusSplit, summary: Mapping[str, object], validation_fraction: float) -> Dict[str, object]:
    """Make training diagnostics JSON-safe and receipt exposed internal indices."""
    def json_ready(value):
        if isinstance(value, real_numpy.generic):
            return value.item()
        if isinstance(value, real_numpy.ndarray):
            return value.tolist()
        if isinstance(value, Mapping):
            return {key: json_ready(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_ready(item) for item in value]
        return value

    report: Dict[str, object] = {
        "evaluation_partition": "training_pool_internal_validation",
        "internal_validation_fraction": validation_fraction,
        "summary": json_ready(summary),
    }
    if "train_indices" in summary and "validation_indices" in summary:
        train_indices = list(summary["train_indices"])
        validation_indices = list(summary["validation_indices"])
        if (set(train_indices) & set(validation_indices) or
                sorted(train_indices + validation_indices) != list(range(len(split.train_texts)))):
            raise ValueError("training summary returned invalid train/validation partition indices")
        for name, indices in (("train", train_indices), ("validation", validation_indices)):
            report[name] = {
                "samples": len(indices),
                "text_sha256": sorted(hashlib.sha256(split.train_texts[index].encode("utf-8")).hexdigest() for index in indices),
                "languages": language_histogram([split.train_languages[index] for index in indices]),
            }
    else:
        report["internal_split_receipt"] = "Backend did not expose internal split indices; only the external train_pool/test separation is verified."
    return report


def _compare_numpy_stub(seed: int = 0) -> Dict[str, float | bool | None]:
    rng = real_numpy.random.default_rng(seed)
    base = rng.standard_normal(64).reshape(8, 8)
    scaled = base * 1.5 + rng.standard_normal(base.shape) * 0.05
    stub_arr = np_stub.array(scaled.tolist())
    stub_result = np_stub.tanh(stub_arr).to_list()
    real_result = real_numpy.tanh(scaled)
    diff = real_result - real_numpy.array(stub_result)
    linf = float(real_numpy.max(real_numpy.abs(diff)))
    mse = float(real_numpy.mean(diff**2))
    return {"available": True, "linf": linf, "mse": mse}


def _encode_latencies(ait: OnePassAIT, texts: Sequence[str], runs: int = 3) -> List[float]:
    latencies: List[float] = []
    for _ in range(max(1, runs)):
        for text in texts:
            ait._encode_cache.pop(text, None)
            start = time.perf_counter()
            ait.encode(text)
            latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def _segment_latencies(segmenter, texts: Sequence[str], runs: int = 1) -> List[float]:
    latencies: List[float] = []
    for _ in range(max(1, runs)):
        for text in texts:
            start = time.perf_counter()
            segmenter(text)
            latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def run_benchmark(
    *,
    languages: Sequence[str] | None = None,
    include_reflective: bool = True,
    max_samples: int | None = 12,
    output_dir: str | None = "reports",
    seed: int = 5042,
    test_fraction: float = 0.25,
) -> Dict[str, object]:
    """Train on a separate pool and score only held-out examples and variants."""
    split = load_benchmark_split(
        languages=languages,
        include_reflective=include_reflective,
        max_samples=max_samples,
        test_fraction=test_fraction,
        seed=seed,
    )
    texts, segments, tags = split.test_texts, split.test_segments, split.test_languages

    ait = OnePassAIT(latent_dim=32, seed=seed)
    cfg = StudentTrainingConfig(
        lr=0.05,
        epochs=8,
        batch_size=2,
        validation_split=0.25 if len(split.train_texts) > 1 else 0.0,
        patience=3,
        hidden_dim=20,
        emb_dim=14,
        window=2,
        phase_lr=0.3,
        cache_sequences=False,
        shuffle_train=False,
    )
    training_summary = ait.train_student(split.train_texts, split.train_segments, cfg=cfg)

    baseline_scores: Dict[str, float] = {}
    rule_scores: Dict[str, float] = {}
    generator = PerturbationGenerator(seed=seed)
    variant_scores: Dict[str, List[float]] = {}
    variant_drop: Dict[str, List[float]] = {}
    variant_changed: Dict[str, int] = {}
    variant_collisions: Dict[str, int] = {}
    training_texts = set(split.train_texts)
    actual_languages: List[str] = []

    for text, seg, tag in zip(texts, segments, tags):
        lang = _language_for_text(text, tag)
        actual_languages.append(lang)
        predicted_result = ait.student.decode_with_logit_bias(text, 0.0)
        predicted = predicted_result["tokens"] if isinstance(predicted_result, dict) else predicted_result
        f1 = segmentation_f1(text, seg, predicted)
        baseline_scores[text] = f1
        rule_segments = _materialize_segments(text, naive_segments(text))
        rule_scores[text] = segmentation_f1(text, seg, rule_segments)
        for variant in generator.generate_variants(text, seg, language=lang):
            if variant.text in training_texts:
                variant_collisions[variant.tag] = variant_collisions.get(variant.tag, 0) + 1
                continue
            pred_result = ait.student.decode_with_logit_bias(variant.text, 0.0)
            pred_segments = pred_result["tokens"] if isinstance(pred_result, dict) else pred_result
            vf1 = segmentation_f1(variant.text, variant.segments, pred_segments)
            variant_scores.setdefault(variant.tag, []).append(vf1)
            variant_changed[variant.tag] = variant_changed.get(variant.tag, 0) + int(variant.text != text)
            baseline = f1
            drop = 0.0 if baseline <= 1e-8 else max(0.0, (baseline - vf1) / baseline)
            variant_drop.setdefault(variant.tag, []).append(drop)

    def evaluate_setting(*, context: bool, aif: bool) -> Dict[str, object]:
        original_context = bool(getattr(ait.student, "use_encoder_context", True))
        scores: Dict[str, float] = {}
        policy_hist: Dict[str, int] = {}
        seg_latencies: List[float] = []

        def segment_one(text: str) -> Sequence[str]:
            if aif:
                result = ait.segment_text(text, return_metadata=True, use_aif=True)
                if isinstance(result, dict):
                    policy = result.get("chosen_policy")
                    if isinstance(policy, str) and policy:
                        policy_hist[policy] = policy_hist.get(policy, 0) + 1
                    return result.get("tokens", [])
                return result  # type: ignore[return-value]
            decoded = ait.student.decode_with_logit_bias(text, 0.0)
            return decoded["tokens"] if isinstance(decoded, dict) else decoded  # type: ignore[return-value]

        try:
            ait.student.use_encoder_context = bool(context)
            for text, seg in zip(texts, segments):
                start = time.perf_counter()
                predicted = list(segment_one(text))
                seg_latencies.append((time.perf_counter() - start) * 1000.0)
                scores[text] = segmentation_f1(text, seg, predicted)
        finally:
            ait.student.use_encoder_context = original_context

        mean_latency = float(statistics.mean(seg_latencies)) if seg_latencies else 0.0
        p95_latency = _percentile(seg_latencies, 95.0)
        max_latency = max(seg_latencies) if seg_latencies else 0.0

        info: Dict[str, object] = {
            "evaluation_partition": "held_out_test",
            "kind": "inference_ablation",
            "retrained": False,
            "mean_f1": float(statistics.mean(scores.values())) if scores else 0.0,
            "per_text": scores,
            "segment_latency_ms": {
                "samples": len(seg_latencies),
                "mean": mean_latency,
                "p95": p95_latency,
                "max": max_latency,
            },
        }
        if aif:
            info["policy_hist"] = policy_hist
        return info

    latencies = _encode_latencies(ait, texts)
    mean_latency = float(statistics.mean(latencies)) if latencies else 0.0
    p95_latency = _percentile(latencies, 95.0)
    max_latency = max(latencies) if latencies else 0.0

    dataset_hist = language_histogram(list(split.train_languages) + actual_languages)
    catalog_languages = sorted(dataset_hist)
    dataset_info = {
        "size": len(split.train_texts) + len(texts),
        "languages": dataset_hist,
        "test_languages": language_histogram(actual_languages),
        "split": split.receipt,
        "label_source": "repository-curated teacher segmentations",
        "scope": "Small synthetic corpus; no external-corpus generalization claim.",
        "license": corpus_license(),
        "catalog": corpus_catalog(catalog_languages),
    }

    baseline_info = {
        "evaluation_partition": "held_out_test",
        "samples": len(texts),
        "f1": float(statistics.mean(baseline_scores.values())) if baseline_scores else 0.0,
        "per_text": baseline_scores,
        "per_language": {
            lang: {
                "samples": actual_languages.count(lang),
                "f1": statistics.mean(baseline_scores[text] for text, text_lang in zip(texts, actual_languages) if text_lang == lang),
            }
            for lang in sorted(set(actual_languages))
        },
        "latency_ms": {
            "samples": len(latencies),
            "mean": mean_latency,
            "p95": p95_latency,
            "max": max_latency,
        },
    }

    ablations = {
        "context_on_aif_off": evaluate_setting(context=True, aif=False),
        "context_on_aif_on": evaluate_setting(context=True, aif=True),
        "context_off_aif_off": evaluate_setting(context=False, aif=False),
        "context_off_aif_on": evaluate_setting(context=False, aif=True),
    }

    variants_info: Dict[str, Dict[str, float]] = {}
    for tag, scores in variant_scores.items():
        mean_score = float(statistics.mean(scores)) if scores else 0.0
        drops = variant_drop.get(tag, [])
        mean_drop = float(statistics.mean(drops)) if drops else 0.0
        variants_info[tag] = {
            "mean_f1": mean_score,
            "mean_degradation": mean_drop,
            "samples": len(scores),
            "changed_samples": variant_changed.get(tag, 0),
            "excluded_training_collisions": variant_collisions.get(tag, 0),
        }

    np_metrics = _compare_numpy_stub(seed=seed)

    metrics: Dict[str, object] = {
        "schema_version": 2,
        "runtime": {
            "python": platform.python_version(),
            "numpy": real_numpy.__version__,
            "machine": platform.machine(),
            "encoder_backend": ait.encoder_backend_name(),
        },
        "training_config": asdict(cfg),
        "dataset": dataset_info,
        "baseline": baseline_info,
        "reference_baselines": {
            "whitespace_punctuation": {
                "evaluation_partition": "held_out_test",
                "f1": float(statistics.mean(rule_scores.values())),
                "per_text": rule_scores,
                "description": "Text-covering whitespace/punctuation rule, with no curated-label lookup.",
            },
        },
        "training": training_report(split, training_summary, cfg.validation_split),
        "protocol": {
            "score_partition": "held_out_test",
            "variant_source_partition": "held_out_test",
            "variant_training_collisions_excluded": variant_collisions,
            "ablation_kind": "inference_ablation",
            "ablation_note": "Context and AIF are toggled at inference on one trained student; this is not a separately retrained architecture comparison.",
            "teacher_note": "Curated teacher segments define the gold labels; teacher lookup is not an independent baseline.",
            "latency_note": "Encode cache is cleared before each baseline latency sample; ablation latency is descriptive and not a controlled speed comparison.",
        },
        "ablations": ablations,
        "variants": variants_info,
        "np_stub": np_metrics,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "benchmark_report.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        _write_markdown_report(metrics, os.path.join(output_dir, "benchmark_report.md"))

    return metrics


def _write_markdown_report(metrics: Mapping[str, object], path: str) -> None:
    dataset = metrics.get("dataset", {})
    baseline = metrics.get("baseline", {})
    ablations = metrics.get("ablations", {})
    variants = metrics.get("variants", {})
    np_metrics = metrics.get("np_stub", {})

    lines = ["# SpiralReality AIT Benchmark", ""]
    if isinstance(dataset, Mapping):
        split = dataset.get("split", {})
        if isinstance(split, Mapping):
            train = split.get("train_pool", {})
            test = split.get("test", {})
            lines.extend([
                "## Evaluation Protocol", "",
                f"- Unique selected samples: {split.get('selected_samples')}; train pool: {train.get('samples')}; held-out test: {test.get('samples')}.",
                "- Training and internal validation use only the train pool. Headline F1 and perturbation scores use the held-out test.",
                f"- Split seed: {split.get('seed')}; exact duplicate rows removed: {split.get('duplicate_samples_removed')}.",
                f"- Languages omitted by the sample cap: {', '.join(split.get('omitted_languages', [])) or 'none'}.",
                "- Text and label manifest hashes are recorded in the JSON report.",
                "- This small synthetic corpus does not establish external-corpus generalization.", "",
            ])
    license_info = dataset.get("license", {}) if isinstance(dataset, Mapping) else {}
    if license_info:
        lines.append("## Dataset License")
        lines.append("")
        for key in ("id", "name", "url", "attribution"):
            value = license_info.get(key)
            if value:
                lines.append(f"- **{key}**: {value}")
        notes = license_info.get("notes")
        if notes:
            lines.append(f"- **notes**: {notes}")
        lines.append("")

    lines.append("## Held-out Student Metrics")
    lines.append("")
    if isinstance(baseline, Mapping):
        f1 = baseline.get("f1", 0.0)
        latency = baseline.get("latency_ms", {})
        lines.append(f"- Mean F1: {f1:.4f}")
        if isinstance(latency, Mapping):
            lines.append(
                "- Encode latency (ms): mean={:.3f}, p95={:.3f}, max={:.3f}".format(
                    latency.get("mean", 0.0),
                    latency.get("p95", 0.0),
                    latency.get("max", 0.0),
                )
            )
    lines.append("")

    if isinstance(baseline, Mapping) and baseline.get("per_language"):
        lines.extend(["| Language | Test samples | Mean F1 |", "| --- | --- | --- |"])
        for language, info in baseline["per_language"].items():
            lines.append(f"| {language} | {info['samples']} | {info['f1']:.4f} |")
        lines.extend(["", "Per-language counts are small; these scores are descriptive, not general accuracy estimates.", ""])

    references = metrics.get("reference_baselines", {})
    if isinstance(references, Mapping) and references:
        lines.extend(["## Independent Rule Baseline", ""])
        for name, info in references.items():
            lines.append(f"- {name}: held-out mean F1={info['f1']:.4f}. {info['description']}")
        lines.append("Curated teacher lookup defines gold labels and is not an independent prediction baseline.")
        lines.append("")

    if isinstance(ablations, Mapping) and ablations:
        lines.append("## Inference Ablations (Context / AIF)")
        lines.append("")
        lines.append("One trained student; context-off is an inference ablation, not a retrained architecture comparison.")
        lines.append("")
        lines.append("| Setting | Mean F1 | Segment latency mean (ms) | p95 (ms) | Policy hist |")
        lines.append("| --- | --- | --- | --- | --- |")
        for key, info in ablations.items():
            if not isinstance(info, Mapping):
                continue
            mean_f1 = float(info.get("mean_f1", 0.0) or 0.0)
            seg_latency = info.get("segment_latency_ms", {})
            seg_mean = seg_p95 = 0.0
            if isinstance(seg_latency, Mapping):
                seg_mean = float(seg_latency.get("mean", 0.0) or 0.0)
                seg_p95 = float(seg_latency.get("p95", 0.0) or 0.0)
            policy_hist = info.get("policy_hist")
            policy_str = ""
            if isinstance(policy_hist, Mapping) and policy_hist:
                items = sorted(policy_hist.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
                policy_str = ", ".join(f"{name}:{count}" for name, count in items)
            lines.append(f"| {key} | {mean_f1:.4f} | {seg_mean:.3f} | {seg_p95:.3f} | {policy_str} |")
        lines.append("")

    if variants:
        lines.append("## Perturbation Variants")
        lines.append("")
        lines.append("| Variant | Mean F1 | Degradation (%) | Samples | Changed texts |")
        lines.append("| --- | --- | --- | --- | --- |")
        for tag, info in variants.items():
            if not isinstance(info, Mapping):
                continue
            mean_f1 = info.get("mean_f1", 0.0)
            mean_drop = info.get("mean_degradation", 0.0) * 100.0
            samples = info.get("samples", 0)
            lines.append(f"| {tag} | {mean_f1:.4f} | {mean_drop:.2f} | {samples} | {info.get('changed_samples', 0)} |")
        lines.append("")

    if isinstance(np_metrics, Mapping) and np_metrics.get("available"):
        lines.append("## NumPy vs np_stub")
        lines.append(
            "- L_inf error: {linf:.6f}\n- MSE: {mse:.6f}".format(
                linf=np_metrics.get("linf", 0.0) or 0.0,
                mse=np_metrics.get("mse", 0.0) or 0.0,
            )
        )
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


__all__ = [
    "run_benchmark",
    "segmentation_f1",
    "CorpusSplit",
    "split_corpus",
    "load_benchmark_split",
    "training_report",
]
