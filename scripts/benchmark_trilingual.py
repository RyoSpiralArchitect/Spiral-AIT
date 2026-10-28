#!/usr/bin/env python3
"""Run a pinned, paired document-CV pilot without publishing source text.

Prepare first with --prepare-only to freeze the entire contract. Re-running an
output directory resumes completed folds only when code, data, and config match.
Checkpoints and detailed receipts are local; --publish writes compact evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from spiralreality_AIT_onepass_aifcore_integrated.integrated.checkpoint import save_checkpoint
from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import _materialize_segments, naive_segments
from spiralreality_AIT_onepass_aifcore_integrated.integrated.evaluation_metrics import (
    aggregate_probability_sums, boundary_counts, boundary_targets, fit_temperature,
    paired_document_interval, probability_sums, temperature_scale,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import (
    LANGUAGES, group_parallel_corpus, json_hash, load_pinned_sources, sha256,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig
from spiralreality_AIT_onepass_aifcore_integrated.integrated.streaming_inference import ChunkedStreamingSegmenter


def measure_stream(model, row, config, temperature):
    calls = 0
    def segment(text):
        nonlocal calls
        calls += 1
        result = model.segment_text(text, include_confidence=True)
        result["boundary_probabilities"] = temperature_scale(result["boundary_probabilities"], temperature).tolist()
        return result
    stream = ChunkedStreamingSegmenter(segment, **config["streaming"])
    tokens, waits, committed, peak = [], Counter(), 0, 0
    started = time.perf_counter()
    width = config["feed_chunk_chars"]
    for offset in range(0, len(row.text), width):
        received = min(offset + width, len(row.text))
        emitted = stream.feed(row.text[offset:received])
        peak = max(peak, len(stream.pending_text))
        for token in emitted:
            committed += len(token)
            waits[received - committed] += 1
        tokens.extend(emitted)
    for token in stream.flush():
        committed += len(token)
        waits[len(row.text) - committed] += 1
        tokens.append(token)
    return {"boundary": boundary_counts(row.text, row.segments, tokens),
            "nonwhitespace": boundary_counts(row.text, row.segments, tokens, exclude_whitespace=True),
            "compute_ms": (time.perf_counter() - started) * 1000,
            "forced_splits": stream.forced_split_count, "tokens": len(tokens),
            "wait_char_histogram": dict(waits), "peak_pending_chars_after_feed": peak,
            "model_calls": calls, "prediction_sha256": json_hash(tokens)}


def fit_development_temperatures(model, rows, candidates):
    by_language = {lang: ([], []) for lang in LANGUAGES}
    for row in rows:
        probabilities = model.student.boundary_probs(row.text).tolist()
        labels = boundary_targets(row.text, row.segments).tolist()
        by_language[row.language][0].extend(probabilities)
        by_language[row.language][1].extend(labels)
    return {lang: fit_temperature(probabilities, labels, candidates)
            for lang, (probabilities, labels) in by_language.items()}


def evaluate(model, rows, corpus, config, temperatures):
    results = []
    for row in rows:
        started = time.perf_counter()
        result = model.segment_text(row.text, include_confidence=True)
        elapsed = (time.perf_counter() - started) * 1000
        labels = boundary_targets(row.text, row.segments)
        probabilities = np.array(result["boundary_probabilities"])
        calibrated = temperature_scale(probabilities, temperatures[row.language])
        mask = np.array([not row.text[i].isspace() and not row.text[i + 1].isspace() for i in range(len(row.text) - 1)])
        aif = model.segment_text(row.text, use_aif=True, return_metadata=True)
        entry = {
            "language": row.language, "sentence_id": row.sentence_id, "parallel_id": row.parallel_id,
            "document_group": corpus.group_for_document[row.document_id], "characters": len(row.text),
            "labels_sha256": json_hash(row.segments), "prediction_sha256": json_hash(result["tokens"]),
            "boundary": boundary_counts(row.text, row.segments, result["tokens"]),
            "nonwhitespace": boundary_counts(row.text, row.segments, result["tokens"], exclude_whitespace=True),
            "rule": boundary_counts(row.text, row.segments, _materialize_segments(row.text, naive_segments(row.text))),
            "confidence_raw": probability_sums(probabilities, labels),
            "confidence_calibrated": probability_sums(calibrated, labels),
            "nonwhitespace_confidence_raw": probability_sums(probabilities[mask], labels[mask]),
            "nonwhitespace_confidence_calibrated": probability_sums(calibrated[mask], labels[mask]),
            "compute_ms": elapsed,
            "aif": boundary_counts(row.text, row.segments, aif["tokens"]), "aif_policy": aif["chosen_policy"],
        }
        entry["stream_raw"] = measure_stream(model, row, config, 1.0)
        entry["stream_calibrated"] = measure_stream(model, row, config, temperatures[row.language])
        results.append(entry)
    return results


def aggregate_counts(rows):
    totals = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "positions")}
    denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    return {**totals, "macro_f1": float(np.mean([row["f1"] for row in rows])),
            "micro_f1": 2 * totals["tp"] / denominator if denominator else 1.0}


def aggregate_stream(rows):
    waits = Counter()
    for row in rows:
        waits.update({int(key): count for key, count in row["wait_char_histogram"].items()})
    total = sum(waits.values())
    cumulative, p95 = 0, 0
    for value, count in sorted(waits.items()):
        cumulative += count
        if cumulative >= 0.95 * total:
            p95 = value
            break
    return {"boundary": aggregate_counts([row["boundary"] for row in rows]),
            "nonwhitespace": aggregate_counts([row["nonwhitespace"] for row in rows]),
            "forced_splits": sum(row["forced_splits"] for row in rows),
            "emitted_tokens": sum(row["tokens"] for row in rows),
            "mean_wait_chars": sum(value * count for value, count in waits.items()) / total,
            "p95_wait_chars": p95, "max_wait_chars": max(waits),
            "median_text_compute_ms": float(np.median([row["compute_ms"] for row in rows])),
            "p95_text_compute_ms": float(np.quantile([row["compute_ms"] for row in rows], 0.95))}


def summarize(fold_results, contract):
    rows = {arm: [] for arm in ("fixed", "trained")}
    for fold in fold_results:
        for arm in rows:
            rows[arm].extend(fold["arms"][arm]["results"])
    summary = {"contract_sha256": json_hash(contract), "folds_completed": len(fold_results),
               "config": contract["config"], "corpus": {key: contract["partition"][key] for key in
                   ("eligible_rows", "parallel_sentences", "document_groups", "labels_sha256")},
               "languages": {}, "limitations": [contract["config"]["protocol"],
                   "One training seed per fold; document intervals do not measure training-seed uncertainty.",
                   "Token boundary calibration is a marginal log-odds temperature transform; it does not retrain the CRF or alter full-text decoding.",
                   "Wait is measured in received characters with synthetic fixed-size feeds; CPU times are descriptive local measurements."]}
    compact_rows = []
    for lang in LANGUAGES:
        language_summary = {}
        lookup = {}
        for arm, arm_rows in rows.items():
            subset = [row for row in arm_rows if row["language"] == lang]
            lookup[arm] = {row["parallel_id"]: row for row in subset}
            language_summary[arm] = {"samples": len(subset)}
            for metric in ("boundary", "nonwhitespace", "rule", "aif"):
                language_summary[arm][metric] = aggregate_counts([row[metric] for row in subset])
            for metric in ("confidence_raw", "confidence_calibrated", "nonwhitespace_confidence_raw", "nonwhitespace_confidence_calibrated"):
                language_summary[arm][metric] = aggregate_probability_sums([row[metric] for row in subset])
            for metric in ("stream_raw", "stream_calibrated"):
                language_summary[arm][metric] = aggregate_stream([row[metric] for row in subset])
            language_summary[arm]["aif_policy_histogram"] = dict(Counter(row["aif_policy"] for row in subset))
            language_summary[arm]["median_text_compute_ms"] = float(np.median([row["compute_ms"] for row in subset]))
        if lookup["fixed"].keys() != lookup["trained"].keys():
            raise ValueError("Unmatched paired evaluation rows")
        differences = [(row["document_group"], lookup["trained"][key]["boundary"]["f1"] - row["boundary"]["f1"])
                       for key, row in lookup["fixed"].items()]
        language_summary["paired"] = paired_document_interval(differences)
        summary["languages"][lang] = language_summary
        for key, fixed in sorted(lookup["fixed"].items()):
            trained = lookup["trained"][key]
            compact_rows.append([lang, key, fixed["document_group"], fixed["labels_sha256"],
                                 fixed["boundary"]["f1"], trained["boundary"]["f1"],
                                 fixed["nonwhitespace"]["f1"], trained["nonwhitespace"]["f1"],
                                 fixed["prediction_sha256"], trained["prediction_sha256"]])
    scores = {"columns": ["language", "parallel_id", "document_group", "labels_sha256", "fixed_f1", "trained_f1",
                           "fixed_nonwhitespace_f1", "trained_nonwhitespace_f1", "fixed_prediction_sha256", "trained_prediction_sha256"],
              "rows": compact_rows}
    return summary, scores


def source_fingerprints():
    paths = sorted((ROOT / "spiralreality_AIT_onepass_aifcore_integrated").rglob("*.py")) + [Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in paths}


def run(args):
    config = json.loads(Path(args.config).read_text())
    rows, source_receipt = load_pinned_sources(args.sources, args.cache_dir, download=args.download)
    corpus = group_parallel_corpus(rows, seed=config["partition_seed"], folds=config["fold_count"], max_chars=config["max_chars"])
    fold_ids = config["test_folds"]
    if len(set(fold_ids)) != len(fold_ids) or not fold_ids:
        raise ValueError("Each requested fold must appear exactly once")
    partitions = {fold: corpus.partition(fold, max_train_groups=config["max_train_groups"]) for fold in fold_ids}
    contract = {"config": config, "sources": source_receipt, "partition": corpus.receipt,
                "fold_receipts": {str(fold): partitions[fold][1] for fold in fold_ids}, "code_sha256": source_fingerprints()}
    contract_hash = json_hash(contract)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text())["contract_sha256"] != contract_hash:
            raise ValueError("Frozen study differs from current code/data/config; use a new output directory")
    else:
        plan = {"contract_sha256": contract_hash, "contract": contract,
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                            "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
                "git_head_at_prepare": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        save_checkpoint(plan_path, plan)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "contract_sha256": contract_hash, "eligible_rows": len(corpus.records)}), flush=True)
        return
    fold_results = []
    for fold in fold_ids:
        fold_path = output / f"fold_{fold:02d}.json"
        if fold_path.exists():
            finished = json.loads(fold_path.read_text())
            if finished["contract_sha256"] != contract_hash:
                raise ValueError("Completed fold has a different contract")
            fold_results.append(finished)
            continue
        partition, receipt = partitions[fold]
        arms = {}
        # Alternate arm order by fold; no task-specific tuning between arms.
        order = ("fixed", "trained") if fold % 2 == 0 else ("trained", "fixed")
        for arm in order:
            print(json.dumps({"fold": fold, "arm": arm, "stage": "training", "training_rows": len(partition["train"])}), flush=True)
            model = OnePassAIT(seed=config["model_seed"], **config["model"])
            initial = {key: value.copy() for key, value in model.encoder.trainable_parameters().items()}
            cfg = StudentTrainingConfig(**{**config["training"], "train_context_encoder": arm == "trained",
                                          "encoder_lr": config["supervised_encoder_lr"] if arm == "trained" else 0})
            if cfg.validation_split != 0:
                raise ValueError("This protocol uses fixed epochs and separate document development data")
            training = model.train_student([row.text for row in partition["train"]], [row.segments for row in partition["train"]], cfg)
            parameter_change = sum(float(np.sum((parameter - initial[key]) ** 2))
                                   for key, parameter in model.encoder.trainable_parameters().items()) ** 0.5
            if (arm == "fixed" and parameter_change != 0) or (arm == "trained" and parameter_change <= 0):
                raise ValueError("Encoder intervention did not match the frozen protocol")
            checkpoint = output / "checkpoints" / f"fold_{fold:02d}_{arm}.json"
            save_checkpoint(checkpoint, model.state_dict())
            print(json.dumps({"fold": fold, "arm": arm, "stage": "calibration_and_test", "encoder_change_l2": parameter_change}), flush=True)
            temperatures = fit_development_temperatures(model, partition["dev"], config["temperature_candidates"])
            results = evaluate(model, partition["test"], corpus, config, temperatures)
            arms[arm] = {"training_config": asdict(cfg), "encoder_parameter_change_l2": parameter_change,
                         "checkpoint_sha256": sha256(checkpoint.read_bytes()),
                         "training_seconds": training["train_seconds"], "epochs_completed": training["epochs_completed"],
                         "development_temperatures": temperatures, "results": results}
        finished = {"contract_sha256": contract_hash, "fold": fold, "partition": receipt, "arms": arms}
        save_checkpoint(fold_path, finished)
        fold_results.append(finished)
        print(json.dumps({"fold": fold, "stage": "complete"}), flush=True)
    summary, scores = summarize(fold_results, contract)
    summary["fold_receipt_sha256"] = {f"fold_{fold:02d}.json": sha256((output / f"fold_{fold:02d}.json").read_bytes()) for fold in fold_ids}
    summary["training_runs"] = [{"fold": result["fold"], "arm": arm, **{key: value for key, value in payload.items() if key != "results"}}
                                for result in fold_results for arm, payload in result["arms"].items()]
    save_checkpoint(output / "summary.json", summary)
    if args.publish:
        destination = Path(args.publish)
        destination.mkdir(parents=True, exist_ok=True)
        if any((destination / name).exists() for name in ("plan.json", "summary.json", "paired_scores.jsonl")):
            raise FileExistsError("Refusing to overwrite previously published evidence")
        for name, payload in (("plan.json", json.loads(plan_path.read_text())), ("summary.json", summary)):
            target = destination / name
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite published evidence: {target}")
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        target = destination / "paired_scores.jsonl"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite published evidence: {target}")
        with target.open("w") as handle:
            handle.write(json.dumps({"contract_sha256": contract_hash, "columns": scores["columns"]}) + "\n")
            for row in scores["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"complete": True, "contract_sha256": contract_hash, "languages": {
        lang: {"fixed_f1": value["fixed"]["boundary"]["macro_f1"], "trained_f1": value["trained"]["boundary"]["macro_f1"]}
        for lang, value in summary["languages"].items()}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/trilingual-context-pilot.json"))
    parser.add_argument("--sources", default=str(ROOT / "configs/pud-v2.18-sources.json"))
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache/pud"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--publish")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
