#!/usr/bin/env python3
"""Frozen native-Ukrainian / Slavic boundary adaptation and retention study."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from spiralreality_AIT_onepass_aifcore_integrated.integrated.evaluation_metrics import (
    aggregate_probability_sums, boundary_counts, boundary_targets, paired_document_interval, probability_sums,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import (
    LANGUAGES, group_parallel_corpus, load_pinned_sources, normalized_text,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig

SLAVIC = ("uk", "pl", "cs")
ARMS = ("slavic_only", "slavic_rehearsal")
ROLES = ("train", "dev", "test")
THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
PACKAGE = "spiralreality_AIT_onepass_aifcore_integrated"


def tree(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(tree(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(path, value):
    path, value = Path(path), tree(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Immutable evidence differs: " + path.name)
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def fingerprints():
    paths = sorted((ROOT / PACKAGE).rglob("*.py")) + [Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported study configuration")
    for name in ("model_seed", "partition_seed", "native_partition_seed", "selection_seed"):
        if isinstance(config[name], bool) or not isinstance(config[name], int):
            raise ValueError("Partition and model seeds must be integers")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]) or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in config["seeds"]):
        raise ValueError("Declare distinct integer shuffle seeds")
    if set(config["caps_per_language"]) != set(ROLES) or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in config["caps_per_language"].values()):
        raise ValueError("Declare positive caps for every role")
    if (isinstance(config["rehearsal_per_language"], bool) or not isinstance(config["rehearsal_per_language"], int)
            or config["rehearsal_per_language"] < 1 or config["max_chars"] < 2):
        raise ValueError("Invalid rehearsal or length budget")
    cfg = config["training"]
    if cfg.get("warm_start") is not True or cfg.get("validation_split") != 0 or cfg.get("phase_lr") != 0:
        raise ValueError("Use genuine warm continuation, fixed epochs and frozen phase")
    if isinstance(cfg["epochs"], bool) or not isinstance(cfg["epochs"], int) or cfg["epochs"] < 1:
        raise ValueError("Declare positive fixed training epochs")
    for key in ("hidden_dim", "emb_dim", "window", "dtype", "context_hidden_dim", "lexical_buckets", "use_encoder_context"):
        if key in cfg:
            raise ValueError("Architecture must be derived from the pinned checkpoint")


def row_record(row, role, group):
    source = getattr(row, "source_id", "original_three_language_PUD")
    upstream = getattr(row, "upstream_split", "test")
    result = {"language": row.language, "source_id": source, "upstream_split": upstream,
              "sentence_id": row.sentence_id, "parallel_id": row.parallel_id,
              "document_id": row.document_id, "document_group": group, "role": role,
              "text": row.text, "segments": list(row.segments), "characters": len(row.text),
              "source_sha256": hashlib.sha256(row.text.encode()).hexdigest(), "labels_sha256": digest(list(row.segments))}
    result["sample_id"] = digest([row.language, source, row.sentence_id, upstream])
    return result


def identity(row):
    return {key: value for key, value in row.items() if key not in ("text", "segments")}


def select_rows(rows, cap, seed, label):
    return sorted(rows, key=lambda row: digest([seed, label, row["sample_id"]]))[:cap]


def guard_slavic_roles(candidates, legacy):
    """Remove cross-role text exposure without collapsing entire source dates."""
    roles, labels = defaultdict(set), defaultdict(set)
    for row in [*candidates, *legacy]:
        roles[normalized_text(row["text"])].add(row["role"])
        labels[row["text"]].add(row["labels_sha256"])
    kept, rejected, seen = [], [], set()
    for row in sorted(candidates, key=lambda item: item["sample_id"]):
        key = normalized_text(row["text"])
        duplicate = (row["role"], row["language"], key)
        reason = ("normalized_duplicate_cross_role_including_legacy" if len(roles[key]) > 1 else
                  "ambiguous_exact_surface_supervision" if len(labels[row["text"]]) > 1 else
                  "normalized_duplicate_within_role_language" if duplicate in seen else None)
        if reason:
            rejected.append({**identity(row), "reason": reason,
                             "normalized_text_sha256": hashlib.sha256(key.encode()).hexdigest()})
        else:
            seen.add(duplicate)
            kept.append(row)
    return kept, rejected


def canonical_acquisition(value):
    # Whether an already pinned file happened to need downloading must not
    # change the experiment identity when the same bytes are loaded on resume.
    if isinstance(value, dict):
        return {key: canonical_acquisition(item) for key, item in value.items() if key != "downloaded"}
    if isinstance(value, list):
        return [canonical_acquisition(item) for item in value]
    return value


def prepare_inputs(config, args):
    from spiralreality_AIT_onepass_aifcore_integrated.integrated.slavic_corpus import load_slavic_sources, partition_ukrainian
    legacy_rows, old_sources = load_pinned_sources(args.legacy_sources, args.legacy_cache, download=args.download)
    corpus = group_parallel_corpus(legacy_rows, seed=config["partition_seed"], folds=config["fold_count"], max_chars=config["legacy_max_chars"])
    original, old_partition = corpus.partition(config["test_fold"])
    documents, parallel_roles = {}, {}
    for row in legacy_rows:
        if row.parallel_id in documents and documents[row.parallel_id] != row.document_id:
            raise ValueError("Original parallel rows disagree about source documents")
        documents[row.parallel_id] = row.document_id
    legacy = []
    for role, rows in original.items():
        for row in rows:
            if row.parallel_id in parallel_roles and parallel_roles[row.parallel_id] != role:
                raise ValueError("Original parallel roles overlap")
            parallel_roles[row.parallel_id] = role
            legacy.append(row_record(row, role, "pud:" + corpus.group_for_document[row.document_id]))
    legacy_guard = []
    for row in legacy_rows:
        group = corpus.group_for_document[row.document_id]
        fold = corpus.fold_for_group.get(group)
        if fold is not None:
            role = "test" if fold == config["test_fold"] else "dev" if fold == (config["test_fold"] + 1) % config["fold_count"] else "train"
            legacy_guard.append(row_record(row, role, "pud:" + group))
    manifest = ROOT / config["slavic_manifest"]
    if file_hash(manifest) != config["slavic_manifest_sha256"]:
        raise ValueError("Slavic source manifest identity mismatch")
    slavic, source_receipt = load_slavic_sources(manifest, args.slavic_cache, pud_document_for_parallel=documents, download=args.download)
    native, native_receipt = partition_ukrainian([row for row in slavic if row.language == "uk"],
        seed=config["native_partition_seed"], folds=config["native_requested_folds"], test_fold=0, max_chars=None)
    candidates, exclusions = [], []
    for role, rows in native.items():
        candidates.extend(row_record(row, role, "native:" + row.document_id) for row in rows)
    for row in slavic:
        if row.language == "uk":
            continue
        role = parallel_roles.get(row.parallel_id)
        if role is None:
            exclusions.append({"language": row.language, "sentence_id": row.sentence_id,
                               "reason": "parallel_not_eligible_in_original_three_language_partition"})
        else:
            candidates.append(row_record(row, role, "pud:" + corpus.group_for_document[row.document_id]))
    candidates, duplicate_exclusions = guard_slavic_roles(candidates, legacy_guard)
    exclusions.extend(duplicate_exclusions)
    eligible = []
    for row in candidates:
        if not 2 <= row["characters"] <= config["max_chars"]:
            exclusions.append({**identity(row), "reason": "length_filter"})
        else:
            eligible.append(row)
    selected = {role: [] for role in ROLES}
    available = {}
    for role in ROLES:
        for language in SLAVIC:
            rows = [row for row in eligible if row["role"] == role and row["language"] == language]
            available[role + "/" + language] = len(rows)
            choice = select_rows(rows, config["caps_per_language"][role], config["selection_seed"], role + "/" + language)
            if not choice:
                raise ValueError("Every Slavic language/role requires real retained examples")
            selected[role].extend(choice)
    rehearsal = []
    for language in LANGUAGES:
        rehearsal.extend(select_rows([row for row in legacy if row["role"] == "train" and row["language"] == language],
                                     config["rehearsal_per_language"], config["selection_seed"], "rehearsal/" + language))
    retention = [row for row in legacy if row["role"] == "test"]
    if any(sum(row["language"] == language for row in retention) != config["expected_retention_per_language"] for language in LANGUAGES):
        raise ValueError("Old-language retention cases differ from the pinned comparison")
    evaluation = [{**row, "evaluation_role": "slavic_" + role} for role in ("dev", "test") for row in selected[role]]
    evaluation += [{**row, "evaluation_role": "retention_test"} for row in retention]
    receipts = {"legacy_sources": canonical_acquisition(old_sources), "legacy_partition": old_partition,
                "legacy_all_source_role_guard_rows": len(legacy_guard),
                "slavic_sources": canonical_acquisition(source_receipt), "native_partition": native_receipt,
                "pre_cap_available": available, "exclusions": exclusions,
                "selected": {role: [identity(row) for row in rows] for role, rows in selected.items()},
                "rehearsal": [identity(row) for row in rehearsal], "retention": [identity(row) for row in retention]}
    return {"slavic_train": selected["train"], "rehearsal": rehearsal, "evaluation": evaluation}, receipts


def load_model(state, seed):
    encoder = state["encoder"]
    model = OnePassAIT(latent_dim=state["latent_dim"], seed=seed, encoder_layers=encoder["n_layers"],
                       encoder_heads=encoder["n_heads"], encoder_backend="numpy")
    model.load_state_dict(state)
    return model


def parameter_state(model):
    head = model.student._capture_state()
    names = ("embeddings", "W_window", "b_window", "W_out", "b_out", "gate_w", "gate_b", "transitions",
             "lexical", "ctx_W1", "ctx_b1", "ctx_w", "ctx_b")
    return {"head": {name: head[name] for name in names if name in head},
            "encoder": model.encoder.export_state(), "phase": model.phase.export_state().basis}


def training_settings(state, config):
    student = state["student"]
    architecture = student["architecture"]
    return {**config["training"], **{name: architecture[name] for name in ("hidden_dim", "emb_dim", "window", "dtype")},
            "lexical_buckets": student["lexical"]["buckets"], "use_encoder_context": student["use_encoder_context"],
            "context_hidden_dim": student["context_hidden_dim"]}


def evaluate_rows(model, rows):
    before = digest(parameter_state(model))
    output = []
    for row in rows:
        started = time.perf_counter()
        observed = model.student.observe_sequence(row["text"])
        elapsed = (time.perf_counter() - started) * 1000
        tokens, probabilities = observed["tokens"], list(map(float, observed["boundary_probabilities"]))
        if "".join(tokens) != row["text"] or len(probabilities) != max(0, row["characters"] - 1):
            raise ValueError("Prediction failed exact source reconstruction")
        targets = boundary_targets(row["text"], row["segments"])
        output.append({**identity(row), "prediction_sha256": digest(tokens), "probability_sha256": digest(probabilities),
                       "boundary": boundary_counts(row["text"], row["segments"], tokens),
                       "nonwhitespace": boundary_counts(row["text"], row["segments"], tokens, exclude_whitespace=True),
                       "calibration": probability_sums(probabilities, targets), "inference_ms": elapsed})
    if digest(parameter_state(model)) != before:
        raise ValueError("Evaluation altered model parameters")
    return output


def aggregate(rows):
    result = {"samples": len(rows), "documents": len({row["document_group"] for row in rows}), "characters": sum(row["characters"] for row in rows)}
    for name in ("boundary", "nonwhitespace"):
        counts = {key: sum(row[name][key] for row in rows) for key in ("tp", "fp", "fn", "positions")}
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        result[name] = {**counts, "micro_f1": 2 * counts["tp"] / denominator if denominator else 1.0,
                        "macro_f1": statistics.mean(row[name]["f1"] for row in rows)}
    result["calibration"] = aggregate_probability_sums([row["calibration"] for row in rows])
    result["median_inference_ms"] = statistics.median(row["inference_ms"] for row in rows)
    return result


def validate_results(results, expected):
    if not isinstance(results, list) or len(results) != len(expected):
        raise ValueError("Incomplete evaluation results")
    for result, row in zip(results, expected):
        if any(result.get(key) != value for key, value in identity(row).items()):
            raise ValueError("Evaluation row identity/order mismatch")
        if isinstance(result["inference_ms"], bool) or not math.isfinite(result["inference_ms"]) or result["inference_ms"] < 0:
            raise ValueError("Invalid inference timing")
        for name in ("boundary", "nonwhitespace"):
            value = result[name]
            if set(value) != {"tp", "fp", "fn", "positions", "f1"} or any(
                    isinstance(value[k], bool) or not isinstance(value[k], int) or value[k] < 0 for k in ("tp", "fp", "fn", "positions")):
                raise ValueError("Invalid boundary counts")
            denominator = 2 * value["tp"] + value["fp"] + value["fn"]
            if (value["f1"] != (2 * value["tp"] / denominator if denominator else 1.0)
                    or value["positions"] > row["characters"] - 1
                    or value["tp"] + value["fp"] + value["fn"] > value["positions"]):
                raise ValueError("Boundary score/count mismatch")
        calibration = result["calibration"]
        if calibration["positions"] != row["characters"] - 1 or any(
                not math.isfinite(calibration[k]) or calibration[k] < 0 for k in ("squared_error_sum", "negative_log_likelihood_sum")):
            raise ValueError("Invalid calibration measurement")
        bins = ("bin_counts", "bin_probability_sums", "bin_label_sums")
        if any(len(calibration[key]) != 10 for key in bins) or sum(calibration["bin_counts"]) != calibration["positions"]:
            raise ValueError("Incomplete calibration bins")
        for count, probability, label in zip(*(calibration[key] for key in bins)):
            if (isinstance(count, bool) or not isinstance(count, int) or count < 0
                    or any(not math.isfinite(value) or not 0 <= value <= count + 1e-8 for value in (probability, label))):
                raise ValueError("Invalid calibration-bin totals")
        for name in ("prediction_sha256", "probability_sha256"):
            value = result[name]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid prediction digest")


def worker(task_path):
    task = json.loads(Path(task_path).read_text())
    if not Path(sys.modules[OnePassAIT.__module__].__file__).resolve().is_relative_to(ROOT):
        raise ValueError("Worker imported a different package source")
    if task["code_sha256"] != fingerprints() or any(os.environ.get(key) != value for key, value in THREADS.items()):
        raise ValueError("Training source or thread environment changed")
    if task["arm"] not in ARMS or any(task["runtime"][key] != value for key, value in
            {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}.items()):
        raise ValueError("Training arm or runtime differs from the frozen task")
    source = Path(task["checkpoint"])
    if file_hash(source) != task["checkpoint_sha256"]:
        raise ValueError("Starting checkpoint integrity mismatch")
    payload = json.loads(Path(task["input"]).read_text())
    if digest(payload) != task["input_sha256"]:
        raise ValueError("Training input integrity mismatch")
    state = json.loads(source.read_text())
    model = load_model(state, task["seed"])
    initial = parameter_state(model)
    if digest(initial) != task["initial_parameter_sha256"]:
        raise ValueError("Starting parameters differ from the common pretrained checkpoint")
    examples = payload["slavic_train"] + (payload["rehearsal"] if task["arm"] == "slavic_rehearsal" else [])
    cfg = StudentTrainingConfig(**task["training"])
    model.student.rng.seed(task["seed"])
    first = examples[0]
    model.train_student([first["text"]], [first["segments"]], replace(cfg, epochs=0))
    if parameter_state(model) != initial:
        raise ValueError("Warm-start zero-epoch check reset pretrained parameters")
    model.student.rng.seed(task["seed"])
    started = time.perf_counter()
    summary = model.train_student([row["text"] for row in examples], [row["segments"] for row in examples], cfg)
    elapsed = time.perf_counter() - started
    final = parameter_state(model)
    if (summary["epochs_completed"] != cfg.epochs or final["head"] == initial["head"]
            or cfg.train_context_encoder and final["encoder"] == initial["encoder"]
            or final["phase"] != initial["phase"]):
        raise ValueError("Training did not match fixed epochs, actual updates or frozen phase")
    destination = Path(task["trained_checkpoint"])
    immutable_json(destination, model.state_dict())
    results = tree(evaluate_rows(model, payload["evaluation"]))
    validate_results(results, payload["evaluation"])
    if fingerprints() != task["code_sha256"]:
        raise ValueError("Implementation changed during training")
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss) * (1 if platform.system() == "Darwin" else 1024)
    receipt = {key: task[key] for key in ("contract_sha256", "run_id", "arm", "seed", "input_sha256", "training")}
    receipt.update(initial_parameter_sha256=digest(initial), zero_epoch_parameter_sha256=digest(initial),
                   final_parameter_sha256=digest(final), checkpoint_sha256=file_hash(destination), checkpoint_bytes=destination.stat().st_size,
                   initial_component_sha256={key: digest(value) for key, value in initial.items()},
                   final_component_sha256={key: digest(value) for key, value in final.items()},
                   training_wall_seconds=elapsed, training_examples=len(examples),
                   training_examples_per_language=dict(Counter(row["language"] for row in examples)),
                   processed_examples=len(examples) * cfg.epochs,
                   history=summary["history"], results=results, results_sha256=digest(results),
                   whole_worker_peak_rss_bytes=rss_bytes,
                   runtime={"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                            "thread_environment": THREADS})
    immutable_json(task["receipt"], receipt)
    print(json.dumps({"stage": "trained", "run_id": task["run_id"], "examples": len(examples), "epochs": cfg.epochs}), flush=True)


def run_id(arm, seed):
    return f"{arm}_seed_{seed}"


def execute_worker(task_path):
    environment = {**os.environ, **THREADS, "PYTHONHASHSEED": "0"}
    environment.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(Path(task_path).resolve())],
                   cwd=ROOT, env=environment, check=True)


def validate_training(receipt, task, payload, original):
    required = {"contract_sha256", "run_id", "arm", "seed", "input_sha256", "training", "initial_parameter_sha256",
                "zero_epoch_parameter_sha256", "final_parameter_sha256", "initial_component_sha256", "final_component_sha256",
                "checkpoint_sha256", "checkpoint_bytes", "training_wall_seconds", "training_examples",
                "training_examples_per_language", "processed_examples", "history", "results", "results_sha256",
                "whole_worker_peak_rss_bytes", "runtime"}
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ValueError("Incomplete training receipt schema")
    if (any(receipt[key] != task[key] for key in ("contract_sha256", "run_id", "arm", "seed", "input_sha256", "training"))
            or receipt["checkpoint_sha256"] != file_hash(task["trained_checkpoint"])
            or receipt["checkpoint_bytes"] != Path(task["trained_checkpoint"]).stat().st_size
            or receipt["results_sha256"] != digest(receipt["results"])
            or receipt["initial_parameter_sha256"] != task["initial_parameter_sha256"]
            or receipt["zero_epoch_parameter_sha256"] != task["initial_parameter_sha256"]):
        raise ValueError("Completed training receipt identity or content mismatch")
    examples = payload["slavic_train"] + (payload["rehearsal"] if task["arm"] == "slavic_rehearsal" else [])
    if (receipt["training_examples"] != len(examples) or receipt["processed_examples"] != len(examples) * task["training"]["epochs"]
            or receipt["training_examples_per_language"] != dict(Counter(row["language"] for row in examples))
            or len(receipt["history"]) != task["training"]["epochs"]):
        raise ValueError("Training exposure or fixed-epoch receipt mismatch")
    for index, metrics in enumerate(receipt["history"], 1):
        if metrics.get("epoch") != index or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                               or not math.isfinite(value) for value in metrics.values()):
            raise ValueError("Invalid training history")
    for name in ("training_wall_seconds", "whole_worker_peak_rss_bytes"):
        if isinstance(receipt[name], bool) or not isinstance(receipt[name], (int, float)) or not math.isfinite(receipt[name]) or receipt[name] < 0:
            raise ValueError("Invalid training resource measurement")
    if receipt["runtime"] != {**task["runtime"], "thread_environment": THREADS}:
        raise ValueError("Training runtime mismatch")
    restored = load_model(json.loads(Path(task["trained_checkpoint"]).read_text()), task["seed"])
    final = parameter_state(restored)
    initial_components = {key: digest(value) for key, value in original.items()}
    final_components = {key: digest(value) for key, value in final.items()}
    if (receipt["initial_component_sha256"] != initial_components or receipt["final_component_sha256"] != final_components
            or receipt["final_parameter_sha256"] != digest(final) or final["head"] == original["head"]
            or task["training"]["train_context_encoder"] and final["encoder"] == original["encoder"]
            or final["phase"] != original["phase"]):
        raise ValueError("Saved checkpoint does not preserve the declared learning intervention")
    validate_results(receipt["results"], payload["evaluation"])


def summarize(evaluations, training, config, contract):
    summary = {"contract_sha256": contract, "protocol": config["protocol"], "measurement_protocol": config["measurement_protocol"],
               "conditions": {}, "paired": {}, "training": training}
    for condition, rows in evaluations.items():
        summary["conditions"][condition] = {role: {language: aggregate([row for row in rows if row["evaluation_role"] == role and row["language"] == language])
                                          for language in (*SLAVIC, *LANGUAGES) if any(row["evaluation_role"] == role and row["language"] == language for row in rows)}
                                          for role in ("slavic_dev", "slavic_test", "retention_test")}
    reference = {row["sample_id"]: row for row in evaluations["pre_adaptation"]}
    for arm in ARMS:
        summary["paired"][arm] = {}
        for role in ("slavic_dev", "slavic_test", "retention_test"):
            for language in (*SLAVIC, *LANGUAGES):
                selected = [row for row in evaluations["pre_adaptation"] if row["evaluation_role"] == role and row["language"] == language]
                if not selected:
                    continue
                differences = []
                per_seed = []
                for seed in config["seeds"]:
                    lookup = {row["sample_id"]: row for row in evaluations[run_id(arm, seed)]}
                    differences.extend((row["document_group"], lookup[row["sample_id"]]["boundary"]["f1"] - row["boundary"]["f1"]) for row in selected)
                    per_seed.append({"seed": seed, "macro_f1": statistics.mean(lookup[row["sample_id"]]["boundary"]["f1"] for row in selected)})
                interval = paired_document_interval(differences, **config["bootstrap"])
                interval["scope"] = "Conditional on these source documents/dates, declared shuffle seeds and one pretrained checkpoint; few Ukrainian test dates and initialization uncertainty are not resolved."
                summary["paired"][arm][role + "/" + language] = {**interval, "per_seed": per_seed}
    return summary


def run(args):
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    checkpoint = Path(args.checkpoint) if args.checkpoint else ROOT / config["checkpoint"]
    if file_hash(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Pinned starting checkpoint integrity mismatch")
    state = json.loads(checkpoint.read_text())
    payload, provenance = prepare_inputs(config, args)
    code = fingerprints()
    initial_parameters = parameter_state(load_model(state, config["model_seed"]))
    initial_hash = digest(initial_parameters)
    contract = {"config": config, "provenance": provenance, "code_sha256": code,
                "checkpoint_sha256": config["checkpoint_sha256"], "input_sha256": digest(payload),
                "initial_parameter_sha256": initial_hash, "derived_training": training_settings(state, config),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
                "training_jobs": [{"arm": arm, "seed": seed} for arm in ARMS for seed in config["seeds"]]}
    contract_hash = digest(contract)
    output = Path(args.output)
    plan = {"contract_sha256": contract_hash, "contract": contract}
    immutable_json(output / "plan.json", plan)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "contract_sha256": contract_hash,
                          "training": len(payload["slavic_train"]), "rehearsal": len(payload["rehearsal"]),
                          "evaluation": len(payload["evaluation"]), "training_jobs": len(contract["training_jobs"])}), flush=True)
        return
    input_path = output / "input.json"
    immutable_json(input_path, payload)
    evaluations, training, hashes = {}, {}, {}
    for job in contract["training_jobs"]:
        name = run_id(job["arm"], job["seed"])
        receipt_path, checkpoint_path = output / "training" / (name + ".json"), output / "checkpoints" / (name + ".json")
        task = {**job, "run_id": name, "contract_sha256": contract_hash, "code_sha256": code,
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": config["checkpoint_sha256"],
                "input": str(input_path.resolve()), "input_sha256": contract["input_sha256"],
                "initial_parameter_sha256": initial_hash, "training": contract["derived_training"],
                "runtime": {key: contract["runtime"][key] for key in ("python", "numpy", "platform")},
                "trained_checkpoint": str(checkpoint_path.resolve()), "receipt": str(receipt_path.resolve())}
        task_path = output / "tasks" / (name + ".json")
        immutable_json(task_path, task)
        if not receipt_path.exists():
            attempt = output / "attempts" / (name + ".json")
            if attempt.exists():
                raise ValueError("Incomplete training attempt retained; do not silently retry it: " + name)
            immutable_json(attempt, {"contract_sha256": contract_hash, "run_id": name,
                                     "started_at_utc": datetime.now(timezone.utc).isoformat()})
            execute_worker(task_path)
        receipt = json.loads(receipt_path.read_text())
        validate_training(receipt, task, payload, initial_parameters)
        evaluations[name] = receipt["results"]
        training[name] = {key: value for key, value in receipt.items() if key not in ("results", "results_sha256")}
        hashes[name] = file_hash(receipt_path)
        print(json.dumps({"stage": "completed", "run_id": name, "training_seconds": receipt["training_wall_seconds"]}), flush=True)
    baseline_path = output / "pre_adaptation.json"
    if not baseline_path.exists():
        result = tree(evaluate_rows(load_model(state, config["model_seed"]), payload["evaluation"]))
        immutable_json(baseline_path, {"contract_sha256": contract_hash, "results": result, "results_sha256": digest(result)})
    baseline = json.loads(baseline_path.read_text())
    if baseline["contract_sha256"] != contract_hash or baseline["results_sha256"] != digest(baseline["results"]):
        raise ValueError("Pre-adaptation receipt integrity mismatch")
    validate_results(baseline["results"], payload["evaluation"])
    evaluations["pre_adaptation"] = baseline["results"]
    if fingerprints() != code:
        raise ValueError("Implementation changed during the frozen study")
    summary = summarize(evaluations, training, config, contract_hash)
    summary.update(training_receipt_sha256=hashes, pre_adaptation_receipt_sha256=file_hash(baseline_path),
                   evaluated_cases=len(payload["evaluation"]), total_evaluations=sum(map(len, evaluations.values())))
    immutable_json(output / "summary.json", summary)
    if args.publish:
        public = Path(args.publish)
        if any((public / name).exists() for name in ("plan.json", "summary.json", "paired_scores.jsonl")):
            raise FileExistsError("Refusing to replace published evidence")
        immutable_json(public / "plan.json", plan)
        immutable_json(public / "summary.json", summary)
        with (public / "paired_scores.jsonl").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"contract_sha256": contract_hash, "conditions": list(evaluations),
                                     "description": "Matched case identities and boundary/calibration scores; raw corpus and predictions remain local."}) + "\n")
            for index, row in enumerate(payload["evaluation"]):
                compact = {**identity(row), "conditions": {name: {key: value[key] for key in
                           ("prediction_sha256", "probability_sha256", "boundary", "nonwhitespace", "calibration", "inference_ms")}
                           for name, rows in evaluations.items() for value in [rows[index]]}}
                handle.write(json.dumps(compact, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "contract_sha256": contract_hash, "training_runs": len(training),
                      "total_evaluations": summary["total_evaluations"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/slavic-learning-pilot.json"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--legacy-sources", default=str(ROOT / "configs/pud-v2.18-sources.json"))
    parser.add_argument("--legacy-cache", default=str(ROOT / ".cache/pud"))
    parser.add_argument("--slavic-cache", default=str(ROOT / ".cache/slavic-ud"))
    parser.add_argument("--output")
    parser.add_argument("--publish")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.output:
        run(args)
    else:
        parser.error("--output is required")
