#!/usr/bin/env python3
"""Compare identical AIT behavior in isolated, untraced and memory workers.

Raw source/checkpoint inputs and detailed worker receipts stay local. The frozen
plan pins a Git-exported baseline separately from the optimized source tree.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "spiralreality_AIT_onepass_aifcore_integrated"
LANGUAGES = ("en", "ja", "zh")
MODES = ("fixed_channel", "bayesian_regime")
THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
PINNED_ARTIFACTS = {"plan.json", "channel.json", "channel_fit.json", "warm_started_model.json", "warm_start.json"}


def json_tree(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(json_tree(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(path, payload):
    path, payload = Path(path), json_tree(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("Immutable evidence differs: " + path.name)
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def source_hashes(root):
    root = Path(root)
    return {str(path.relative_to(root)): file_hash(path) for path in sorted((root / PACKAGE).rglob("*.py"))}


def supporting_hashes(config):
    names = {"scripts/benchmark_observation_world.py", "scripts/benchmark_active_context.py", config["resource_probe_script"]}
    return {name: file_hash(ROOT / name) for name in sorted(names)}


def select_source(root):
    """Select before package imports; workers never import coordinator helpers."""
    root = Path(root).resolve()
    sys.path[:] = [item for item in sys.path if Path(item or os.getcwd()).resolve() != ROOT]
    sys.path.insert(0, str(root))
    return root


def export_baseline(path, revision):
    path = Path(path)
    resolved = subprocess.run(["git", "rev-parse", revision + "^{commit}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if resolved != revision:
        raise ValueError("Pin a complete baseline commit hash")
    marker = path / ".ait-efficiency-baseline.json"
    if path.exists():
        saved = json.loads(marker.read_text())
        if saved != {"revision": revision, "source_sha256": source_hashes(path)}:
            raise ValueError("Exported baseline source integrity mismatch")
        return saved["source_sha256"]
    payload = subprocess.run(["git", "archive", "--format=tar", revision], cwd=ROOT, check=True, capture_output=True).stdout
    path.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not (member.isfile() or member.isdir()):
                raise ValueError("Baseline archive contains an unsupported path")
        archive.extractall(path)
    hashes = source_hashes(path)
    immutable_json(marker, {"revision": revision, "source_sha256": hashes})
    return hashes


def validate_config(config):
    if config.get("schema_version") != 1 or config.get("modes") != list(MODES):
        raise ValueError("Declare both frozen controller modes")
    if config.get("timing_repeats") != 3 or config.get("memory_repeats") != 1:
        raise ValueError("Freeze three untraced timing and one separate memory repetition")
    revision = config.get("baseline_revision", "")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("Pin a complete baseline revision")
    if set(config.get("source_artifact_sha256", {})) != PINNED_ARTIFACTS:
        raise ValueError("Pin the complete prior-study artifact set")
    if (isinstance(config.get("char_cost"), bool) or not isinstance(config.get("char_cost"), (int, float))
            or not math.isfinite(config["char_cost"]) or config["char_cost"] < 0):
        raise ValueError("Declare a finite nonnegative acquisition cost")


def mode_settings(active, mode, cost):
    if mode not in MODES:
        raise ValueError("Unknown efficiency mode")
    return {**active, "char_cost": cost, "observation_model": mode, "informative_prior": 0.5,
            "learn_observation_regime": True, "model_epistemic_weight": 0.0,
            "phase_gain": 0.0, "surprise_gain": 0.0, "epistemic_weight": 0.05, "learn_dynamics": False}


def jobs(config):
    result = [{"variant": variant, "mode": mode, "measurement": measure, "repeat": repeat}
              for variant in ("baseline", "optimized") for mode in MODES
              for measure, repeats in (("timing", config["timing_repeats"]), ("memory", config["memory_repeats"]))
              for repeat in range(repeats)]
    return sorted(result, key=lambda job: digest([config["order_seed"], job]))


def job_name(job):
    return "_".join(str(job[key]) for key in ("variant", "mode", "measurement", "repeat"))


def rss_bytes(value, system):
    """ru_maxrss is bytes on Darwin and KiB on Linux; never guess elsewhere."""
    if system not in ("Darwin", "Linux"):
        raise ValueError("RSS units are supported only on Darwin and Linux")
    return int(value) * (1 if system == "Darwin" else 1024)


def prepare_inputs(config, source_study, sources, cache):
    select_source(ROOT)
    from scripts.benchmark_observation_world import load_source_study
    from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import group_parallel_corpus, load_pinned_sources
    artifacts, channel = load_source_study(source_study, config)
    historical = artifacts["plan.json"]["contract"]
    old = historical["config"]
    rows, source_receipt = load_pinned_sources(sources, cache, download=False)
    corpus = group_parallel_corpus(rows, seed=old["partition_seed"], folds=old["fold_count"], max_chars=old["max_chars"])
    partition, receipt = corpus.partition(old["test_fold"])
    if source_receipt != historical["sources"] or corpus.receipt != historical["partition"] or receipt != historical["selection_partition"]:
        raise ValueError("Sources or complete partition differ from the pinned study")
    records = [{"language": row.language, "parallel_id": row.parallel_id,
                "document_group": corpus.group_for_document[row.document_id], "text": row.text,
                "segments": list(row.segments), "characters": len(row.text), "source_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
                "labels_sha256": digest(list(row.segments))} for row in partition["test"]]
    public = [{key: value for key, value in row.items() if key not in ("text", "segments")} for row in records]
    return {"state": artifacts["warm_started_model.json"], "channel": channel.export_state(), "rows": records,
            "model_seed": old["model_seed"], "active": old["active"]}, {
                "sources": source_receipt, "partition": receipt, "row_identity_sha256": digest(public),
                "rows": public, "historical_contract_sha256": config["source_contract_sha256"]}


def worker(request_path, result_path, source_root, job):
    """A fresh process executes exactly one condition, with no timing trace mix."""
    import gc
    import resource
    import tracemalloc
    request = json.loads(Path(request_path).read_text())
    if digest(request["inputs"]) != request["input_sha256"]:
        raise ValueError("Worker input integrity mismatch")
    source_root = select_source(source_root)
    expected = request["source_sha256"][job["variant"]]
    if source_hashes(source_root) != expected or dict((name, os.environ.get(name)) for name in THREADS) != THREADS:
        raise ValueError("Worker source or thread environment differs from frozen plan")
    import numpy as np
    from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
    from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import ActiveContextConfig, ActiveTextSession, StringTextSource
    from spiralreality_AIT_onepass_aifcore_integrated.integrated.evaluation_metrics import boundary_counts
    from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT
    actual_runtime = {"python": platform.python_version(), "numpy": np.__version__, "system": platform.system()}
    if any(actual_runtime[key] != request["runtime"][key] for key in actual_runtime):
        raise ValueError("Worker runtime differs from the frozen plan")
    module = sys.modules[OnePassAIT.__module__]
    if not Path(module.__file__).resolve().is_relative_to(source_root):
        raise ValueError("Worker imported the wrong source tree")
    bundle, state = request["inputs"], request["inputs"]["state"]
    for row in bundle["rows"]:
        if (len(row["text"]) != row["characters"] or "".join(row["segments"]) != row["text"]
                or hashlib.sha256(row["text"].encode()).hexdigest() != row["source_sha256"]
                or digest(row["segments"]) != row["labels_sha256"]):
            raise ValueError("Worker source/label identity mismatch")
    encoder = state["encoder"]
    def restore():
        model = OnePassAIT(latent_dim=state["latent_dim"], seed=bundle["model_seed"], encoder_layers=encoder["n_layers"],
                           encoder_heads=encoder["n_heads"], encoder_backend="numpy")
        model.load_state_dict(state)
        return model
    channel = ContextAcquisitionModel.from_state(bundle["channel"]).freeze()
    settings = ActiveContextConfig(**mode_settings(bundle["active"], job["mode"], request["config"]["char_cost"]))
    def episode(model, text):
        return ActiveTextSession(model, channel, settings).run(StringTextSource(text))
    # One same-input warmup per language; no world weights fit in any condition.
    for language in LANGUAGES:
        row = next(row for row in bundle["rows"] if row["language"] == language)
        model = restore()
        episode(model, row["text"])
        del model
    gc.collect()
    system = platform.system()
    floor = rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, system)
    output = []
    for row in bundle["rows"]:
        model = restore()
        gc.collect()
        measured = job["measurement"] == "memory"
        if measured:
            tracemalloc.start()
        started = time.perf_counter()
        result = episode(model, row["text"])
        elapsed = (time.perf_counter() - started) * 1000
        if measured:
            retained, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            retained = peak = None
        if (not result.get("done") or "".join(result["tokens"]) != row["text"]
                or result["metrics"]["world_model_updates"] != 0
                or result["metrics"]["trace_truncated"] != 0
                or result["metrics"]["events"] != len(result["trace"])
                or result["metrics"]["max_window_chars_seen"] > settings.max_window_chars):
            raise ValueError("Worker episode violated completion, source, window or frozen-weight contract")
        behavior = {name: result[name] for name in ("tokens", "metrics", "trace", "decoder", "done")}
        output.append({key: row[key] for key in ("language", "parallel_id", "document_group", "source_sha256", "labels_sha256")})
        output[-1].update(characters=len(row["text"]), prediction_sha256=digest(result["tokens"]),
                          behavior_sha256=digest(behavior), state_sha256=digest(model.state_dict()),
                          boundary=boundary_counts(row["text"], row["segments"], result["tokens"]),
                          nonwhitespace=boundary_counts(row["text"], row["segments"], result["tokens"], exclude_whitespace=True),
                          episode_elapsed_ms=None if measured else elapsed,
                          python_peak_bytes=peak, python_retained_bytes=retained)
        del model, result, behavior
    receipt = {"contract_sha256": request["contract_sha256"], "job": job, "source_sha256": expected,
               "results": output, "results_sha256": digest(output), "runtime": {
                   "python": platform.python_version(), "numpy": np.__version__, "system": system,
                   "thread_environment": THREADS, "process_rss_scope": "whole worker including startup, input/checkpoint parsing, restoration, verification and scoring",
                   "rss_floor_bytes": floor, "process_peak_rss_bytes": rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, system)}}
    if source_hashes(source_root) != expected:
        raise ValueError("Worker implementation changed while evaluating")
    immutable_json(result_path, receipt)


def validate_receipt(receipt, job, identity, contract_hash, expected_source, expected_runtime):
    if (receipt.get("contract_sha256") != contract_hash or receipt.get("job") != job
            or receipt.get("source_sha256") != expected_source
            or receipt.get("results_sha256") != digest(receipt.get("results"))):
        raise ValueError("Worker receipt identity or content hash mismatch")
    results = receipt["results"]
    if not isinstance(results, list) or len(results) != len(identity):
        raise ValueError("Incomplete worker results")
    required = {"characters", "prediction_sha256", "behavior_sha256", "state_sha256", "boundary", "nonwhitespace",
                "episode_elapsed_ms", "python_peak_bytes", "python_retained_bytes"}
    for row, expected in zip(results, identity):
        if any(row.get(key) != value for key, value in expected.items()) or set(row) != set(expected) | required:
            raise ValueError("Worker row identity, order or schema mismatch")
        timing = job["measurement"] == "timing"
        values = [row["episode_elapsed_ms"]] if timing else [row["python_peak_bytes"], row["python_retained_bytes"]]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Invalid worker measurement")
        if (timing and (row["python_peak_bytes"] is not None or row["python_retained_bytes"] is not None)
                or not timing and row["episode_elapsed_ms"] is not None):
            raise ValueError("Traced memory measurements cannot supply speed timing")
        for name in ("boundary", "nonwhitespace"):
            counts = row[name]
            if not isinstance(counts, dict) or set(counts) != {"tp", "fp", "fn", "f1", "positions"}:
                raise ValueError("Missing finite task quality")
            if any(isinstance(counts[key], bool) or not isinstance(counts[key], int) or counts[key] < 0 for key in ("tp", "fp", "fn", "positions")):
                raise ValueError("Invalid boundary counts")
            denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
            if (counts["f1"] != (2 * counts["tp"] / denominator if denominator else 1.0)
                    or counts["tp"] + counts["fp"] + counts["fn"] > counts["positions"]
                    or counts["positions"] > max(0, row["characters"] - 1)):
                raise ValueError("Boundary F1 does not match its counts")
        for field in ("prediction_sha256", "behavior_sha256", "state_sha256"):
            value = row[field]
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("Invalid behavior or state digest")
    runtime = receipt["runtime"]
    if (any(runtime[key] != expected_runtime[key] for key in ("python", "numpy", "system"))
            or runtime["thread_environment"] != THREADS or runtime["process_peak_rss_bytes"] < runtime["rss_floor_bytes"]
            or any(isinstance(runtime[key], bool) or not isinstance(runtime[key], int) or runtime[key] < 0
                   for key in ("rss_floor_bytes", "process_peak_rss_bytes"))):
        raise ValueError("Invalid worker runtime or RSS bounds")


def equality_gate(receipts):
    fields = ("prediction_sha256", "behavior_sha256", "state_sha256", "boundary", "nonwhitespace")
    for mode in MODES:
        selected = [item for item in receipts if item["job"]["mode"] == mode]
        reference = next(item for item in selected if item["job"] == {"variant": "baseline", "mode": mode, "measurement": "timing", "repeat": 0})
        for item in selected:
            for before, after in zip(reference["results"], item["results"]):
                if any(before[field] != after[field] for field in fields):
                    raise ValueError("Exact prediction, plan/trace/metrics or continuation-state equality gate failed")


def summarize(receipts, config):
    result = {}
    for mode in MODES:
        variants = {}
        for variant in ("baseline", "optimized"):
            timing = sorted([r for r in receipts if r["job"]["mode"] == mode and r["job"]["variant"] == variant and r["job"]["measurement"] == "timing"], key=lambda r: r["job"]["repeat"])
            memory = next(r for r in receipts if r["job"] == {"variant": variant, "mode": mode, "measurement": "memory", "repeat": 0})
            languages = {}
            for language in LANGUAGES:
                rows = [row for r in timing for row in r["results"] if row["language"] == language]
                memory_rows = [row for row in memory["results"] if row["language"] == language]
                totals = [math.fsum(row["episode_elapsed_ms"] for row in r["results"] if row["language"] == language) / 1000 for r in timing]
                languages[language] = {"samples_per_repeat": len(memory_rows), "macro_f1": statistics.mean(row["boundary"]["f1"] for row in memory_rows),
                                       "nonwhitespace_macro_f1": statistics.mean(row["nonwhitespace"]["f1"] for row in memory_rows),
                                       "median_episode_ms": statistics.median(row["episode_elapsed_ms"] for row in rows),
                                       "total_seconds_by_repeat": totals, "median_total_seconds": statistics.median(totals),
                                       "median_python_peak_bytes": statistics.median(row["python_peak_bytes"] for row in memory_rows),
                                       "max_python_peak_bytes": max(row["python_peak_bytes"] for row in memory_rows)}
            variants[variant] = {"languages": languages, "untraced_timing_workers": [
                {"repeat": item["job"]["repeat"], **item["runtime"]} for item in timing],
                "separate_traced_memory_worker": memory["runtime"]}
        result[mode] = {"variants": variants, "comparisons": {}}
        for language in LANGUAGES:
            a, b = (variants[variant]["languages"][language] for variant in ("baseline", "optimized"))
            result[mode]["comparisons"][language] = {"speedup_from_median_total": a["median_total_seconds"] / b["median_total_seconds"],
                                                      "python_peak_max_ratio": b["max_python_peak_bytes"] / a["max_python_peak_bytes"],
                                                      "f1_delta": b["macro_f1"] - a["macro_f1"]}
    return result


def execute_worker(request_path, result_path, source_root, baseline_directory, job):
    environment = {**os.environ, **THREADS, "PYTHONHASHSEED": "0"}
    environment.pop("PYTHONPATH", None)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(Path(request_path).resolve()),
               "--worker-result", str(Path(result_path).resolve()), "--worker-source", str(Path(source_root).resolve()),
               "--worker-job", json.dumps(job)]
    subprocess.run(command, env=environment, cwd=baseline_directory, check=True)


def run(args):
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    baseline = Path(args.baseline) if args.baseline else ROOT / config["baseline_directory"]
    sources = {"baseline": export_baseline(baseline, config["baseline_revision"]), "optimized": source_hashes(ROOT)}
    bundle, provenance = prepare_inputs(config, args.source_study or ROOT / config["source_study"], args.sources, args.cache_dir)
    contract = {"config": config, "sources": sources, "provenance": provenance, "worker_sha256": file_hash(__file__),
                "supporting_code_sha256": supporting_hashes(config),
                "evaluation_order": jobs(config), "input_sha256": digest(bundle), "runtime": {"python": platform.python_version(),
                    "numpy": sys.modules["numpy"].__version__, "system": platform.system(), "platform": platform.platform()},
                "thread_environment": THREADS}
    identity = digest(contract)
    output = Path(args.output)
    if output.resolve() == baseline.resolve():
        raise ValueError("Output cannot replace the baseline source export")
    plan = {"contract_sha256": identity, "contract": contract}
    immutable_json(output / "plan.json", plan)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "contract_sha256": identity, "workers": len(jobs(config)), "sentences_per_worker": len(bundle["rows"])}), flush=True)
        return
    request = {"contract_sha256": identity, "config": config, "inputs": bundle, "input_sha256": contract["input_sha256"],
               "source_sha256": sources, "runtime": contract["runtime"]}
    request_path = output / "worker_input.json"
    immutable_json(request_path, request)
    marker = output / "evaluation_start.json"
    if marker.exists():
        if json.loads(marker.read_text())["contract_sha256"] != identity:
            raise ValueError("Evaluation-start identity mismatch")
    else:
        immutable_json(marker, {"contract_sha256": identity, "started_at_utc": datetime.now(timezone.utc).isoformat()})
    receipts, hashes = [], {}
    for job in jobs(config):
        path = output / "workers" / (job_name(job) + ".json")
        if not path.exists():
            execute_worker(request_path, path, baseline if job["variant"] == "baseline" else ROOT, baseline, job)
        receipt = json.loads(path.read_text())
        validate_receipt(receipt, job, provenance["rows"], identity, sources[job["variant"]], contract["runtime"])
        receipts.append(receipt)
        hashes[job_name(job)] = file_hash(path)
        print(json.dumps({"stage": "worker", **job, "samples": len(receipt["results"])}), flush=True)
    equality_gate(receipts)
    if (source_hashes(ROOT) != sources["optimized"] or source_hashes(baseline) != sources["baseline"]
            or file_hash(__file__) != contract["worker_sha256"] or supporting_hashes(config) != contract["supporting_code_sha256"]):
        raise ValueError("Implementation changed during the study")
    if digest(json.loads(request_path.read_text())) != digest(request):
        raise ValueError("Worker input file changed during the study")
    summary = {"contract_sha256": identity, "protocol": config["protocol"], "measurement_protocol": config["measurement_protocol"],
               "exact_behavior_gate": "passed", "episodes": sum(len(r["results"]) for r in receipts), "worker_receipt_sha256": hashes,
               "conditions": summarize(receipts, config)}
    immutable_json(output / "summary.json", summary)
    if args.publish:
        destination = Path(args.publish)
        if destination.exists() and any((destination / name).exists() for name in ("plan.json", "summary.json", "paired_scores.jsonl")):
            raise FileExistsError("Refusing to replace published evidence")
        immutable_json(destination / "plan.json", plan)
        immutable_json(destination / "summary.json", summary)
        references = [r for r in receipts if r["job"]["variant"] == "baseline" and r["job"]["measurement"] == "timing" and r["job"]["repeat"] == 0]
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "paired_scores.jsonl").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"contract_sha256": identity, "description": "Exact outputs, behavior/plan/trace and continuation state match across both variants and all repeats; timing/memory are in summary."}) + "\n")
            for reference in references:
                for row in reference["results"]:
                    handle.write(json.dumps({"mode": reference["job"]["mode"], **{key: value for key, value in row.items() if key not in ("episode_elapsed_ms", "python_peak_bytes", "python_retained_bytes")}}, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "contract_sha256": identity, "episodes": summary["episodes"], "exact_behavior_gate": "passed"}), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default=str(ROOT / "configs/ait-efficiency-pilot.json"))
    result.add_argument("--source-study")
    result.add_argument("--sources", default=str(ROOT / "configs/pud-v2.18-sources.json"))
    result.add_argument("--cache-dir", default=str(ROOT / ".cache/pud"))
    result.add_argument("--baseline")
    result.add_argument("--output")
    result.add_argument("--publish")
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--worker")
    result.add_argument("--worker-result")
    result.add_argument("--worker-source")
    result.add_argument("--worker-job")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if args.worker:
        worker(args.worker, args.worker_result, args.worker_source, json.loads(args.worker_job))
    elif args.output:
        run(args)
    else:
        parser().error("--output is required")
