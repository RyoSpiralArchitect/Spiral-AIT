#!/usr/bin/env python3
"""Frozen learning-budget and AIF repair comparison; corpus text stays local."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.benchmark_trilingual import aggregate_counts, fit_development_temperatures
from spiralreality_AIT_onepass_aifcore_integrated.integrated.checkpoint import save_checkpoint
from spiralreality_AIT_onepass_aifcore_integrated.integrated.evaluation_metrics import boundary_counts, paired_document_interval
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import LANGUAGES, group_parallel_corpus, json_hash, load_pinned_sources, sha256
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, SegmentationAIFConfig

PACKAGE = "spiralreality_AIT_onepass_aifcore_integrated"
POLICIES = ("direct", "legacy_efe", "posterior_risk", "calibrated_risk")
ARMS = ("legacy_fixed", "repaired_fixed", "repaired_trained")


def fingerprints():
    paths = sorted((ROOT / PACKAGE).rglob("*.py"))
    paths += [ROOT / "scripts" / name for name in
              ("benchmark_policy_learning.py", "context_training_worker.py", "benchmark_trilingual.py")]
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in paths}


def reference_snapshot(commit, destination):
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("The reference implementation must be an immutable commit")
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "--", PACKAGE], cwd=ROOT, text=True).splitlines()
    if not paths:
        raise ValueError("Reference implementation is missing")
    result = {}
    for relative in paths:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Invalid reference path")
        payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        target = destination / relative
        if target.exists() and target.read_bytes() != payload:
            raise ValueError("Refusing to repair a modified reference snapshot")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(payload)
        result[relative] = sha256(payload)
    return result


def completed_training(output, run_id, contract_hash):
    receipt_path = output / "training" / f"{run_id}.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt["run_id"] != run_id or receipt["contract_sha256"] != contract_hash:
        raise ValueError("Training receipt identity mismatch")
    checkpoint = output / "checkpoints" / f"{run_id}.json"
    if sha256(checkpoint.read_bytes()) != receipt["checkpoint_sha256"]:
        raise ValueError("Completed checkpoint integrity mismatch")
    return receipt


def load_model(output, receipt):
    model = OnePassAIT(seed=receipt["seed"], **receipt["model_config"])
    model.load_state_dict(json.loads((output / "checkpoints" / f"{receipt['run_id']}.json").read_text()))
    probes = []
    for text in ("Hello world.", "日本語です。", "中文测试。"):
        result = model.segment_text(text, include_confidence=True)
        probes.append([result["tokens"], result["boundary_probabilities"]])
    if json_hash(probes) != receipt["inference_probe_sha256"]:
        raise ValueError("Restored inference differs from the training implementation")
    return model


def development_score(model, rows):
    by_language = defaultdict(list)
    for row in rows:
        by_language[row.language].append(boundary_counts(row.text, row.segments, model.segment_text(row.text))["f1"])
    if set(by_language) != set(LANGUAGES):
        raise ValueError("Development selection requires all three languages")
    scores = {language: float(np.mean(values)) for language, values in by_language.items()}
    return {"macro_f1": float(np.mean(list(scores.values()))), "per_language_f1": scores}


def choose_on_development(candidates, fixed_score):
    if not candidates or any(not np.isfinite(row["development"]["macro_f1"]) for row in candidates):
        raise ValueError("Finite development scores are required")
    chosen = min(candidates, key=lambda row: (-row["development"]["macro_f1"], row["encoder_lr"]))
    return {"run_id": chosen["run_id"], "encoder_lr": chosen["encoder_lr"],
            "development_macro_f1": chosen["development"]["macro_f1"],
            "fixed_development_macro_f1": fixed_score,
            "development_prefers_context": chosen["development"]["macro_f1"] > fixed_score + 1e-12}


def evaluate_model(model, rows, corpus, temperatures):
    results = []
    for row in rows:
        logits = model.student._python_logits(row.text)
        modes = {"direct": (model.student.decode_with_logit_bias(row.text, 0, logits=logits)["tokens"], None)}
        for mode in POLICIES[1:]:
            cfg = SegmentationAIFConfig(selection_mode="legacy_efe" if mode == "legacy_efe" else "posterior_risk",
                                        marginal_temperature=temperatures[row.language] if mode == "calibrated_risk" else 1.0)
            selection = model._select_policy_from_logits(row.text, logits, cfg)
            candidate = next(item for item in selection["candidates"] if item["policy"] == selection["chosen_policy"])
            tokens = model.student.decode_with_logit_bias(row.text, candidate["logit_bias"], logits=logits)["tokens"]
            modes[mode] = (tokens, {"policy": selection["chosen_policy"], "risk": candidate["risk"],
                                   "epistemic": candidate["epistemic"], "total": candidate["total"],
                                   "baseline_risk": selection.get("baseline_risk")})
        result = {"language": row.language, "parallel_id": row.parallel_id,
                  "document_group": corpus.group_for_document[row.document_id], "labels_sha256": json_hash(row.segments)}
        for mode, (tokens, diagnostic) in modes.items():
            result[mode] = {"boundary": boundary_counts(row.text, row.segments, tokens),
                            "nonwhitespace": boundary_counts(row.text, row.segments, tokens, exclude_whitespace=True),
                            "prediction_sha256": json_hash(tokens), "selection": diagnostic}
        results.append(result)
    return results


def summarize(evaluations, config, contract_hash, selections):
    summary = {"contract_sha256": contract_hash, "protocol": config["protocol"],
               "seeds": config["seeds"], "development_selections": selections, "budgets": {}}
    document_rows = []
    for budget in dict.fromkeys(run["budget"] for run in evaluations):
        summary["budgets"][budget] = {}
        budget_runs = [run for run in evaluations if run["budget"] == budget]
        for language in LANGUAGES:
            language_summary, lookups = {}, {}
            for arm in ARMS:
                runs = [run for run in budget_runs if run["arm"] == arm]
                per_seed = []
                for run in runs:
                    rows = [row for row in run["results"] if row["language"] == language]
                    lookups[(run["seed"], arm)] = {row["parallel_id"]: row for row in rows}
                    per_seed.append({"seed": run["seed"], "run_id": run["run_id"], "samples": len(rows),
                                     **{mode: aggregate_counts([row[mode]["boundary"] for row in rows])["macro_f1"] for mode in POLICIES},
                                     "nonwhitespace": aggregate_counts([row["direct"]["nonwhitespace"] for row in rows])["macro_f1"]})
                    grouped = defaultdict(list)
                    for row in rows:
                        grouped[row["document_group"]].append(row)
                    for doc, items in grouped.items():
                        document_rows.append([budget, run["seed"], arm, language, doc, len(items),
                                              *[sum(item[mode]["boundary"]["f1"] for item in items) for mode in POLICIES],
                                              sum(item["direct"]["nonwhitespace"]["f1"] for item in items)])
                all_rows = [row for run in runs for row in run["results"] if row["language"] == language]
                language_summary[arm] = {"unique_test_sentences": per_seed[0]["samples"], "per_seed": per_seed}
                for mode in (*POLICIES, "nonwhitespace"):
                    values = [row[mode] for row in per_seed]
                    language_summary[arm][mode] = {"mean_macro_f1": float(np.mean(values)),
                                                  "seed_min": min(values), "seed_max": max(values)}
                language_summary[arm]["policy_histograms"] = {
                    mode: dict(Counter(row[mode]["selection"]["policy"] for row in all_rows)) for mode in POLICIES[1:]}
            pairs = (("repaired_fixed", "legacy_fixed"), ("repaired_trained", "repaired_fixed"))
            language_summary["paired"] = {}
            for treatment, reference in pairs:
                differences = []
                for seed in config["seeds"]:
                    left, right = lookups[(seed, treatment)], lookups[(seed, reference)]
                    if left.keys() != right.keys():
                        raise ValueError("Unmatched evaluation rows")
                    differences.extend((row["document_group"], left[key]["direct"]["boundary"]["f1"] - row["direct"]["boundary"]["f1"])
                                       for key, row in right.items())
                interval = paired_document_interval(differences)
                interval["scope"] = "document uncertainty conditional on these three fitted seeds and this one previously inspected PUD split; not population training-seed uncertainty"
                language_summary["paired"][treatment + "_minus_" + reference] = interval
            summary["budgets"][budget][language] = language_summary
    return summary, document_rows


def run(args):
    config = json.loads(Path(args.config).read_text())
    rates = config["encoder_learning_rates"]
    if not rates or len(set(rates)) != len(rates) or any(not np.isfinite(value) or value <= 0 for value in rates):
        raise ValueError("Declare unique positive encoder learning rates")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Declare distinct initialization seeds")
    if config["training"]["validation_split"] != 0 or config["training"]["phase_lr"] != 0:
        raise ValueError("Use fixed epochs, separate document development data, and frozen phase")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, sources = load_pinned_sources(args.sources, args.cache_dir, download=args.download)
    corpus = group_parallel_corpus(rows, seed=config["partition_seed"], folds=config["fold_count"], max_chars=config["max_chars"])
    partitions = {}
    for cap in config["training_document_caps"]:
        key = "all" if cap is None else str(cap)
        if key in partitions:
            raise ValueError("Training budgets must be unique")
        partitions[key] = corpus.partition(config["test_fold"], max_train_groups=cap)
    reference = output / "reference"
    reference_files = reference_snapshot(config["reference_commit"], reference)
    current_files = fingerprints()
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}
    contract = {"config": config, "sources": sources, "partition": corpus.receipt,
                "budget_partitions": {key: value[1] for key, value in partitions.items()},
                "code_sha256": current_files, "reference_files_sha256": reference_files, "runtime": runtime,
                "worker_threads": {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    contract_hash = json_hash(contract)
    plan = {"contract_sha256": contract_hash, "contract": contract}
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen study differs from current code/data/config/runtime; use a new output directory")
    if not plan_path.exists():
        save_checkpoint(plan_path, plan)
    tasks = []
    for budget, (partition, receipt) in partitions.items():
        training_input = output / "inputs" / f"budget_{budget}.json"
        payload = json.dumps([asdict(row) for row in partition["train"]], ensure_ascii=False).encode()
        training_input.parent.mkdir(parents=True, exist_ok=True)
        if training_input.exists() and training_input.read_bytes() != payload:
            raise ValueError("Frozen training input differs")
        if not training_input.exists():
            training_input.write_bytes(payload)
        candidates = [("legacy_fixed", False, 0), ("repaired_fixed", False, 0)]
        candidates += [(f"context_lr_{rate:g}", True, rate) for rate in rates]
        for seed in config["seeds"]:
            for name, trainable, rate in candidates:
                run_id = f"budget_{budget}_seed_{seed}_{name}"
                legacy = name == "legacy_fixed"
                tasks.append({"run_id": run_id, "budget": budget, "seed": seed, "arm": name,
                              "contract_sha256": contract_hash, "model": config["model"],
                              "implementation": config["reference_commit"] if legacy else "current_source_hashes",
                              "implementation_root": str(reference if legacy else ROOT),
                              "implementation_files": reference_files if legacy else {key: value for key, value in current_files.items() if key.startswith(PACKAGE + "/")},
                              "training_input": str(training_input), "training_input_sha256": sha256(payload),
                              "training": {**config["training"], "train_context_encoder": trainable, "encoder_lr": rate},
                              "checkpoint": str(output / "checkpoints" / f"{run_id}.json"),
                              "receipt": str(output / "training" / f"{run_id}.json")})
    if args.prepare_only:
        print(json.dumps({"prepared": True, "contract_sha256": contract_hash, "training_runs": len(tasks),
                          "train_per_language": {key: receipt["train"]["per_language"] for key, (_, receipt) in partitions.items()},
                          "test_per_language": next(iter(partitions.values()))[1]["test"]["per_language"]}), flush=True)
        return

    def train(task):
        done = completed_training(output, task["run_id"], contract_hash)
        if done is not None:
            return done
        path = output / "tasks" / f"{task['run_id']}.json"
        save_checkpoint(path, task)
        log = output / "tasks" / f"{task['run_id']}.log"
        with log.open("w") as handle:
            process = subprocess.run([sys.executable, str(ROOT / "scripts/context_training_worker.py"), str(path)],
                                     cwd=output, env={**os.environ, **contract["worker_threads"]}, stdout=handle, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f"Training failed for {task['run_id']}; inspect its local task log")
        return completed_training(output, task["run_id"], contract_hash)

    receipts = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(train, task): task for task in tasks}
        for future in as_completed(futures):
            receipt = future.result()
            receipts[receipt["run_id"]] = receipt
            print(json.dumps({"stage": "training", "completed": len(receipts), "total": len(tasks), "run_id": receipt["run_id"]}), flush=True)
    # Finish every development choice and persist it before evaluating test rows.
    development, selections = {}, []
    for task in tasks:
        run_id = task["run_id"]
        model = load_model(output, receipts[run_id])
        development[run_id] = development_score(model, partitions[task["budget"]][0]["dev"])
    chosen_tasks = []
    for budget in partitions:
        for seed in config["seeds"]:
            group = [task for task in tasks if task["budget"] == budget and task["seed"] == seed]
            fixed = next(task for task in group if task["arm"] == "repaired_fixed")
            candidates = [{"run_id": task["run_id"], "encoder_lr": task["training"]["encoder_lr"], "development": development[task["run_id"]]}
                          for task in group if task["training"]["train_context_encoder"]]
            selected = {"budget": budget, "seed": seed, **choose_on_development(candidates, development[fixed["run_id"]]["macro_f1"])}
            selections.append(selected)
            for task in group:
                if task["arm"] in ("legacy_fixed", "repaired_fixed") or task["run_id"] == selected["run_id"]:
                    chosen_tasks.append({**task, "arm": "repaired_trained" if task["training"]["train_context_encoder"] else task["arm"]})
    selection_receipt = {"contract_sha256": contract_hash, "development": development, "selections": selections}
    selection_path = output / "development_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection_receipt:
        raise ValueError("Development selection differs from the saved decision")
    if not selection_path.exists():
        save_checkpoint(selection_path, selection_receipt)
    evaluations = []
    for task in chosen_tasks:
        result_path = output / "evaluation" / f"{task['run_id']}.json"
        model = load_model(output, receipts[task["run_id"]])
        partition = partitions[task["budget"]][0]
        temperatures = fit_development_temperatures(model, partition["dev"], config["temperature_candidates"])
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (result["contract_sha256"] != contract_hash
                    or result["checkpoint_sha256"] != receipts[task["run_id"]]["checkpoint_sha256"]
                    or any(result[key] != task[key] for key in ("run_id", "budget", "seed", "arm"))
                    or result["temperatures"] != temperatures):
                raise ValueError("Evaluation receipt identity mismatch")
        else:
            result = {"contract_sha256": contract_hash, "budget": task["budget"], "seed": task["seed"], "arm": task["arm"],
                      "run_id": task["run_id"], "checkpoint_sha256": receipts[task["run_id"]]["checkpoint_sha256"],
                      "temperatures": temperatures, "results": evaluate_model(model, partition["test"], corpus, temperatures)}
            save_checkpoint(result_path, result)
        evaluations.append(result)
        print(json.dumps({"stage": "test", "completed": len(evaluations), "total": len(chosen_tasks), "run_id": task["run_id"]}), flush=True)
    if fingerprints() != current_files:
        raise ValueError("Implementation changed during the study")
    summary, document_rows = summarize(evaluations, config, contract_hash, selections)
    summary["training_runs"] = [receipts[task["run_id"]] for task in tasks]
    summary["development_scores"] = development
    summary["evaluation_receipt_sha256"] = {f"{task['run_id']}.json": sha256((output / "evaluation" / f"{task['run_id']}.json").read_bytes()) for task in chosen_tasks}
    save_checkpoint(output / "summary.json", summary)
    if args.publish:
        destination = Path(args.publish)
        destination.mkdir(parents=True, exist_ok=True)
        if any((destination / name).exists() for name in ("plan.json", "summary.json", "document_scores.jsonl")):
            raise FileExistsError("Refusing to overwrite published evidence")
        for name, payload in (("plan.json", plan), ("summary.json", summary)):
            save_checkpoint(destination / name, payload)
        with (destination / "document_scores.jsonl").open("w") as handle:
            handle.write(json.dumps({"contract_sha256": contract_hash, "columns": ["budget", "seed", "arm", "language", "document_group", "sentence_count", *[mode + "_f1_sum" for mode in POLICIES], "nonwhitespace_f1_sum"]}) + "\n")
            for row in document_rows:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps({"complete": True, "contract_sha256": contract_hash, "training_runs": len(receipts), "evaluated_models": len(evaluations)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/policy-learning-comparison.json"))
    parser.add_argument("--sources", default=str(ROOT / "configs/pud-v2.18-sources.json"))
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache/pud"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--publish")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
