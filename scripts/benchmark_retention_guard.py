#!/usr/bin/env python3
"""Prospective Chinese-retention / Ukrainian-adaptation comparison.

Only the old training partition trains candidates. Fresh official test corpora
are evaluation-only. All controls, shuffle seeds and weight mixing are fixed
before scoring. Failures and orphaned attempts are retained, never retried.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_slavic_learning as old
from spiralreality_AIT_onepass_aifcore_integrated.integrated.checkpoint_blend import blend_checkpoints
from spiralreality_AIT_onepass_aifcore_integrated.integrated.datasets import iter_samples
from spiralreality_AIT_onepass_aifcore_integrated.integrated.evaluation_metrics import paired_document_interval
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import group_parallel_corpus, load_pinned_sources
from spiralreality_AIT_onepass_aifcore_integrated.integrated.heldout_corpus import load_heldout_sources, select_unseen
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import StudentTrainingConfig
from spiralreality_AIT_onepass_aifcore_integrated.integrated.slavic_corpus import load_slavic_sources


def read(path):
    return json.loads(Path(path).read_text())


def fingerprints():
    return {**old.fingerprints(), str(Path(__file__).relative_to(ROOT)): old.file_hash(__file__)}


def runtime():
    return {"python": platform.python_version(), "numpy": old.np.__version__, "platform": platform.platform()}


def validate_config(cfg):
    if (cfg.get("schema_version") != 1 or cfg.get("adaptation_weight") != 0.75
            or cfg.get("replay_per_language") != {"en": 64, "ja": 64, "zh": 160}
            or cfg.get("seeds") != [5042, 101] or cfg.get("fresh_cap_per_language") != 128
            or cfg.get("fresh_max_chars") != 192):
        raise ValueError("Use the declared prospective retention protocol")


def build_inputs(cfg, *, download=False):
    prior_plan = read(ROOT / cfg["prior_plan"])
    contract = prior_plan["contract"]
    if old.digest(contract) != cfg["prior_contract_sha256"] or prior_plan["contract_sha256"] != cfg["prior_contract_sha256"]:
        raise ValueError("Prior experiment contract mismatch")
    # Prior numerical files remain intact. Additional modules do not invalidate
    # the earlier immutable source receipts or cause the old study to be rerun.
    if any(old.file_hash(ROOT / path) != digest for path, digest in contract["code_sha256"].items()):
        raise ValueError("An implementation file used by the prior study changed")
    args = SimpleNamespace(legacy_sources=ROOT / "configs/pud-v2.18-sources.json", legacy_cache=ROOT / ".cache/pud",
                           slavic_cache=ROOT / ".cache/slavic-ud", download=download)
    previous, _ = old.prepare_inputs(contract["config"], args)
    if old.digest(previous) != contract["input_sha256"]:
        raise ValueError("Prior training/evaluation cases changed")
    legacy, _ = load_pinned_sources(args.legacy_sources, args.legacy_cache, download=download)
    corpus = group_parallel_corpus(legacy, seed=contract["config"]["partition_seed"],
                                  folds=contract["config"]["fold_count"], max_chars=contract["config"]["legacy_max_chars"])
    roles, _ = corpus.partition(contract["config"]["test_fold"])
    replay = []
    for language, cap in cfg["replay_per_language"].items():
        rows = [old.row_record(row, "train", "pud:" + corpus.group_for_document[row.document_id])
                for row in roles["train"] if row.language == language]
        choice = old.select_rows(rows, cap, contract["config"]["selection_seed"], "rehearsal/" + language)
        if len(choice) != cap:
            raise ValueError("Insufficient old training replay cases")
        replay.extend(choice)
    if len(replay) != len(previous["rehearsal"]):
        raise ValueError("Replay exposure must match the historical rehearsal arm")
    all_slavic, _ = load_slavic_sources(ROOT / contract["config"]["slavic_manifest"], args.slavic_cache,
        pud_document_for_parallel={row.parallel_id: row.document_id for row in legacy}, download=download)
    manifest = ROOT / cfg["fresh_manifest"]
    if old.file_hash(manifest) != cfg["fresh_manifest_sha256"]:
        raise ValueError("Fresh manifest identity mismatch")
    fresh, sources = load_heldout_sources(manifest, ROOT / ".cache/retention-ood", download=download)
    prior_texts = [row.text for row in [*legacy, *all_slavic, *iter_samples()]]
    selected, exclusions = select_unseen(fresh, prior_texts, cap=cfg["fresh_cap_per_language"],
                                         max_chars=cfg["fresh_max_chars"], seed=cfg["selection_seed"])
    fresh_rows = []
    for row in selected:
        item = SimpleNamespace(**asdict(row), parallel_id=None, upstream_split="test")
        group = row.source_id + ":" + (row.document_id if row.document_id is not None else "unresolved-source-file")
        record = old.row_record(item, "test", group)
        record.update(evaluation_role="fresh_test", grouping_unit="provider_document" if row.document_id is not None else "unresolved_source_file")
        fresh_rows.append(record)
    return {"slavic_train": previous["slavic_train"], "rehearsal": replay,
            "evaluation": previous["evaluation"] + fresh_rows}, {
        "fresh_sources": sources, "selection": exclusions, "all_prior_source_overlap_guard_rows": len(prior_texts),
        "slavic_train": [old.identity(row) for row in previous["slavic_train"]],
        "replay": [old.identity(row) for row in replay], "evaluation": [old.identity(row) for row in previous["evaluation"] + fresh_rows]}


def train_worker(task_path):
    task = read(task_path)
    if task["source_sha256"] != fingerprints() or task["runtime"] != runtime() or any(os.environ.get(k) != v for k, v in old.THREADS.items()):
        raise ValueError("Frozen source/runtime/thread identity mismatch")
    if old.file_hash(task["initial_checkpoint"]) != task["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint integrity mismatch")
    payload = read(task["input"])
    if old.digest(payload) != task["input_sha256"]:
        raise ValueError("Training input integrity mismatch")
    model = old.load_model(read(task["initial_checkpoint"]), task["seed"])
    initial = old.parameter_state(model)
    if old.digest(initial) != task["initial_parameter_sha256"]:
        raise ValueError("Initial parameters differ")
    cfg = StudentTrainingConfig(**task["training"])
    examples = payload["slavic_train"] + payload["rehearsal"]
    from dataclasses import replace
    model.train_student([examples[0]["text"]], [examples[0]["segments"]], replace(cfg, epochs=0))
    if old.parameter_state(model) != initial:
        raise ValueError("Warm continuation changed zero-epoch parameters")
    model.student.rng.seed(task["seed"])
    start = time.perf_counter()
    summary = model.train_student([row["text"] for row in examples], [row["segments"] for row in examples], cfg)
    elapsed = time.perf_counter() - start
    final = old.parameter_state(model)
    if (summary["epochs_completed"] != cfg.epochs or final["head"] == initial["head"]
            or final["encoder"] == initial["encoder"] or final["phase"] != initial["phase"]):
        raise ValueError("Continuation intervention mismatch")
    old.immutable_json(task["output_checkpoint"], model.state_dict())
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    receipt = {key: task[key] for key in ("contract_sha256", "seed", "input_sha256", "training", "initial_parameter_sha256", "runtime")}
    receipt.update(zero_epoch_parameter_sha256=old.digest(initial), final_parameter_sha256=old.digest(final),
                   initial_component_sha256={k: old.digest(v) for k, v in initial.items()},
                   final_component_sha256={k: old.digest(v) for k, v in final.items()},
                   checkpoint_sha256=old.file_hash(task["output_checkpoint"]), checkpoint_bytes=Path(task["output_checkpoint"]).stat().st_size,
                   training_wall_seconds=elapsed, whole_training_worker_peak_rss_bytes=int(peak),
                   training_examples_per_language=dict(Counter(row["language"] for row in examples)),
                   processed_examples=len(examples) * cfg.epochs, history=summary["history"])
    if fingerprints() != task["source_sha256"]:
        raise ValueError("Source changed during training")
    old.immutable_json(task["receipt"], receipt)


def validate_training(receipt, task, payload):
    initial = old.parameter_state(old.load_model(read(task["initial_checkpoint"]), task["seed"]))
    final = old.parameter_state(old.load_model(read(task["output_checkpoint"]), task["seed"]))
    examples = payload["slavic_train"] + payload["rehearsal"]
    for key in ("contract_sha256", "seed", "input_sha256", "training", "initial_parameter_sha256", "runtime"):
        if receipt.get(key) != task[key]:
            raise ValueError("Training receipt identity mismatch")
    if (receipt["zero_epoch_parameter_sha256"] != old.digest(initial)
            or receipt["final_parameter_sha256"] != old.digest(final)
            or receipt["initial_component_sha256"] != {k: old.digest(v) for k, v in initial.items()}
            or receipt["final_component_sha256"] != {k: old.digest(v) for k, v in final.items()}
            or final["head"] == initial["head"] or final["encoder"] == initial["encoder"] or final["phase"] != initial["phase"]
            or receipt["checkpoint_sha256"] != old.file_hash(task["output_checkpoint"])
            or receipt["checkpoint_bytes"] != Path(task["output_checkpoint"]).stat().st_size
            or receipt["processed_examples"] != len(examples) * task["training"]["epochs"]
            or receipt["training_examples_per_language"] != dict(Counter(row["language"] for row in examples))
            or len(receipt["history"]) != task["training"]["epochs"]):
        raise ValueError("Training continuation/exposure/checkpoint mismatch")
    for index, row in enumerate(receipt["history"], 1):
        if row["epoch"] != index or any(not isinstance(v, (int, float)) or not old.math.isfinite(v) for v in row.values()):
            raise ValueError("Invalid training history")
    if any(not old.math.isfinite(receipt[k]) or receipt[k] < 0 for k in ("training_wall_seconds", "whole_training_worker_peak_rss_bytes")):
        raise ValueError("Invalid training resource receipt")


def summarize(evaluations, cfg):
    conditions, paired = {}, {}
    for name, rows in evaluations.items():
        cells = sorted({(row["evaluation_role"], row["language"]) for row in rows})
        conditions[name] = {}
        for role, lang in cells:
            selected = [row for row in rows if (row["evaluation_role"], row["language"]) == (role, lang)]
            result = old.aggregate(selected)
            if role == "fresh_test":
                result["documents"] = None if any(row["document_id"] is None for row in selected) else result["documents"]
                result["grouping_caveat"] = "Provider IDs are not proof of independent original documents; Chinese original document IDs are unavailable."
            conditions[name][role + "/" + lang] = result
    for method in ("priority_replay", "priority_blended"):
        paired[method] = {}
        for key in conditions["pre_adaptation"]:
            role, lang = key.split("/")
            if role == "slavic_dev":
                continue
            for reference in ("pre_adaptation", "historical_rehearsal"):
                differences = []
                for seed in cfg["seeds"]:
                    target = evaluations[f"{method}_seed_{seed}"]
                    control = evaluations[reference if reference == "pre_adaptation" else f"{reference}_seed_{seed}"]
                    for current, before in zip(target, control):
                        if (current["evaluation_role"], current["language"]) == (role, lang):
                            differences.append((current["document_group"], current["boundary"]["f1"] - before["boundary"]["f1"]))
                interval = paired_document_interval(differences, **cfg["bootstrap"])
                if role == "fresh_test" and lang in ("zh", "ja"):
                    interval.update(document_bootstrap_95_percent=None, document_groups=None,
                                    scope="Point estimate only: Chinese lacks original document IDs; Japanese provider units equal sentences, so original document uncertainty is unresolved.")
                else:
                    interval["scope"] = "Conditional provider-document/date resampling; one starting checkpoint and two shuffle orders, no initialization or population guarantee."
                paired[method][key + "/vs_" + reference] = interval

    def mean(method, cell, metric, kind):
        names = [method] if method == "pre_adaptation" else [f"{method}_seed_{seed}" for seed in cfg["seeds"]]
        return statistics.mean(conditions[name][cell][metric][kind] for name in names)

    gates = {}
    for method in ("priority_replay", "priority_blended"):
        criteria = {
            "historical_uk_nonwhite_at_least_97_percent": mean(method, "slavic_test/uk", "nonwhitespace", "micro_f1") >= .97,
            "fresh_uk_within_half_point_of_historical_rehearsal": mean(method, "fresh_test/uk", "nonwhitespace", "micro_f1") >= mean("historical_rehearsal", "fresh_test/uk", "nonwhitespace", "micro_f1") - .005,
            "fresh_zh_no_worse_than_historical_rehearsal": mean(method, "fresh_test/zh", "boundary", "macro_f1") >= mean("historical_rehearsal", "fresh_test/zh", "boundary", "macro_f1"),
        }
        for role in ("retention_test", "fresh_test"):
            for lang in ("en", "ja", "zh"):
                tolerance = cfg["retention_tolerance"][lang]
                criteria[f"{role}_{lang}_within_predeclared_tolerance"] = mean(method, role + "/" + lang, "boundary", "macro_f1") >= mean("pre_adaptation", role + "/" + lang, "boundary", "macro_f1") - tolerance
        gates[method] = {"passed": all(criteria.values()), "criteria": criteria}
    return {"conditions": conditions, "paired_full_boundary_macro": paired, "gates": gates,
            "decision_policy": "Fixed candidates, no outcome-based model selection or retuning. Passing is descriptive on the frozen sets; unknown document provenance and a single pretrained initialization remain unresolved."}


def run(args):
    cfg = read(args.config)
    validate_config(cfg)
    if any(os.environ.get(k) != v for k, v in old.THREADS.items()):
        raise ValueError("Set all three BLAS thread environment variables to one before launch")
    previous = read(ROOT / cfg["prior_plan"])["contract"]
    checkpoint = ROOT / previous["config"]["checkpoint"]
    if old.file_hash(checkpoint) != previous["checkpoint_sha256"]:
        raise ValueError("Pre-adaptation checkpoint identity mismatch")
    initial_state = read(checkpoint)
    historical = {}
    previous_summary = read(ROOT / cfg["prior_summary"])
    if old.file_hash(ROOT / cfg["prior_summary"]) != cfg["prior_summary_sha256"]:
        raise ValueError("Prior summary identity mismatch")
    for seed in cfg["seeds"]:
        name = f"slavic_rehearsal_seed_{seed}"
        path = ROOT / ".cache/slavic-learning-pilot-v1/checkpoints" / (name + ".json")
        if old.file_hash(path) != previous_summary["training"][name]["checkpoint_sha256"]:
            raise ValueError("Historical fitted checkpoint identity mismatch")
        historical[f"historical_rehearsal_seed_{seed}"] = path
    payload, provenance = build_inputs(cfg, download=args.download)
    settings = old.training_settings(initial_state, previous["config"])
    contract = {"config": cfg, "runtime": runtime(), "source_sha256": fingerprints(), "input_sha256": old.digest(payload),
                "initial_checkpoint_sha256": old.file_hash(checkpoint), "initial_parameter_sha256": old.digest(old.parameter_state(old.load_model(initial_state, 5042))),
                "historical_checkpoint_sha256": {k: old.file_hash(v) for k, v in historical.items()},
                "training": settings, "provenance": provenance}
    contract_hash = old.digest(contract)
    plan = {"contract_sha256": contract_hash, "contract": contract}
    output = Path(args.output)
    old.immutable_json(output / "plan.json", plan)
    print(json.dumps({"prepared": True, "contract_sha256": contract_hash, "training_jobs": len(cfg["seeds"]),
                      "training_cases": len(payload["slavic_train"]) + len(payload["rehearsal"]), "evaluation_cases": len(payload["evaluation"])}), flush=True)
    if args.prepare_only:
        return
    input_path = output / "input.json"
    old.immutable_json(input_path, payload)
    models, training, blends = {"pre_adaptation": checkpoint, **historical}, {}, {}
    for seed in cfg["seeds"]:
        name = f"priority_replay_seed_{seed}"
        destination, receipt_path = output / "checkpoints" / (name + ".json"), output / "training" / (name + ".json")
        task = {"contract_sha256": contract_hash, "seed": seed, "source_sha256": contract["source_sha256"], "runtime": runtime(),
                "initial_checkpoint": str(checkpoint.resolve()), "initial_checkpoint_sha256": contract["initial_checkpoint_sha256"],
                "initial_parameter_sha256": contract["initial_parameter_sha256"], "input": str(input_path.resolve()),
                "input_sha256": contract["input_sha256"], "training": settings, "output_checkpoint": str(destination.resolve()), "receipt": str(receipt_path.resolve())}
        task_path = output / "tasks" / (name + ".json")
        old.immutable_json(task_path, task)
        if not receipt_path.exists():
            attempt = output / "attempts" / (name + ".json")
            if attempt.exists():
                raise ValueError("Incomplete training attempt retained: " + name)
            old.immutable_json(attempt, {"contract_sha256": contract_hash, "started_at_utc": datetime.now(timezone.utc).isoformat()})
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(task_path.resolve())],
                           cwd=ROOT, env={**os.environ, **old.THREADS, "PYTHONHASHSEED": "0"}, check=True)
        receipt = read(receipt_path)
        validate_training(receipt, task, payload)
        training[name] = {**receipt, "receipt_sha256": old.file_hash(receipt_path)}
        models[name] = destination
        blended_name = f"priority_blended_seed_{seed}"
        blended_path = output / "checkpoints" / (blended_name + ".json")
        start = time.perf_counter()
        blended = blend_checkpoints(initial_state, read(destination), adaptation_weight=cfg["adaptation_weight"])
        blend_seconds = time.perf_counter() - start
        old.immutable_json(blended_path, blended)
        models[blended_name] = blended_path
        blend_receipt = output / "blends" / (blended_name + ".json")
        if not blend_receipt.exists():
            old.immutable_json(blend_receipt, {"contract_sha256": contract_hash, "adaptation_weight": cfg["adaptation_weight"],
                               "reference_checkpoint_sha256": old.file_hash(checkpoint), "adapted_checkpoint_sha256": old.file_hash(destination),
                               "checkpoint_sha256": old.file_hash(blended_path), "parameter_sha256": old.digest(old.parameter_state(old.load_model(blended, seed))),
                               "checkpoint_bytes": blended_path.stat().st_size, "construction_seconds": blend_seconds})
        blends[blended_name] = read(blend_receipt)
        if (blends[blended_name]["checkpoint_sha256"] != old.file_hash(blended_path)
                or blends[blended_name]["contract_sha256"] != contract_hash
                or blends[blended_name]["reference_checkpoint_sha256"] != old.file_hash(checkpoint)
                or blends[blended_name]["adapted_checkpoint_sha256"] != old.file_hash(destination)
                or blends[blended_name]["adaptation_weight"] != cfg["adaptation_weight"]
                or blends[blended_name]["checkpoint_bytes"] != blended_path.stat().st_size
                or not old.math.isfinite(blends[blended_name]["construction_seconds"])
                or blends[blended_name]["construction_seconds"] < 0
                or blends[blended_name]["parameter_sha256"] != old.digest(old.parameter_state(old.load_model(blended, seed)))):
            raise ValueError("Blended checkpoint receipt mismatch")
        print(json.dumps({"training_completed": name, "training_seconds": receipt["training_wall_seconds"]}), flush=True)
    evaluations, receipt_hashes = {}, {}
    for name, path in models.items():
        destination = output / "evaluation" / (name + ".json")
        if not destination.exists():
            attempt = output / "attempts" / ("evaluation_" + name + ".json")
            if attempt.exists():
                raise ValueError("Incomplete evaluation retained: " + name)
            old.immutable_json(attempt, {"contract_sha256": contract_hash, "started_at_utc": datetime.now(timezone.utc).isoformat()})
            results = old.tree(old.evaluate_rows(old.load_model(read(path), 5042), payload["evaluation"]))
            old.immutable_json(destination, {"contract_sha256": contract_hash, "checkpoint_sha256": old.file_hash(path),
                                            "results": results, "results_sha256": old.digest(results)})
        result = read(destination)
        if (result["contract_sha256"] != contract_hash or result["checkpoint_sha256"] != old.file_hash(path)
                or result["results_sha256"] != old.digest(result["results"])):
            raise ValueError("Evaluation receipt integrity mismatch")
        old.validate_results(result["results"], payload["evaluation"])
        evaluations[name] = result["results"]
        receipt_hashes[name] = old.file_hash(destination)
        print(json.dumps({"evaluated": name, "cases": len(result["results"])}), flush=True)
    if fingerprints() != contract["source_sha256"]:
        raise ValueError("Frozen source changed during the comparison")
    summary = {"contract_sha256": contract_hash, **summarize(evaluations, cfg), "training": training, "blends": blends,
               "evaluation_receipt_sha256": receipt_hashes, "evaluated_cases": len(payload["evaluation"]),
               "total_evaluations": sum(len(rows) for rows in evaluations.values())}
    old.immutable_json(output / "summary.json", summary)
    if args.publish:
        public = Path(args.publish)
        if public.exists():
            raise FileExistsError("Published evidence is immutable; use a new directory")
        old.immutable_json(public / "plan.json", plan)
        old.immutable_json(public / "summary.json", summary)
        with (public / "paired_scores.jsonl").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"contract_sha256": contract_hash, "conditions": list(evaluations)}) + "\n")
            for index, row in enumerate(payload["evaluation"]):
                values = {name: {key: rows[index][key] for key in ("prediction_sha256", "probability_sha256", "boundary", "nonwhitespace", "calibration", "inference_ms")} for name, rows in evaluations.items()}
                handle.write(json.dumps({**old.identity(row), "conditions": values}, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "total_evaluations": summary["total_evaluations"], "gates": summary["gates"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/retention-guard-pilot.json"))
    parser.add_argument("--output")
    parser.add_argument("--publish")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        train_worker(args.worker)
    elif args.output:
        run(args)
    else:
        parser.error("--output is required")
