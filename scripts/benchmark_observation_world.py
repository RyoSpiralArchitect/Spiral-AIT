#!/usr/bin/env python3
"""Frozen comparison of Bayesian observation hypotheses on actual text.

Reuse pinned artifacts from the earlier study without treating its historical
code hashes as today's implementation. New plans bind today's code separately.
Detailed observations stay local; published rows contain scores and hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts import benchmark_active_context as previous
from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import ActiveContextConfig, ActiveTextSession, StringTextSource
from spiralreality_AIT_onepass_aifcore_integrated.integrated.evaluation_metrics import boundary_counts, boundary_targets
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import LANGUAGES, group_parallel_corpus, json_hash, load_pinned_sources, sha256
from spiralreality_AIT_onepass_aifcore_integrated.integrated.observation_world import BayesianObservationWorld

MODES = ("fixed_channel", "fixed_mixture", "adaptive", "null_channel")
WORLD_METRICS = ("observation_world_updates", "regime_information_gain_sum", "outcome_nll_sum", "outcome_brier_sum", "outcome_forecasts")
REQUIRED_METRICS = {"acquire_calls", "acquired_chars", "deliberate_acquire_calls", "commit_actions", "forced_splits", "model_calls",
                    "window_square_sum", "max_window_chars_seen", "world_model_updates", "wait_char_histogram", "policy_counts",
                    *previous.GAIN_METRICS, *WORLD_METRICS}
SOURCE_ARTIFACTS = {"plan.json", "channel.json", "channel_fit.json", "warm_started_model.json", "warm_start.json"}


def fingerprints():
    paths = sorted((ROOT / previous.PACKAGE).rglob("*.py")) + [Path(__file__).resolve(), Path(previous.__file__).resolve()]
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in paths}


def json_tree(value):
    """Normalize JSON object keys before hashing what will actually be stored.

    Waiting histograms use integer keys in memory. JSON turns them into strings;
    sorting before versus after that conversion differs for keys such as 2/10.
    Normalize once so a stored receipt retains its canonical hash on reload.
    Reject nonfinite values before any evidence file is created.
    """
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def immutable_json(path, payload):
    """Create evidence once; an identical resumed read is allowed."""
    payload = json_tree(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Immutable evidence differs: {path.name}")
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def validate_config(config):
    if config.get("schema_version") != 1 or config.get("modes") != list(MODES):
        raise ValueError("Declare the four frozen observation-world modes")
    costs = config.get("char_costs")
    if (not isinstance(costs, list) or not costs or len(set(costs)) != len(costs)
            or any(isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0 for cost in costs)):
        raise ValueError("Declare distinct finite nonnegative costs before evaluation")
    if len({f"{cost:g}" for cost in costs}) != len(costs):
        raise ValueError("Cost names collide at the evidence filename precision")
    if set(config.get("source_artifact_sha256", {})) != SOURCE_ARTIFACTS:
        raise ValueError("Pin the complete prior-study artifact set")
    bootstrap = config["bootstrap"]
    if (isinstance(bootstrap["replicates"], bool) or not isinstance(bootstrap["replicates"], int) or bootstrap["replicates"] < 1
            or not isinstance(bootstrap["seed"], int) or isinstance(bootstrap["seed"], bool)
            or not 0 < bootstrap["confidence"] < 1):
        raise ValueError("Invalid document bootstrap contract")
    if not 0 <= config["informative_prior"] <= 1 or config["model_epistemic_weight"] != 0:
        raise ValueError("This study freezes the regime prior and zero model-information reward")
    probe = config["probe"]
    if (probe["focus"] != "maximum_initial_boundary_entropy" or probe["initial_chars"] < 2
            or probe["max_window_chars"] < probe["initial_chars"] or not probe["action_cycle"]
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in probe["action_cycle"])):
        raise ValueError("Invalid shared-observation probe contract")


def load_source_study(directory, config):
    directory = Path(directory)
    artifacts = {}
    for name, expected in config["source_artifact_sha256"].items():
        data = (directory / name).read_bytes()
        if sha256(data) != expected:
            raise ValueError(f"Prior study artifact integrity mismatch: {name}")
        artifacts[name] = json.loads(data)
    plan = artifacts["plan.json"]
    contract_hash = config["source_contract_sha256"]
    if plan["contract_sha256"] != contract_hash or json_hash(plan["contract"]) != contract_hash:
        raise ValueError("Prior study contract identity mismatch")
    fit, warm = artifacts["channel_fit.json"], artifacts["warm_start.json"]
    if (fit["contract_sha256"] != contract_hash or warm["contract_sha256"] != contract_hash
            or fit["channel_sha256"] != json_hash(artifacts["channel.json"])
            or warm["checkpoint_sha256"] != config["source_artifact_sha256"]["warm_started_model.json"]
            or fit["warm_transitions_sha256"] != warm["warm_transitions_sha256"]):
        raise ValueError("Prior fitted channel/world receipt mismatch")
    state = artifacts["warm_started_model.json"]
    protected = {name: state[name] for name in ("student", "encoder", "phase")}
    if json_hash(protected) != warm["protected_segmenter_sha256"]:
        raise ValueError("Prior protected segmenter identity mismatch")
    channel = ContextAcquisitionModel.from_state(artifacts["channel.json"]).freeze()
    return artifacts, channel


def load_parent_study(directory):
    """Load pinned historical weights; callers separately bind current code."""
    config = json.loads((ROOT / "configs/observation-world-pilot.json").read_text())
    validate_config(config)
    artifacts, channel = load_source_study(directory, config)
    identity = {"contract_sha256": config["source_contract_sha256"],
                "artifact_sha256": config["source_artifact_sha256"],
                "protected_segmenter_sha256": artifacts["warm_start.json"]["protected_segmenter_sha256"]}
    return artifacts["warm_started_model.json"], channel, identity, artifacts["plan.json"]["contract"]["config"]["active"].copy()


def mode_values(mode, config):
    if mode not in MODES:
        raise ValueError("Unknown observation-world mode")
    return {"observation_model": "fixed_channel" if mode == "fixed_channel" else "bayesian_regime",
            "informative_prior": 0.0 if mode == "null_channel" else config["informative_prior"],
            "learn_observation_regime": mode != "fixed_mixture",
            "model_epistemic_weight": config["model_epistemic_weight"],
            "phase_gain": 0.0, "surprise_gain": 0.0, "epistemic_weight": 0.05, "learn_dynamics": False}


def run_episode(model, text, channel, prior_config, config, *, mode, char_cost):
    settings = {**prior_config["active"], "char_cost": char_cost, **mode_values(mode, config)}
    session = ActiveTextSession(model, channel, ActiveContextConfig(**settings))
    return session.run(StringTextSource(text))


def row_identity(row, corpus):
    return {"language": row.language, "parallel_id": row.parallel_id,
            "document_group": corpus.group_for_document[row.document_id], "characters": len(row.text),
            "source_sha256": sha256(row.text.encode()), "labels_sha256": json_hash(row.segments)}


def evaluate_rows(state, rows, channel, prior_config, config, *, mode, char_cost, corpus):
    results = []
    original_channel = json_hash(channel.export_state())
    for row in rows:
        model = previous.load_model(state, prior_config["model_seed"])
        started = time.perf_counter()
        episode = run_episode(model, row.text, channel, prior_config, config, mode=mode, char_cost=char_cost)
        elapsed = (time.perf_counter() - started) * 1000
        if "".join(episode["tokens"]) != row.text or not episode.get("done") or episode.get("decoder") != "marginal_map":
            raise ValueError("Episode failed to complete source-preserving marginal-MAP output")
        metrics = episode["metrics"]
        if not REQUIRED_METRICS <= metrics.keys():
            raise ValueError(f"Observation metrics missing: {sorted(REQUIRED_METRICS - metrics.keys())}")
        if (metrics["acquired_chars"] != len(row.text) or metrics["max_window_chars_seen"] > prior_config["active"]["max_window_chars"]
                or metrics["world_model_updates"] != 0):
            raise ValueError("Episode violated acquisition, window, or frozen latent-weight contract")
        final_state = model.state_dict()
        protected = {name: final_state[name] for name in ("student", "encoder", "phase")}
        if protected != {name: state[name] for name in protected}:
            raise ValueError("Episode altered protected segmenter or local phase state")
        results.append({**row_identity(row, corpus), "episode_elapsed_ms": elapsed,
                        "decoder": "marginal_map", "belief_model": mode,
                        "prediction_sha256": json_hash(episode["tokens"]),
                        "boundary": boundary_counts(row.text, row.segments, episode["tokens"]),
                        "nonwhitespace": boundary_counts(row.text, row.segments, episode["tokens"], exclude_whitespace=True),
                        "metrics": metrics, "trace": episode["trace"]})
    if json_hash(channel.export_state()) != original_channel:
        raise ValueError("Evaluation mutated the frozen supervised channel")
    return results


def validate_results(results, rows, corpus, *, mode, max_window):
    if not isinstance(results, list) or len(results) != len(rows):
        raise ValueError("Incomplete or duplicate evaluation rows")
    for result, row in zip(results, rows):
        if any(result.get(key) != value for key, value in row_identity(row, corpus).items()):
            raise ValueError("Evaluation row identity/order mismatch")
        if result.get("belief_model") != mode or result.get("decoder") != "marginal_map":
            raise ValueError("Evaluation decoder/mode mismatch")
        metrics = result["metrics"]
        if not REQUIRED_METRICS <= metrics.keys():
            raise ValueError("Incomplete observation metrics")
        numeric = [result["episode_elapsed_ms"], *(metrics[name] for name in REQUIRED_METRICS - {"wait_char_histogram", "policy_counts", "realized_entropy_reduction_sum"})]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
               for value in numeric):
            raise ValueError("Invalid finite evaluation metrics")
        if not math.isfinite(metrics["realized_entropy_reduction_sum"]):
            raise ValueError("Invalid signed information measurement")
        if metrics["acquired_chars"] != len(row.text) or metrics["world_model_updates"] != 0 or metrics["max_window_chars_seen"] > max_window:
            raise ValueError("Invalid frozen-world evaluation receipt")


def aggregate(rows):
    result = previous.aggregate(rows)
    for name in WORLD_METRICS:
        result[name] = math.fsum(row["metrics"][name] for row in rows)
    count = result["outcome_forecasts"]
    result["mean_outcome_nll"] = result["outcome_nll_sum"] / count if count else None
    result["mean_outcome_brier"] = result["outcome_brier_sum"] / count if count else None
    return result


def _entropy(probability):
    return -probability * math.log(probability) - (1 - probability) * math.log1p(-probability) if 0 < probability < 1 else 0.0


def shared_observation_probe(state, rows, channel, prior_config, config, corpus):
    """Same initial boundary and real prefix extensions for all four filters.

    No action uses a label; only final posterior scoring reads the gold label.
    Prefix observations are produced once and supplied identically to all modes.
    The probe follows no controller's preferred policy and establishes no F1
    gain. Its repeated transitions are correlated within a sentence/document.
    """
    from bisect import bisect_right
    settings = config["probe"]
    initial, maximum, cycle = settings["initial_chars"], settings["max_window_chars"], settings["action_cycle"]
    output = []
    worlds = {mode: BayesianObservationWorld(channel, informative_prior=mode_values(mode, config)["informative_prior"])
              for mode in MODES if mode != "fixed_channel"}
    for row in rows:
        identity = row_identity(row, corpus)
        if len(row.text) < initial:
            output.append({**identity, "steps": 0, "reason": "shorter_than_initial", "modes": {}})
            continue
        model = previous.load_model(state, prior_config["model_seed"])
        raw = list(map(float, model.student.observe_sequence(row.text[:initial])["boundary_probabilities"]))
        focus = max(range(len(raw)), key=lambda index: (_entropy(raw[index]), -index))
        beliefs = {mode: worlds[mode].initial_belief(raw[focus]) if mode in worlds else raw[focus] for mode in MODES}
        totals = {mode: {"forecasts": 0, "outcome_nll_sum": 0.0, "outcome_brier_sum": 0.0,
                         "label_brier_sum": 0.0, "regime_information_gain_sum": 0.0} for mode in MODES}
        length, step = initial, 0
        observed_raw = raw[focus]
        while True:
            amount = cycle[step % len(cycle)]
            if length + amount > min(maximum, len(row.text)):
                break
            age = length - focus - 1
            estimates = {mode: worlds[mode].estimate(beliefs[mode], observed_raw, age, amount) if mode in worlds
                         else channel.estimate(beliefs[mode], age, amount, conditioning_probability=observed_raw)
                         for mode in MODES}
            # All forecasts have been frozen before the next prefix is inspected.
            after = float(model.student.observe_sequence(row.text[:length + amount])["boundary_probabilities"][focus])
            outcome = min(len(channel.outcome_edges) - 2, bisect_right(channel.outcome_edges, after) - 1)
            label = int(boundary_targets(row.text, row.segments)[focus])
            for mode, estimate in estimates.items():
                if not estimate["eligible"]:
                    beliefs[mode] = worlds[mode].initial_belief(after) if mode in worlds else after
                    continue
                masses, posterior = estimate["outcome_probabilities"], estimate["posterior_probabilities"][outcome]
                totals[mode]["forecasts"] += 1
                totals[mode]["outcome_nll_sum"] -= math.log(masses[outcome])
                totals[mode]["outcome_brier_sum"] += math.fsum((mass - int(index == outcome)) ** 2 for index, mass in enumerate(masses))
                totals[mode]["label_brier_sum"] += (posterior - label) ** 2
                totals[mode]["regime_information_gain_sum"] += estimate.get("regime_information_gain", 0.0)
                beliefs[mode] = (worlds[mode].initial_belief(posterior) if mode == "fixed_mixture"
                                 else estimate["joint_posteriors"][outcome] if mode in worlds else posterior)
            observed_raw = after
            length += amount
            step += 1
        for mode in worlds:
            totals[mode]["final_informative_probability"] = worlds[mode].informative_probability(beliefs[mode])
        output.append({**identity, "steps": step, "focus_offset": focus + 1, "observed_chars": length, "modes": totals})
    return output


def aggregate_probe(rows):
    result = {}
    for language in LANGUAGES:
        result[language] = {}
        for mode in MODES:
            selected = [row["modes"][mode] for row in rows if row["language"] == language and mode in row["modes"]]
            totals = {name: math.fsum(row[name] for row in selected) for name in
                      ("forecasts", "outcome_nll_sum", "outcome_brier_sum", "label_brier_sum", "regime_information_gain_sum")}
            count = totals["forecasts"]
            result[language][mode] = {**totals, **{f"mean_{key[:-4]}": value / count if count else None
                                                        for key, value in totals.items() if key.endswith("_sum")}}
    return result


def validate_probe(results, rows, corpus, config):
    """Reject incomplete rehashed receipts, not only accidental byte changes."""
    if not isinstance(results, list) or len(results) != len(rows):
        raise ValueError("Shared probe row count mismatch")
    settings = config["probe"]
    initial, maximum, cycle = settings["initial_chars"], settings["max_window_chars"], settings["action_cycle"]
    totals_keys = {"forecasts", "outcome_nll_sum", "outcome_brier_sum", "label_brier_sum", "regime_information_gain_sum"}
    for result, row in zip(results, rows):
        identity = row_identity(row, corpus)
        if any(result.get(key) != value for key, value in identity.items()):
            raise ValueError("Shared probe row identity/order mismatch")
        if len(row.text) < initial:
            if result != {**identity, "steps": 0, "reason": "shorter_than_initial", "modes": {}}:
                raise ValueError("Shared probe short-row schema mismatch")
            continue
        expected_length, expected_steps = initial, 0
        while expected_length + cycle[expected_steps % len(cycle)] <= min(maximum, len(row.text)):
            expected_length += cycle[expected_steps % len(cycle)]
            expected_steps += 1
        if (set(result) != {*identity, "steps", "focus_offset", "observed_chars", "modes"}
                or isinstance(result["steps"], bool) or result["steps"] != expected_steps
                or isinstance(result["observed_chars"], bool) or result["observed_chars"] != expected_length
                or isinstance(result["focus_offset"], bool) or not isinstance(result["focus_offset"], int)
                or not 1 <= result["focus_offset"] < initial):
            raise ValueError("Shared probe prefix/focus contract mismatch")
        if not isinstance(result["modes"], dict) or set(result["modes"]) != set(MODES):
            raise ValueError("Shared probe requires all four mode totals")
        counts = set()
        for mode, totals in result["modes"].items():
            expected = totals_keys if mode == "fixed_channel" else totals_keys | {"final_informative_probability"}
            if not isinstance(totals, dict) or set(totals) != expected:
                raise ValueError("Shared probe mode-total schema mismatch")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in totals.values()):
                raise ValueError("Shared probe totals must be finite and nonnegative")
            count = totals["forecasts"]
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= expected_steps:
                raise ValueError("Shared probe forecast count outside actual steps")
            counts.add(count)
            if (totals["outcome_brier_sum"] > 2 * count + 1e-12 or totals["label_brier_sum"] > count + 1e-12
                    or totals["regime_information_gain_sum"] > math.log(2) * count + 1e-12):
                raise ValueError("Shared probe score outside proper-score bounds")
            if mode != "fixed_channel" and totals["final_informative_probability"] > 1:
                raise ValueError("Shared probe regime marginal outside probability bounds")
        if len(counts) != 1:
            raise ValueError("Shared probe modes scored different outcome counts")


def paired_probe_bootstrap(rows, baseline, settings):
    """Proper-score deltas on the identical focused outcomes, grouped by document."""
    groups = sorted({row["document_group"] for row in rows})
    group_index = {group: index for index, group in enumerate(groups)}
    values = np.zeros((len(groups), len(LANGUAGES), 3), dtype=float)
    for row in rows:
        if not row["modes"]:
            continue
        before, after = row["modes"][baseline], row["modes"]["adaptive"]
        if before["forecasts"] != after["forecasts"]:
            raise ValueError("Common-observation modes scored different outcome counts")
        values[group_index[row["document_group"]], LANGUAGES.index(row["language"])] += [
            after["outcome_nll_sum"] - before["outcome_nll_sum"],
            after["outcome_brier_sum"] - before["outcome_brier_sum"], after["forecasts"]]
    totals = values.sum(axis=0)
    rng = np.random.default_rng(settings["seed"])
    selections = rng.integers(0, len(groups), size=(settings["replicates"], len(groups)))
    sums = values[selections].sum(axis=1)
    low, high = (1 - settings["confidence"]) / 2, (1 + settings["confidence"]) / 2
    result = {"method": "paired_document_group_percentile_bootstrap_on_common_outcomes", "baseline": baseline,
              "negative_delta_favors_adaptive": True, "groups": len(groups), "seed": settings["seed"],
              "replicates": settings["replicates"], "confidence": settings["confidence"], "languages": {}}
    for index, language in enumerate(LANGUAGES):
        valid = sums[:, index, 2] > 0
        if totals[index, 2] == 0 or not np.any(valid):
            result["languages"][language] = {"forecasts": 0, "outcome_nll": None, "outcome_brier": None}
            continue
        distribution = sums[valid, index, :2] / sums[valid, index, 2:3]
        point = totals[index, :2] / totals[index, 2]
        result["languages"][language] = {"forecasts": int(totals[index, 2]), "valid_bootstrap_replicates": int(valid.sum()),
                                         **{name: {"delta": float(point[column]),
                                                   "interval": list(map(float, np.quantile(distribution[:, column], [low, high])))}
                                            for column, name in enumerate(("outcome_nll", "outcome_brier"))}}
    return result


def paired_bootstrap(left, right, settings):
    """Paired sentence macro-F1 differences; documents jointly resampled by language."""
    key = lambda row: (row["language"], row["parallel_id"])
    if {key(row) for row in left} != {key(row) for row in right}:
        raise ValueError("Paired bootstrap requires matching sentence identities")
    before = {key(row): row for row in left}
    groups = sorted({row["document_group"] for row in right})
    group_index = {group: index for index, group in enumerate(groups)}
    # Per group/language: sums of F1 deltas, nonwhite deltas, and sentence counts.
    values = np.zeros((len(groups), len(LANGUAGES), 3), dtype=float)
    changed = Counter()
    for row in right:
        old = before[key(row)]
        if row["document_group"] != old["document_group"] or row["labels_sha256"] != old["labels_sha256"]:
            raise ValueError("Paired bootstrap document/label mismatch")
        cell = values[group_index[row["document_group"]], LANGUAGES.index(row["language"])]
        cell += [row["boundary"]["f1"] - old["boundary"]["f1"], row["nonwhitespace"]["f1"] - old["nonwhitespace"]["f1"], 1]
        changed[row["language"]] += int(row["prediction_sha256"] != old["prediction_sha256"])
    rng = np.random.default_rng(settings["seed"])
    selections = rng.integers(0, len(groups), size=(settings["replicates"], len(groups)))
    sums = values[selections].sum(axis=1)
    if np.any(sums[:, :, 2] == 0):
        raise ValueError("Bootstrap document sample omitted an entire language")
    samples = sums[:, :, :2] / sums[:, :, 2:3]
    totals = values.sum(axis=0)
    point = totals[:, :2] / totals[:, 2:3]
    low, high = (1 - settings["confidence"]) / 2, (1 + settings["confidence"]) / 2
    result = {"method": "paired_document_group_percentile_bootstrap", "groups": len(groups),
              "seed": settings["seed"], "replicates": settings["replicates"], "confidence": settings["confidence"],
              "prediction_changed_sentences": dict(changed), "languages": {}}
    for index, language in enumerate((*LANGUAGES, "equal_language_mean")):
        distribution = samples[:, index, :] if index < len(LANGUAGES) else samples.mean(axis=1)
        estimate = point[index] if index < len(LANGUAGES) else point.mean(axis=0)
        result["languages"][language] = {name: {"delta": float(estimate[column]),
                                               "interval": list(map(float, np.quantile(distribution[:, column], [low, high])))}
                                          for column, name in enumerate(("macro_f1", "nonwhitespace_macro_f1"))}
    return result


def _condition(mode, cost):
    return f"{mode}_cost_{cost:g}"


def run(args):
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    source_directory = Path(args.source_study) if args.source_study else ROOT / config["source_study"]
    artifacts, channel = load_source_study(source_directory, config)
    old_contract = artifacts["plan.json"]["contract"]
    old_config = old_contract["config"]
    state = artifacts["warm_started_model.json"]
    rows, sources = load_pinned_sources(args.sources, args.cache_dir, download=False)
    corpus = group_parallel_corpus(rows, seed=old_config["partition_seed"], folds=old_config["fold_count"], max_chars=old_config["max_chars"])
    partitions, partition_receipt = corpus.partition(old_config["test_fold"])
    if sources != old_contract["sources"] or corpus.receipt != old_contract["partition"] or partition_receipt != old_contract["selection_partition"]:
        raise ValueError("Source data or full row partition differs from the historical study")
    previous.split_development(corpus, partitions, old_config["development_split_seed"])
    conditions = [(mode, cost) for cost in config["char_costs"] for mode in MODES]
    conditions.sort(key=lambda condition: sha256(f"{config['bootstrap']['seed']}\0{condition[0]}\0{condition[1]:g}".encode()))
    code = fingerprints()
    contract = {"config": config, "sources": sources, "partition": corpus.receipt,
                "selection_partition": partition_receipt, "prior_study_contract_sha256": config["source_contract_sha256"],
                "historical_source_code_sha256": old_contract["code_sha256"], "current_code_sha256": code,
                "evaluation_order": [[mode, cost] for mode, cost in conditions],
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                            "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
                "limitations": [config["protocol"], "Closed-loop outcome scores use each policy's acquired states; only the separate common-observation probe shares forecast targets.",
                                "Document bootstrap measures sampling variation on this already exposed corpus, not unseen-corpus uncertainty."]}
    contract_hash = json_hash(contract)
    output = Path(args.output)
    if output.resolve() == source_directory.resolve():
        raise ValueError("New evidence output cannot replace the historical study")
    output.mkdir(parents=True, exist_ok=True)
    plan = {"contract_sha256": contract_hash, "contract": contract}
    immutable_json(output / "plan.json", plan)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "contract_sha256": contract_hash, "conditions": len(conditions),
                          "samples_per_condition": len(partitions["test"])}), flush=True)
        return
    # This exclusive immutable marker always predates the first evaluated row.
    start_path = output / "evaluation_start.json"
    if start_path.exists():
        start = json.loads(start_path.read_text())
        if start["contract_sha256"] != contract_hash:
            raise ValueError("Evaluation-start contract mismatch")
    else:
        start = {"contract_sha256": contract_hash, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                 "selection": "none; both predeclared costs are evaluated"}
        immutable_json(start_path, start)
    evaluations, hashes = {}, {}
    for mode, cost in conditions:
        name = _condition(mode, cost)
        path = output / "evaluation" / f"{name}.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            if (receipt["contract_sha256"] != contract_hash or receipt["mode"] != mode or receipt["char_cost"] != cost
                    or receipt["results_sha256"] != json_hash(receipt["results"])):
                raise ValueError("Evaluation receipt identity or content hash mismatch")
            result = receipt["results"]
        else:
            result = json_tree(evaluate_rows(state, partitions["test"], channel, old_config, config, mode=mode, char_cost=cost, corpus=corpus))
            receipt = {"contract_sha256": contract_hash, "mode": mode, "char_cost": cost, "results_sha256": json_hash(result), "results": result}
            immutable_json(path, receipt)
        validate_results(result, partitions["test"], corpus, mode=mode, max_window=old_config["active"]["max_window_chars"])
        evaluations[name], hashes[name] = result, sha256(path.read_bytes())
        print(json.dumps({"stage": "evaluation", "condition": name, "samples": len(result)}), flush=True)
    probe_path = output / "shared_observation_probe.json"
    if probe_path.exists():
        probe_receipt = json.loads(probe_path.read_text())
        if probe_receipt["contract_sha256"] != contract_hash or probe_receipt["results_sha256"] != json_hash(probe_receipt["results"]):
            raise ValueError("Shared-observation probe receipt mismatch")
        probe_rows = probe_receipt["results"]
    else:
        probe_rows = json_tree(shared_observation_probe(state, partitions["test"], channel, old_config, config, corpus))
        validate_probe(probe_rows, partitions["test"], corpus, config)
        immutable_json(probe_path, {"contract_sha256": contract_hash, "results_sha256": json_hash(probe_rows), "results": probe_rows})
    validate_probe(probe_rows, partitions["test"], corpus, config)
    if fingerprints() != code:
        raise ValueError("Implementation changed during the frozen experiment")
    summary = {"contract_sha256": contract_hash, "protocol": config["protocol"], "timing_protocol": config["timing_protocol"],
               "evaluation_started_at_utc": start["started_at_utc"], "selection": start["selection"],
               "episodes": sum(map(len, evaluations.values())), "evaluation_receipt_sha256": hashes,
               "shared_observation_probe_sha256": sha256(probe_path.read_bytes()), "shared_observation_probe": aggregate_probe(probe_rows),
               "shared_probe_paired_comparisons": {baseline: paired_probe_bootstrap(probe_rows, baseline, config["bootstrap"])
                                                    for baseline in ("fixed_channel", "fixed_mixture", "null_channel")},
               "languages": {}, "paired_comparisons": {}}
    for language in LANGUAGES:
        summary["languages"][language] = {name: aggregate([row for row in result if row["language"] == language]) for name, result in evaluations.items()}
    for cost in config["char_costs"]:
        for baseline in ("fixed_channel", "fixed_mixture", "null_channel"):
            summary["paired_comparisons"][f"adaptive_minus_{baseline}_cost_{cost:g}"] = paired_bootstrap(
                evaluations[_condition(baseline, cost)], evaluations[_condition("adaptive", cost)], config["bootstrap"])
    immutable_json(output / "summary.json", summary)
    if args.publish:
        destination = Path(args.publish)
        destination.mkdir(parents=True, exist_ok=True)
        if any((destination / name).exists() for name in ("plan.json", "summary.json", "paired_scores.jsonl")):
            raise FileExistsError("Refusing to overwrite published evidence")
        immutable_json(destination / "plan.json", plan)
        immutable_json(destination / "summary.json", summary)
        order = [_condition(mode, cost) for cost in config["char_costs"] for mode in MODES]
        with (destination / "paired_scores.jsonl").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"contract_sha256": contract_hash, "conditions": order,
                                     "columns": ["language", "parallel_id", "document_group", "labels_sha256", *order],
                                     "condition_columns": ["f1", "nonwhitespace_f1", "mean_wait_chars", "model_calls", "prediction_sha256", "episode_elapsed_ms",
                                                           "outcome_forecasts", "outcome_nll_sum", "outcome_brier_sum", "observation_world_updates"]}) + "\n")
            for index, row in enumerate(partitions["test"]):
                first = evaluations[order[0]][index]
                compact = [first[key] for key in ("language", "parallel_id", "document_group", "labels_sha256")]
                for name in order:
                    item = evaluations[name][index]
                    metrics = item["metrics"]
                    compact.append([item["boundary"]["f1"], item["nonwhitespace"]["f1"], previous._wait_metrics(metrics["wait_char_histogram"])["mean_wait_chars"],
                                    metrics["model_calls"], item["prediction_sha256"], item["episode_elapsed_ms"],
                                    *[metrics[key] for key in ("outcome_forecasts", "outcome_nll_sum", "outcome_brier_sum", "observation_world_updates")]])
                handle.write(json.dumps(compact, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "contract_sha256": contract_hash, "episodes": summary["episodes"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/observation-world-pilot.json"))
    parser.add_argument("--source-study")
    parser.add_argument("--sources", default=str(ROOT / "configs/pud-v2.18-sources.json"))
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache/pud"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--publish")
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
