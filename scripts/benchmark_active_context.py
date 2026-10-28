#!/usr/bin/env python3
"""Compare actual context acquisition with bounded streaming on pinned text.

Raw corpus text and learned weights remain in the local cache. A prepared plan
binds code, source files, checkpoint, partition, runtime, and selection settings.
All development choices are saved before any test-mode episode is evaluated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
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

from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.checkpoint import save_checkpoint
from spiralreality_AIT_onepass_aifcore_integrated.integrated.evaluation_metrics import boundary_counts, boundary_targets
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import LANGUAGES, group_parallel_corpus, json_hash, load_pinned_sources, sha256
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT
from spiralreality_AIT_onepass_aifcore_integrated.integrated.streaming_inference import ChunkedStreamingSegmenter

PACKAGE = "spiralreality_AIT_onepass_aifcore_integrated"
MODES = ("fixed_stream", "immediate", "acquisition_only", "active", "frozen_world",
         "full_text_viterbi", "full_text_marginal")
DECODERS = {mode: "viterbi" if mode == "full_text_viterbi" else "marginal_map" for mode in MODES}
BELIEFS = {mode: "learned_channel_posterior" if mode in ("active", "acquisition_only", "frozen_world")
           else "joint_crf_viterbi" if mode == "full_text_viterbi" else "raw_crf_marginals" for mode in MODES}
GAIN_METRICS = ("forecast_evaluations", "expected_information_gain_sum", "realized_entropy_reduction_sum",
                "truncated_acquisitions", "channel_boundary_updates", "window_forced_cuts", "budget_forced_cuts")


def fingerprints():
    paths = sorted((ROOT / PACKAGE).rglob("*.py")) + [Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in paths}


def load_model(state, seed):
    encoder = state["encoder"]
    model = OnePassAIT(latent_dim=state["latent_dim"], seed=seed,
                       encoder_layers=encoder["n_layers"], encoder_heads=encoder["n_heads"],
                       encoder_backend="numpy")
    model.load_state_dict(state)
    return model


def _prefix_pairs(model, rows, config, actions, counters):
    maximum = config["active"]["max_window_chars"]
    initial = config["active"]["initial_chars"]
    stride = config["channel_fit"]["prefix_stride"]
    window_stride = config["channel_fit"]["window_stride"]
    if stride < 1 or window_stride < 1:
        raise ValueError("Channel fitting strides must be positive")
    for row in rows:
        gold = boundary_targets(row.text, row.segments)
        for start in range(0, len(row.text), window_stride):
            window = row.text[start:start + maximum]
            if len(window) < initial + min(actions):
                continue
            counters["windows"] += 1
            observations = {}
            def inspect(length):
                if length not in observations:
                    result = model.student.observe_sequence(window[:length])
                    probabilities = result["boundary_probabilities"]
                    context = result["context_vector"]
                    observations[length] = {
                        "probabilities": list(probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities),
                        "context_vector": list(context.tolist() if hasattr(context, "tolist") else context),
                    }
                    counters["prefix_observations"] += 1
                return observations[length]
            for length in range(initial, len(window) - min(actions) + 1, stride):
                before = inspect(length)
                if len(before["probabilities"]) != length - 1:
                    raise ValueError("Prefix inspector returned inconsistent boundary positions")
                for amount in actions:
                    if length + amount > len(window):
                        continue
                    after = inspect(length + amount)
                    if len(after["probabilities"]) != length + amount - 1:
                        raise ValueError("Extended prefix inspector returned inconsistent positions")
                    yield row, start, length, amount, before, after, gold


def fit_channel(model, rows, config):
    """Fit real same-start prefix extensions and original calibration labels.

    Windows may overlap; support counts are correlated calibration observations,
    not IID independent-sample counts or posterior confidence intervals.
    """
    channel = ContextAcquisitionModel(**config["channel"])
    receipt = {"windows": 0, "prefix_observations": 0, "boundary_pairs": 0,
               "per_language_boundary_pairs": {lang: 0 for lang in LANGUAGES},
               "fit_labels_sha256": json_hash([asdict(row) for row in rows])}
    warm_config = config["world_warmup"]
    maximum_transitions = warm_config["max_transitions"]
    if isinstance(maximum_transitions, bool) or not isinstance(maximum_transitions, int) or maximum_transitions < 1:
        raise ValueError("Warm-up transition cap must be positive")
    rng = np.random.default_rng(warm_config["seed"])
    selected, observed_pairs = {}, Counter()
    for row, start, length, amount, before, after, gold in _prefix_pairs(model, rows, config, channel.action_chars, receipt):
        key = (row.language, row.parallel_id)
        observed_pairs[key] += 1
        if rng.integers(observed_pairs[key]) == 0:
            selected[key] = {"before": before["context_vector"], "after": after["context_vector"], "action_chars": amount}
        for index, probability in enumerate(before["probabilities"]):
            channel.record(probability, after["probabilities"][index], int(gold[start + index]),
                           age=length - index - 1, action_chars=amount)
            receipt["boundary_pairs"] += 1
            receipt["per_language_boundary_pairs"][row.language] += 1
    # At most one observed transition per sentence, then a deterministic global
    # reservoir cap, so long sentences cannot monopolize warm-up replay.
    warm_transitions = []
    for number, transition in enumerate(selected.values(), start=1):
        if len(warm_transitions) < maximum_transitions:
            warm_transitions.append(transition)
        else:
            replace = int(rng.integers(number))
            if replace < maximum_transitions:
                warm_transitions[replace] = transition
    channel.freeze()
    receipt["channel_sha256"] = json_hash(channel.export_state())
    receipt["fit_sentences"] = len(rows)
    receipt["world_warmup_transitions"] = len(warm_transitions)
    receipt["warm_transitions_sha256"] = json_hash(warm_transitions)
    return channel, receipt, warm_transitions


def _row_receipt(rows, corpus):
    return {"groups": sorted({corpus.group_for_document[row.document_id] for row in rows}),
            "documents": sorted({row.document_id for row in rows}),
            "parallel_ids": sorted({row.parallel_id for row in rows}),
            "samples": len(rows), "per_language": {language: sum(row.language == language for row in rows) for language in LANGUAGES},
            "labels_sha256": json_hash([asdict(row) for row in rows])}


def split_development(corpus, partition, seed):
    """Separate channel fitting and cost tuning by original document groups."""
    groups = sorted({corpus.group_for_document[row.document_id] for row in partition["dev"]},
                    key=lambda group: sha256(f"{seed}\0{group}".encode()))
    if len(groups) < 2:
        raise ValueError("Two development document groups are required")
    chosen = set(groups[:(len(groups) + 1) // 2])
    calibration = tuple(row for row in partition["dev"] if corpus.group_for_document[row.document_id] in chosen)
    cost_dev = tuple(row for row in partition["dev"] if corpus.group_for_document[row.document_id] not in chosen)
    receipts = {"channel_fit": _row_receipt(calibration, corpus), "cost_dev": _row_receipt(cost_dev, corpus),
                "segmenter_train": _row_receipt(partition["train"], corpus), "test": _row_receipt(partition["test"], corpus)}
    for name, receipt in receipts.items():
        if any(not receipt["per_language"][language] for language in LANGUAGES):
            raise ValueError("Every calibration/tuning/test role requires all three languages")
        for other, other_receipt in receipts.items():
            if name != other and set(receipt["groups"]) & set(other_receipt["groups"]):
                raise ValueError("Document groups overlap across calibration, tuning, training or test")
    return calibration, cost_dev, {"method": "hash_ordered_development_document_halves_v1", "seed": seed, **receipts}


def evaluate_channel(model, rows, channel, config):
    """Held-out one-step channel calibration, with no count or label updates."""
    from bisect import bisect_right
    counters = {"windows": 0, "prefix_observations": 0}
    cells = {}
    def cell():
        return {"pairs": 0, "eligible_pairs": 0, "predictive_outcome_nll_sum": 0.0,
                "conditional_outcome_nll_sum": 0.0, "before_brier_sum": 0.0, "after_brier_sum": 0.0,
                "before_error_sum": 0, "after_error_sum": 0, "expected_information_gain_sum": 0.0,
                "realized_signed_entropy_reduction_sum": 0.0}
    def entropy(probability):
        return 0.0 if probability in (0, 1) else -probability * math.log(probability) - (1 - probability) * math.log1p(-probability)
    original = channel.export_state()
    for row, start, length, amount, before, after, gold in _prefix_pairs(model, rows, config, channel.action_chars, counters):
        key = f"{row.language}/acquire_{amount}"
        scores = cells.setdefault(key, cell())
        for index, prior in enumerate(before["probabilities"]):
            scores["pairs"] += 1
            estimate = channel.estimate(prior, length - index - 1, amount)
            if not estimate["eligible"]:
                continue
            label = int(gold[start + index])
            observed = after["probabilities"][index]
            outcome = min(len(channel.outcome_edges) - 2, bisect_right(channel.outcome_edges, observed) - 1)
            posterior = estimate["posterior_probabilities"][outcome]
            scores["eligible_pairs"] += 1
            scores["predictive_outcome_nll_sum"] -= math.log(estimate["outcome_probabilities"][outcome])
            scores["conditional_outcome_nll_sum"] -= math.log(estimate["likelihoods"][label][outcome])
            scores["before_brier_sum"] += (prior - label) ** 2
            scores["after_brier_sum"] += (posterior - label) ** 2
            scores["before_error_sum"] += int((prior >= 0.5) != label)
            scores["after_error_sum"] += int((posterior >= 0.5) != label)
            scores["expected_information_gain_sum"] += estimate["information_gain"]
            scores["realized_signed_entropy_reduction_sum"] += entropy(prior) - entropy(posterior)
    if channel.export_state() != original:
        raise ValueError("Held-out channel evaluation mutated the fitted channel")
    for scores in cells.values():
        count = scores["eligible_pairs"]
        scores["support_coverage"] = count / scores["pairs"] if scores["pairs"] else 0.0
        for key in list(scores):
            if key.endswith("_sum"):
                scores[key.removesuffix("_sum")] = scores[key] / count if count else None
    return {"method": "heldout_one_step_prefix_extension_channel_check", "overlapping_pairs_are_correlated": True,
            "channel_sha256": json_hash(original), **counters, "language_actions": cells}


def warm_world(state, transitions, seed, *, action_chars=(4, 16)):
    """Learn one world-model batch from actually observed calibration contexts."""
    from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import acquisition_policy
    model = load_model(state, seed)
    canonical = model.state_dict()
    protected = {key: canonical[key] for key in ("student", "encoder", "phase")}
    initial_steps = model.dynamics.trained_steps
    actions = tuple(action_chars)
    for transition in transitions:
        before = np.asarray(transition["before"], dtype=float)
        after = np.asarray(transition["after"], dtype=float)
        policy = acquisition_policy(transition["action_chars"], actions)
        model.observe_transition(before, policy, before, after,
                                 observation_id=model.last_observation_id + 1,
                                 source="pud_heldout_calibration_prefix_extension", train=False)
    if transitions:
        model.dynamics.train()
    model._phi_hist = {key: [] for key in ("AB", "BC", "CA")}
    model.R3_mix = model.R2_time = 0.0
    warmed = model.state_dict()
    if protected != {key: warmed[key] for key in protected}:
        raise ValueError("World warm-up altered the segmenter weights or local phase basis")
    receipt = {"warm_transitions_sha256": json_hash(transitions), "transitions": len(transitions),
               "world_training_calls": model.dynamics.trained_steps - initial_steps,
               "protected_segmenter_sha256": json_hash(protected),
               "last_observation_id": model.last_observation_id,
               "phase_history_reset": True}
    return warmed, receipt


def _wait_metrics(histogram):
    counts = Counter({int(value): int(count) for value, count in histogram.items()})
    total = sum(counts.values())
    if total == 0:
        return {"mean_wait_chars": 0.0, "p95_wait_chars": 0, "max_wait_chars": 0, "emitted_tokens": 0}
    cumulative, p95 = 0, 0
    for value, count in sorted(counts.items()):
        cumulative += count
        if cumulative >= 0.95 * total:
            p95 = value
            break
    return {"mean_wait_chars": sum(value * count for value, count in counts.items()) / total,
            "p95_wait_chars": p95, "max_wait_chars": max(counts), "emitted_tokens": total}


def _fixed_episode(model, text, config):
    from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import marginal_tokens
    settings = dict(config["fixed_stream"])
    width = settings.pop("feed_chars")
    calls, square_sum, peak = 0, 0, 0
    def inspect(window):
        nonlocal calls, square_sum, peak
        calls += 1
        square_sum += len(window) ** 2
        peak = max(peak, len(window))
        result = model.student.observe_sequence(window)
        return {**result, "tokens": marginal_tokens(window, result["boundary_probabilities"])}
    stream = ChunkedStreamingSegmenter(inspect, **settings)
    tokens, waits, committed, acquisitions = [], Counter(), 0, 0
    for start in range(0, len(text), width):
        end = min(len(text), start + width)
        acquisitions += 1
        emitted = stream.feed(text[start:end])
        for token in emitted:
            committed += len(token)
            waits[end - committed] += 1
        tokens.extend(emitted)
    for token in stream.flush():
        committed += len(token)
        waits[len(text) - committed] += 1
        tokens.append(token)
    return {"tokens": tokens, "decoder": "marginal_map", "metrics": {"acquire_calls": acquisitions, "acquired_chars": len(text),
            "deliberate_acquire_calls": 0, "commit_actions": len(tokens), "forced_splits": stream.forced_split_count,
            "model_calls": calls, "window_square_sum": square_sum, "max_window_chars_seen": peak,
            "wait_char_histogram": dict(waits), "policy_counts": {"fixed_feed": acquisitions},
            "world_model_updates": 0, **{name: stream.forced_split_count if name == "window_forced_cuts" else 0 for name in GAIN_METRICS}}, "trace": []}


def run_episode(model, text, channel, config, *, mode, char_cost):
    if mode not in MODES:
        raise ValueError("Unknown comparison mode")
    if mode == "fixed_stream":
        return _fixed_episode(model, text, config)
    if mode in ("full_text_viterbi", "full_text_marginal"):
        result = model.student.observe_sequence(text)
        tokens = result["tokens"]
        if mode == "full_text_marginal":
            from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import marginal_tokens
            tokens = marginal_tokens(text, result["boundary_probabilities"])
        waits, end = Counter(), 0
        for token in tokens:
            end += len(token)
            waits[len(text) - end] += 1
        return {"tokens": tokens, "decoder": DECODERS[mode], "metrics": {"acquire_calls": 1 if text else 0, "acquired_chars": len(text),
                "deliberate_acquire_calls": 0, "commit_actions": 1 if text else 0, "forced_splits": 0,
                "model_calls": 1, "window_square_sum": len(text) ** 2, "max_window_chars_seen": len(text),
                "wait_char_histogram": dict(waits), "policy_counts": {"full_text": 1}, "world_model_updates": 0,
                **{name: 0 for name in GAIN_METRICS}}, "trace": []}
    from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import (
        ActiveContextConfig, ActiveTextSession, StringTextSource,
    )
    values = {**config["active"], "char_cost": char_cost}
    if mode in ("immediate", "acquisition_only"):
        values.update(phase_gain=0.0, surprise_gain=0.0)
    if mode in ("immediate", "acquisition_only", "frozen_world"):
        values["learn_dynamics"] = False
    if mode == "immediate":
        channel = ContextAcquisitionModel(**config["channel"]).freeze()
    return ActiveTextSession(model, channel, config=ActiveContextConfig(**values)).run(StringTextSource(text))


def evaluate_rows(state, rows, channel, config, *, mode, char_cost, corpus):
    results = []
    for row in rows:
        model = load_model(state, config["model_seed"])
        started = time.perf_counter()
        episode = run_episode(model, row.text, channel, config, mode=mode, char_cost=char_cost)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if "".join(episode["tokens"]) != row.text:
            raise ValueError("An episode failed to reconstruct the source text")
        if episode.get("decoder") != DECODERS[mode]:
            raise ValueError("Episode decoder differs from the matched comparison protocol")
        if mode not in ("fixed_stream", "full_text_viterbi", "full_text_marginal") and not episode.get("done", False):
            raise ValueError("Active context episode did not complete")
        metrics = episode["metrics"]
        required = {"acquire_calls", "acquired_chars", "deliberate_acquire_calls", "commit_actions", "forced_splits",
                    "model_calls", "window_square_sum", "max_window_chars_seen", "wait_char_histogram",
                    "policy_counts", "world_model_updates"}
        required.update(GAIN_METRICS)
        if not required <= metrics.keys():
            raise ValueError(f"Episode metrics missing: {sorted(required - metrics.keys())}")
        if metrics["acquired_chars"] != len(row.text):
            raise ValueError("Episode acquisition did not match the observed source length")
        if not mode.startswith("full_text_") and metrics["max_window_chars_seen"] > config["active"]["max_window_chars"]:
            raise ValueError("Episode exceeded the configured inference window")
        results.append({"language": row.language, "parallel_id": row.parallel_id,
                        "document_group": corpus.group_for_document[row.document_id],
                        "characters": len(row.text), "labels_sha256": json_hash(row.segments),
                        "episode_elapsed_ms": elapsed_ms,
                        "decoder": episode["decoder"], "belief_model": BELIEFS[mode],
                        "prediction_sha256": json_hash(episode["tokens"]),
                        "boundary": boundary_counts(row.text, row.segments, episode["tokens"]),
                        "nonwhitespace": boundary_counts(row.text, row.segments, episode["tokens"], exclude_whitespace=True),
                        "metrics": metrics, "trace": episode["trace"]})
    return results


def aggregate(rows):
    if not rows:
        raise ValueError("Nonempty evaluation rows are required")
    waits, policies = Counter(), Counter()
    for row in rows:
        waits.update({int(key): count for key, count in row["metrics"]["wait_char_histogram"].items()})
        policies.update(row["metrics"]["policy_counts"])
    characters = sum(row["characters"] for row in rows)
    decoders = {row["decoder"] for row in rows}
    beliefs = {row["belief_model"] for row in rows}
    if len(decoders) != 1 or len(beliefs) != 1:
        raise ValueError("Aggregation requires one declared decoder and belief model")
    result = {"samples": len(rows), "characters": characters, "decoder": next(iter(decoders)), "belief_model": next(iter(beliefs)),
              "macro_f1": float(np.mean([row["boundary"]["f1"] for row in rows])),
              "nonwhitespace_macro_f1": float(np.mean([row["nonwhitespace"]["f1"] for row in rows])),
              **_wait_metrics(waits), "policy_counts": dict(policies)}
    for key in ("acquire_calls", "acquired_chars", "deliberate_acquire_calls", "commit_actions", "forced_splits",
                "model_calls", "window_square_sum", "world_model_updates", *GAIN_METRICS):
        result[key] = sum(row["metrics"][key] for row in rows)
    count = result["forecast_evaluations"]
    result["mean_matched_expected_information_gain"] = result["expected_information_gain_sum"] / count if count else None
    result["mean_matched_realized_entropy_reduction"] = result["realized_entropy_reduction_sum"] / count if count else None
    result["model_calls_per_1k_chars"] = result["model_calls"] * 1000 / characters
    result["window_square_sum_per_char"] = result["window_square_sum"] / characters
    result["max_window_chars_seen"] = max(row["metrics"]["max_window_chars_seen"] for row in rows)
    timings = [float(row.get("episode_elapsed_ms", 0.0)) for row in rows]
    if any(not math.isfinite(value) or value < 0 for value in timings):
        raise ValueError("Episode elapsed times must be finite and nonnegative")
    result.update(timed_samples=sum("episode_elapsed_ms" in row for row in rows),
                  median_episode_ms=float(np.median(timings)), p95_episode_ms=float(np.quantile(timings, 0.95)),
                  total_episode_seconds=math.fsum(timings) / 1000)
    return result


def development_utility(rows, weights):
    by_language = {language: aggregate([row for row in rows if row["language"] == language]) for language in LANGUAGES}
    terms = {"macro_f1": float(np.mean([row["macro_f1"] for row in by_language.values()])),
             "mean_wait_chars": float(np.mean([row["mean_wait_chars"] for row in by_language.values()])),
             "model_calls_per_1k_chars": float(np.mean([row["model_calls_per_1k_chars"] for row in by_language.values()]))}
    value = terms["macro_f1"] - weights["mean_wait_char_weight"] * terms["mean_wait_chars"] - weights["model_calls_per_1k_char_weight"] * terms["model_calls_per_1k_chars"]
    return {"utility": value, **terms, "per_language": by_language}


def choose_on_development(candidates):
    if not candidates or any(not math.isfinite(row["development"]["utility"]) for row in candidates):
        raise ValueError("Finite development utilities are required")
    chosen = max(candidates, key=lambda row: (row["development"]["utility"], row["char_cost"]))
    return {"char_cost": chosen["char_cost"], "development_utility": chosen["development"]["utility"],
            "tie_break": "larger character cost on exact utility ties"}


def _save_immutable(path, payload):
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Saved receipt differs: {path.name}")
    else:
        save_checkpoint(path, payload)


def run(args):
    config = json.loads(Path(args.config).read_text())
    costs = config["char_cost_candidates"]
    if not costs or len(set(costs)) != len(costs) or any(isinstance(cost, bool) or not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("Declare distinct finite nonnegative character costs")
    weights = config["development_utility"]
    if set(weights) != {"mean_wait_char_weight", "model_calls_per_1k_char_weight"} or any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Development utility weights must be finite and nonnegative")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else ROOT / config["checkpoint"]
    checkpoint_bytes = checkpoint_path.read_bytes()
    if sha256(checkpoint_bytes) != config["checkpoint_sha256"]:
        raise ValueError("Pinned segmenter checkpoint integrity mismatch")
    state = json.loads(checkpoint_bytes)
    rows, sources = load_pinned_sources(args.sources, args.cache_dir, download=args.download)
    corpus = group_parallel_corpus(rows, seed=config["partition_seed"], folds=config["fold_count"], max_chars=config["max_chars"])
    partition, partition_receipt = corpus.partition(config["test_fold"])
    channel_rows, cost_rows, calibration_receipt = split_development(corpus, partition, config["development_split_seed"])
    code = fingerprints()
    contract = {"config": config, "sources": sources, "partition": corpus.receipt,
                "selection_partition": partition_receipt, "checkpoint_sha256": sha256(checkpoint_bytes),
                "calibration_partition": calibration_receipt,
                "code_sha256": code, "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                                                 "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
                "limitations": [config["protocol"], "Overlapping training prefixes are correlated calibration observations.",
                                "World state resets per sentence; no gold labels are delivered to runtime sessions.",
                                "Information gain is expected under a fitted observation channel; realized entropy can rise."]}
    contract_hash = json_hash(contract)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan = {"contract_sha256": contract_hash, "contract": contract}
    _save_immutable(output / "plan.json", plan)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "contract_sha256": contract_hash,
                          "partition_sizes": {key: value["per_language"] for key, value in calibration_receipt.items() if isinstance(value, dict) and "per_language" in value}}), flush=True)
        return
    channel_path = output / "channel.json"
    channel_receipt_path = output / "channel_fit.json"
    if channel_path.exists() or channel_receipt_path.exists():
        channel_state = json.loads(channel_path.read_text())
        fit_receipt = json.loads(channel_receipt_path.read_text())
        if fit_receipt["contract_sha256"] != contract_hash or fit_receipt["channel_sha256"] != json_hash(channel_state):
            raise ValueError("Completed acquisition channel integrity mismatch")
        channel = ContextAcquisitionModel.from_state(channel_state).freeze()
        warm_transition_receipt = json.loads((output / "warm_transitions.json").read_text())
        warm_transitions = warm_transition_receipt["transitions"]
        if (warm_transition_receipt["contract_sha256"] != contract_hash
                or json_hash(warm_transitions) != fit_receipt["warm_transitions_sha256"]):
            raise ValueError("Warm-up transition receipt integrity mismatch")
    else:
        print(json.dumps({"stage": "fit_channel"}), flush=True)
        channel, fit_receipt, warm_transitions = fit_channel(load_model(state, config["model_seed"]), channel_rows, config)
        fit_receipt["contract_sha256"] = contract_hash
        save_checkpoint(output / "warm_transitions.json", {"contract_sha256": contract_hash, "transitions": warm_transitions})
        save_checkpoint(channel_path, channel.export_state())
        save_checkpoint(channel_receipt_path, fit_receipt)
    warm_path, warm_receipt_path = output / "warm_started_model.json", output / "warm_start.json"
    if warm_path.exists() or warm_receipt_path.exists():
        warm_bytes = warm_path.read_bytes()
        warm_receipt = json.loads(warm_receipt_path.read_text())
        if (warm_receipt["contract_sha256"] != contract_hash
                or warm_receipt["checkpoint_sha256"] != sha256(warm_bytes)
                or warm_receipt["warm_transitions_sha256"] != fit_receipt["warm_transitions_sha256"]):
            raise ValueError("Warm-start checkpoint integrity mismatch")
        episode_state = json.loads(warm_bytes)
    else:
        print(json.dumps({"stage": "warm_world", "transitions": len(warm_transitions)}), flush=True)
        episode_state, warm_receipt = warm_world(state, warm_transitions, config["model_seed"], action_chars=config["active"]["action_chars"])
        save_checkpoint(warm_path, episode_state)
        warm_receipt.update(contract_sha256=contract_hash, checkpoint_sha256=sha256(warm_path.read_bytes()))
        save_checkpoint(warm_receipt_path, warm_receipt)
    channel_check_path = output / "heldout_channel_check.json"
    if channel_check_path.exists():
        channel_check = json.loads(channel_check_path.read_text())
        if channel_check["contract_sha256"] != contract_hash or channel_check["channel_sha256"] != fit_receipt["channel_sha256"]:
            raise ValueError("Held-out channel check integrity mismatch")
    else:
        channel_check = evaluate_channel(load_model(episode_state, config["model_seed"]), cost_rows, channel, config)
        channel_check["contract_sha256"] = contract_hash
        save_checkpoint(channel_check_path, channel_check)
    candidates = []
    for cost in costs:
        path = output / "development" / f"cost_{cost:g}.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt["contract_sha256"] != contract_hash or receipt["char_cost"] != cost:
                raise ValueError("Development receipt identity mismatch")
        else:
            result = evaluate_rows(episode_state, cost_rows, channel, config, mode="active", char_cost=cost, corpus=corpus)
            receipt = {"contract_sha256": contract_hash, "char_cost": cost, "development": development_utility(result, weights), "results": result}
            save_checkpoint(path, receipt)
        candidates.append({key: receipt[key] for key in ("char_cost", "development")})
        print(json.dumps({"stage": "development", "char_cost": cost, "utility": receipt["development"]["utility"]}), flush=True)
    selection = {"contract_sha256": contract_hash, "candidates": candidates, **choose_on_development(candidates)}
    _save_immutable(output / "development_selection.json", selection)
    evaluations = {}
    for mode in MODES:
        path = output / "evaluation" / f"{mode}.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt["contract_sha256"] != contract_hash or receipt["mode"] != mode or receipt["char_cost"] != selection["char_cost"]:
                raise ValueError("Test receipt identity mismatch")
        else:
            result = evaluate_rows(episode_state, partition["test"], channel, config, mode=mode, char_cost=selection["char_cost"], corpus=corpus)
            receipt = {"contract_sha256": contract_hash, "mode": mode, "char_cost": selection["char_cost"], "results": result}
            save_checkpoint(path, receipt)
        evaluations[mode] = receipt["results"]
        print(json.dumps({"stage": "test", "mode": mode, "samples": len(receipt["results"])}), flush=True)
    if code != fingerprints():
        raise ValueError("Implementation changed during the study")
    summary = {"contract_sha256": contract_hash, "protocol": config["protocol"], "selection": selection,
               "timing_protocol": config["timing_protocol"],
               "channel_fit": fit_receipt, "heldout_channel_check": channel_check, "world_warmup": warm_receipt, "languages": {},
               "evaluation_receipt_sha256": {mode: sha256((output / "evaluation" / f"{mode}.json").read_bytes()) for mode in MODES},
               "development_receipt_sha256": {f"cost_{cost:g}.json": sha256((output / "development" / f"cost_{cost:g}.json").read_bytes()) for cost in costs}}
    compact = []
    for language in LANGUAGES:
        summary["languages"][language] = {mode: aggregate([row for row in rows if row["language"] == language]) for mode, rows in evaluations.items()}
        by_mode = {mode: {row["parallel_id"]: row for row in rows if row["language"] == language} for mode, rows in evaluations.items()}
        keys = by_mode[MODES[0]].keys()
        if any(by_mode[mode].keys() != keys for mode in MODES):
            raise ValueError("Comparison modes have unmatched test rows")
        for parallel_id in sorted(keys):
            first = by_mode[MODES[0]][parallel_id]
            compact.append([language, parallel_id, first["document_group"], first["labels_sha256"],
                            *[[by_mode[mode][parallel_id]["boundary"]["f1"], by_mode[mode][parallel_id]["nonwhitespace"]["f1"],
                               _wait_metrics(by_mode[mode][parallel_id]["metrics"]["wait_char_histogram"])["mean_wait_chars"],
                               by_mode[mode][parallel_id]["metrics"]["model_calls"], by_mode[mode][parallel_id]["prediction_sha256"]] for mode in MODES]])
    save_checkpoint(output / "summary.json", summary)
    if args.publish:
        destination = Path(args.publish)
        destination.mkdir(parents=True, exist_ok=True)
        names = ("plan.json", "summary.json", "paired_scores.jsonl")
        if any((destination / name).exists() for name in names):
            raise FileExistsError("Refusing to overwrite published evidence")
        save_checkpoint(destination / "plan.json", plan)
        save_checkpoint(destination / "summary.json", summary)
        with (destination / "paired_scores.jsonl").open("w") as handle:
            handle.write(json.dumps({"contract_sha256": contract_hash, "columns": ["language", "parallel_id", "document_group", "labels_sha256", *MODES],
                                     "mode_columns": ["f1", "nonwhitespace_f1", "mean_wait_chars", "model_calls", "prediction_sha256"]}) + "\n")
            for row in compact:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps({"complete": True, "contract_sha256": contract_hash, "selected_char_cost": selection["char_cost"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/active-context-pilot.json"))
    parser.add_argument("--sources", default=str(ROOT / "configs/pud-v2.18-sources.json"))
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache/pud"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--publish")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
