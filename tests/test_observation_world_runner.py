from argparse import Namespace
import json
from types import SimpleNamespace

import pytest

from scripts import benchmark_observation_world as runner
from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import ExternalSentence, group_parallel_corpus, json_hash, sha256
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def config():
    return json.loads((runner.ROOT / "configs/observation-world-pilot.json").read_text())


def prior_config():
    result = json.loads((runner.ROOT / "configs/active-context-pilot.json").read_text())
    result.update(fold_count=3, model_seed=19)
    result["active"].update(max_window_chars=24, context_chars=8, trace_limit=16)
    return result


def channel():
    result = ContextAcquisitionModel(probability_edges=(0, 1), age_edges=(0,), outcome_edges=(0, 1), min_per_label=1)
    for action in result.action_chars:
        for label in (0, 1):
            result.record(0.4, 0.6, label, 2, action)
    return result.freeze()


def fixture_study(tmp_path, monkeypatch):
    settings, old_config = config(), prior_config()
    rows = [ExternalSentence(language, f"doc{index:02d}", f"sentence{index:02d}", f"parallel{index:02d}",
                             f"{language}-abcdefgh-{index:02d}", (f"{language}-", "abcdefgh", f"-{index:02d}"))
            for index in range(12) for language in ("en", "ja", "zh")]
    corpus = group_parallel_corpus(rows, seed=old_config["partition_seed"], folds=old_config["fold_count"], max_chars=old_config["max_chars"])
    partitions, receipt = corpus.partition(old_config["test_fold"])
    sources = {"fixture": "local-only synthetic text"}
    historical = {"config": old_config, "sources": sources, "partition": corpus.receipt,
                  "selection_partition": receipt, "code_sha256": {"historical.py": "unchanged-historical-hash"}}
    historical_hash = json_hash(historical)
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19)
    state = model.state_dict()
    protected = {name: state[name] for name in ("student", "encoder", "phase")}
    channel_state = channel().export_state()
    warm_data = json.dumps(state).encode()
    warm_hash = sha256(warm_data)
    artifacts = {"plan.json": {"contract_sha256": historical_hash, "contract": historical}, "channel.json": channel_state,
                 "channel_fit.json": {"contract_sha256": historical_hash, "channel_sha256": json_hash(channel_state), "warm_transitions_sha256": "fixture"},
                 "warm_start.json": {"contract_sha256": historical_hash, "checkpoint_sha256": warm_hash, "warm_transitions_sha256": "fixture",
                                     "protected_segmenter_sha256": json_hash(protected)}, "warm_started_model.json": state}
    source = tmp_path / "historical"
    source.mkdir()
    for name, value in artifacts.items():
        (source / name).write_text(json.dumps(value))
    settings.update(source_contract_sha256=historical_hash, source_study=str(source), char_costs=[0.002])
    settings["source_artifact_sha256"] = {name: sha256((source / name).read_bytes()) for name in artifacts}
    settings["bootstrap"]["replicates"] = 20
    settings["probe"]["max_window_chars"] = 24
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps(settings))
    monkeypatch.setattr(runner, "load_pinned_sources", lambda *a, **kw: (rows, sources))
    args = Namespace(config=config_path, source_study=source, sources="unused", cache_dir="unused", output=tmp_path / "results",
                     prepare_only=True, publish=None)
    return args, settings, partitions, corpus


def test_modes_keep_task_controls_equal_and_change_only_observation_hypothesis():
    settings = config()
    values = {mode: runner.mode_values(mode, settings) for mode in runner.MODES}
    for row in values.values():
        assert row["learn_dynamics"] is False
        assert row["phase_gain"] == row["surprise_gain"] == row["model_epistemic_weight"] == 0
        assert row["epistemic_weight"] == 0.05
    assert values["fixed_channel"]["observation_model"] == "fixed_channel"
    assert values["fixed_mixture"]["learn_observation_regime"] is False
    assert values["adaptive"]["learn_observation_regime"] is True
    assert values["null_channel"]["informative_prior"] == 0
    assert values["adaptive"]["informative_prior"] == values["fixed_mixture"]["informative_prior"] == 0.5
    with pytest.raises(ValueError, match="Unknown"):
        runner.mode_values("choose_using_test", settings)


def test_source_study_pins_historical_weights_without_claiming_code_matches(tmp_path, monkeypatch):
    args, settings, _, _ = fixture_study(tmp_path, monkeypatch)
    artifacts, loaded = runner.load_source_study(args.source_study, settings)
    assert loaded.frozen
    assert artifacts["plan.json"]["contract"]["code_sha256"] != runner.fingerprints()
    original = (args.source_study / "warm_started_model.json").read_text()
    (args.source_study / "warm_started_model.json").write_text(original + " ")
    with pytest.raises(ValueError, match="artifact integrity.*warm_started_model"):
        runner.load_source_study(args.source_study, settings)


def test_all_forecasts_precede_common_observation_and_labels_only_change_scoring(monkeypatch):
    calls = []
    raw_channel = ContextAcquisitionModel(probability_edges=(0, 1), age_edges=(0,), outcome_edges=(0, 0.5, 1), min_per_label=1)
    for action in (4, 16):
        for _ in range(5):
            raw_channel.record(0.2, 0.1, 0, 1, action)
            raw_channel.record(0.2, 0.9, 1, 1, action)
    raw_channel.freeze()
    original_estimate = ContextAcquisitionModel.estimate
    def estimate(*args, **kwargs):
        calls.append("forecast")
        return original_estimate(*args, **kwargs)
    monkeypatch.setattr(ContextAcquisitionModel, "estimate", estimate)
    def observe(text):
        calls.append("observe")
        return {"boundary_probabilities": [0.2 if len(text) == 8 else 0.9] * (len(text) - 1)}
    monkeypatch.setattr(runner.previous, "load_model", lambda *args: SimpleNamespace(student=SimpleNamespace(observe_sequence=observe)))
    settings = config()
    settings["probe"]["max_window_chars"] = 32
    text = "a" * 32
    a = ExternalSentence("en", "document", "sentence", "parallel", text, (text,))
    b = ExternalSentence("en", "document", "sentence", "parallel", text, ("a", "a" * 31))
    corpus = SimpleNamespace(group_for_document={"document": "document"})
    first = runner.shared_observation_probe({}, [a], raw_channel, prior_config(), settings, corpus)[0]
    assert calls[:6] == ["observe", "forecast", "forecast", "forecast", "forecast", "observe"]
    second = runner.shared_observation_probe({}, [b], raw_channel, prior_config(), settings, corpus)[0]
    assert first["focus_offset"] == second["focus_offset"] == 1
    assert first["steps"] == second["steps"] == 3
    for mode in runner.MODES:
        before, after = first["modes"][mode], second["modes"][mode]
        assert before["forecasts"] == after["forecasts"] == 3
        for key in ("outcome_nll_sum", "outcome_brier_sum", "regime_information_gain_sum"):
            assert before[key] == after[key]
        assert before["label_brier_sum"] != after["label_brier_sum"]
    assert first["modes"]["fixed_mixture"]["final_informative_probability"] == pytest.approx(0.5)
    assert first["modes"]["null_channel"]["final_informative_probability"] == 0


def test_paired_bootstrap_resamples_document_groups_jointly_and_keeps_parallel_languages():
    before, after = [], []
    for group in range(3):
        for language in ("en", "ja", "zh"):
            row = {"language": language, "parallel_id": f"p{group}", "document_group": f"d{group}",
                   "labels_sha256": "same", "prediction_sha256": "before", "boundary": {"f1": 0.4}, "nonwhitespace": {"f1": 0.3}}
            before.append(row)
            after.append({**row, "prediction_sha256": "after", "boundary": {"f1": 0.5}, "nonwhitespace": {"f1": 0.5}})
    result = runner.paired_bootstrap(before, after, {"seed": 19, "replicates": 50, "confidence": 0.95})
    assert result["groups"] == 3
    assert result["prediction_changed_sentences"] == {"en": 3, "ja": 3, "zh": 3}
    for language in result["languages"].values():
        assert language["macro_f1"]["delta"] == pytest.approx(0.1)
        assert language["macro_f1"]["interval"] == pytest.approx([0.1, 0.1])
    with pytest.raises(ValueError, match="matching sentence"):
        runner.paired_bootstrap(before[:-1], after, {"seed": 19, "replicates": 50, "confidence": 0.95})


def test_common_probe_bootstrap_scores_identical_forecasts_and_preserves_null_results():
    rows = []
    for document in range(3):
        for language in ("en", "ja", "zh"):
            rows.append({"language": language, "document_group": f"d{document}", "modes": {
                "fixed_mixture": {"forecasts": 2, "outcome_nll_sum": 2.0, "outcome_brier_sum": 1.0},
                "adaptive": {"forecasts": 2, "outcome_nll_sum": 1.5, "outcome_brier_sum": 1.0}}})
    settings = {"seed": 19, "replicates": 20, "confidence": 0.95}
    result = runner.paired_probe_bootstrap(rows, "fixed_mixture", settings)
    for row in result["languages"].values():
        assert row["forecasts"] == 6
        assert row["outcome_nll"]["delta"] == -0.25
        assert row["outcome_brier"]["delta"] == 0
        assert row["outcome_brier"]["interval"] == [0, 0]
    rows[0]["modes"]["adaptive"]["forecasts"] = 1
    with pytest.raises(ValueError, match="different outcome counts"):
        runner.paired_probe_bootstrap(rows, "fixed_mixture", settings)


def test_episode_timing_excludes_restore_and_score_and_freezes_old_world(monkeypatch):
    state = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19).state_dict()
    clock = [0.0]
    original = runner.previous.load_model
    def restore(*args):
        clock[0] += 100
        return original(*args)
    monkeypatch.setattr(runner.previous, "load_model", restore)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    real_episode = runner.run_episode
    def timed(*args, **kwargs):
        value = real_episode(*args, **kwargs)
        clock[0] += 0.025
        return value
    monkeypatch.setattr(runner, "run_episode", timed)
    row = ExternalSentence("en", "document", "sentence", "parallel", "abcdefghijklmnop", ("abcd", "efgh", "ijkl", "mnop"))
    corpus = SimpleNamespace(group_for_document={"document": "document"})
    result = runner.evaluate_rows(state, [row], channel(), prior_config(), config(), mode="adaptive", char_cost=0.002, corpus=corpus)
    assert result[0]["episode_elapsed_ms"] == pytest.approx(25)
    assert result[0]["metrics"]["world_model_updates"] == 0
    assert result[0]["decoder"] == "marginal_map"


def test_offline_run_freezes_plan_before_evaluation_verifies_full_rows_and_resumes(tmp_path, monkeypatch):
    args, settings, partitions, corpus = fixture_study(tmp_path, monkeypatch)
    runner.run(args)
    assert (args.output / "plan.json").exists()
    assert not (args.output / "evaluation_start.json").exists()
    plan = json.loads((args.output / "plan.json").read_text())
    assert plan["contract"]["config"]["char_costs"] == [0.002]
    assert plan["contract"]["historical_source_code_sha256"] != plan["contract"]["current_code_sha256"]
    original = runner.evaluate_rows
    def verify_started(*a, **kw):
        assert (args.output / "evaluation_start.json").exists()
        result = original(*a, **kw)
        # Multi-digit integer keys sort differently after JSON key conversion.
        # This reproduces the actual first-run resume failure, rather than only
        # exercising short fixtures with single-digit waiting histograms.
        for row in result:
            row["metrics"]["wait_char_histogram"] = {2: 1, 10: 1}
            row["metrics"]["realized_entropy_reduction_sum"] = -0.2
        return result
    monkeypatch.setattr(runner, "evaluate_rows", verify_started)
    args.prepare_only = False
    args.publish = tmp_path / "published"
    runner.run(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["episodes"] == len(partitions["test"]) * 4
    assert len(summary["evaluation_receipt_sha256"]) == 4
    assert len(summary["paired_comparisons"]) == 3
    assert "none" in summary["selection"]
    assert len((args.publish / "paired_scores.jsonl").read_text().splitlines()) == len(partitions["test"]) + 1
    public_text = "".join(path.read_text() for path in args.publish.iterdir())
    assert not any(row.text in public_text for row in partitions["test"])
    for path in (args.output / "evaluation").iterdir():
        receipt = json.loads(path.read_text())
        assert receipt["results_sha256"] == json_hash(receipt["results"])
        assert receipt["results"][0]["metrics"]["wait_char_histogram"] == {"2": 1, "10": 1}
        assert receipt["results"][0]["metrics"]["realized_entropy_reduction_sum"] == -0.2
    monkeypatch.setattr(runner, "evaluate_rows", lambda *a, **kw: pytest.fail("unexpected completed episode rerun"))
    monkeypatch.setattr(runner, "shared_observation_probe", lambda *a, **kw: pytest.fail("unexpected completed probe rerun"))
    args.publish = None
    runner.run(args)
    assert json.loads((args.output / "summary.json").read_text()) == summary
    probe_path = args.output / "shared_observation_probe.json"
    probe = json.loads(probe_path.read_text())
    original_probe = probe_path.read_text()
    del probe["results"][0]["modes"]["adaptive"]
    probe["results_sha256"] = json_hash(probe["results"])
    probe_path.write_text(json.dumps(probe))
    with pytest.raises(ValueError, match="all four mode totals"):
        runner.run(args)
    probe_path.write_text(original_probe)
    path = next((args.output / "evaluation").iterdir())
    receipt = json.loads(path.read_text())
    receipt["results"] = receipt["results"][:-1]
    receipt["results_sha256"] = json_hash(receipt["results"])
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        runner.run(args)


@pytest.mark.parametrize("key,value", [("char_costs", [float("nan")]), ("char_costs", [True]), ("char_costs", [0.002, 0.002]), ("model_epistemic_weight", 0.1)])
def test_invalid_or_posthoc_cost_contracts_reject(key, value):
    settings = config()
    settings[key] = value
    with pytest.raises(ValueError):
        runner.validate_config(settings)


def test_immutable_results_refuse_overwrite(tmp_path):
    path = tmp_path / "plan.json"
    runner.immutable_json(path, {"value": 1})
    runner.immutable_json(path, {"value": 1})
    with pytest.raises(ValueError, match="Immutable evidence differs"):
        runner.immutable_json(path, {"value": 2})


def test_json_tree_normalization_preserves_hashes_after_integer_key_conversion(tmp_path):
    original = {"histogram": {2: 1, 10: 3}, "signed_entropy": -0.5}
    normalized = runner.json_tree(original)
    assert normalized["histogram"] == {"2": 1, "10": 3}
    assert json_hash(normalized) == json_hash(json.loads(json.dumps(normalized)))
    assert json_hash(original) != json_hash(normalized)
    path = tmp_path / "receipt.json"
    runner.immutable_json(path, original)
    runner.immutable_json(path, normalized)
    with pytest.raises(ValueError, match="Out of range float"):
        runner.immutable_json(tmp_path / "invalid.json", {"metric": float("nan")})
    assert not (tmp_path / "invalid.json").exists()
