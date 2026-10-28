from argparse import Namespace
import json

import numpy as np
import pytest

from scripts import benchmark_active_context as runner
from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import ExternalSentence, sha256
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig


def config():
    return json.loads((runner.ROOT / "configs/active-context-pilot.json").read_text())


def test_acquisition_fit_pairs_keep_the_same_absolute_boundary_and_use_gold(monkeypatch):
    settings = config()
    settings["channel"].update(action_chars=[2, 4], min_per_label=1)
    settings["active"].update(initial_chars=4, action_chars=[2, 4], max_window_chars=12)
    settings["channel_fit"].update(prefix_stride=4, window_stride=12)
    observed, records = [], []
    class Inspector:
        def observe_sequence(self, text):
            observed.append(text)
            return {"boundary_probabilities": [index / 100 + len(text) / 1000 for index in range(len(text) - 1)],
                    "context_vector": np.array([len(text), 0, 0, 0], dtype=float)}
    class Model:
        student = Inspector()
    class RecordedChannel(ContextAcquisitionModel):
        def record(self, *args, **kwargs):
            records.append((args, kwargs))
            return super().record(*args, **kwargs)
    monkeypatch.setattr(runner, "ContextAcquisitionModel", RecordedChannel)
    row = ExternalSentence("en", "document", "sentence", "parallel", "abcdefghijkl",
                           ("ab", "cd", "ef", "gh", "ij", "kl"))
    channel, receipt, transitions = runner.fit_channel(Model(), [row], settings)
    assert channel.frozen
    assert receipt["boundary_pairs"] == channel.observations == 20
    assert receipt["prefix_observations"] == 5
    assert set(observed) == {row.text[:length] for length in (4, 6, 8, 10, 12)}
    expected = []
    for length in (4, 8):
        for amount in (2, 4):
            for index in range(length - 1):
                expected.append(((index / 100 + length / 1000, index / 100 + (length + amount) / 1000,
                                  int(index % 2 == 1)), {"age": length - index - 1, "action_chars": amount}))
    assert records == expected
    assert len(transitions) == 1
    assert transitions[0]["after"][0] > transitions[0]["before"][0]
    assert receipt["warm_transitions_sha256"] == runner.json_hash(transitions)


def test_development_selection_uses_only_declared_utility_and_ties_reduce_spending():
    candidates = [{"char_cost": 0.0005, "development": {"utility": 0.8}, "test_f1": 1.0},
                  {"char_cost": 0.01, "development": {"utility": 0.8}, "test_f1": 0.0}]
    assert runner.choose_on_development(candidates)["char_cost"] == 0.01
    candidates[0]["development"]["utility"] = 0.81
    assert runner.choose_on_development(candidates)["char_cost"] == 0.0005
    with pytest.raises(ValueError, match="Finite development"):
        runner.choose_on_development([])


def test_wait_statistics_pool_tokens_rather_than_average_sentence_means():
    assert runner._wait_metrics({0: 9, 10: 1}) == {
        "mean_wait_chars": 1, "p95_wait_chars": 10, "max_wait_chars": 10, "emitted_tokens": 10}
    assert runner._wait_metrics({})["mean_wait_chars"] == 0


def test_episode_timing_excludes_common_model_restore_and_reports_elapsed_distribution(monkeypatch):
    clock = [0.0]
    def restore(*args):
        clock[0] += 100.0
        return object()
    metrics = {name: 0 for name in ("acquire_calls", "acquired_chars", "deliberate_acquire_calls", "commit_actions",
               "forced_splits", "model_calls", "window_square_sum", "max_window_chars_seen", "world_model_updates", *runner.GAIN_METRICS)}
    metrics.update(acquire_calls=1, acquired_chars=4, commit_actions=1, model_calls=1,
                   window_square_sum=16, max_window_chars_seen=4, wait_char_histogram={0: 2}, policy_counts={"full_text": 1})
    def episode(*args, **kwargs):
        clock[0] += 0.025
        return {"tokens": ["ab", "cd"], "decoder": "viterbi", "metrics": metrics, "trace": []}
    monkeypatch.setattr(runner, "load_model", restore)
    monkeypatch.setattr(runner, "run_episode", episode)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    row = ExternalSentence("en", "document", "sentence", "parallel", "abcd", ("ab", "cd"))
    class Corpus:
        group_for_document = {"document": "document"}
    result = runner.evaluate_rows({}, [row], None, config(), mode="full_text_viterbi", char_cost=0.002, corpus=Corpus())[0]
    assert result["episode_elapsed_ms"] == pytest.approx(25)
    summary = runner.aggregate([result, {**result, "episode_elapsed_ms": 75.0}])
    assert summary["timed_samples"] == 2
    assert summary["median_episode_ms"] == pytest.approx(50)
    assert summary["p95_episode_ms"] == pytest.approx(72.5)
    assert summary["total_episode_seconds"] == pytest.approx(0.1)


def test_streaming_and_full_text_references_make_the_decoder_difference_explicit():
    class Inspector:
        def observe_sequence(self, text):
            return {"tokens": [text], "boundary_probabilities": [0.9] * (len(text) - 1)}
    class Model:
        student = Inspector()
    settings = config()
    channel = ContextAcquisitionModel(**settings["channel"]).freeze()
    viterbi = runner.run_episode(Model(), "ab", channel, settings, mode="full_text_viterbi", char_cost=0.002)
    marginal = runner.run_episode(Model(), "ab", channel, settings, mode="full_text_marginal", char_cost=0.002)
    streamed = runner.run_episode(Model(), "ab", channel, settings, mode="fixed_stream", char_cost=0.002)
    assert viterbi["decoder"] == "viterbi"
    assert viterbi["tokens"] == ["ab"]
    assert marginal["decoder"] == streamed["decoder"] == "marginal_map"
    assert marginal["tokens"] == streamed["tokens"] == ["a", "b"]


def test_world_warmup_uses_actual_contexts_and_preserves_segmenter_predictions():
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19)
    transitions = []
    for index in range(32):
        prefix = f"word {index:02d}"
        before = model.student.observe_sequence(prefix)["context_vector"].tolist()
        after = model.student.observe_sequence(prefix + "next")["context_vector"].tolist()
        transitions.append({"before": before, "after": after, "action_chars": 4})
    state = model.state_dict()
    warmed, receipt = runner.warm_world(state, transitions, 19)
    assert receipt["transitions"] == 32
    assert receipt["world_training_calls"] == 1
    assert receipt["last_observation_id"] == 32
    assert warmed["phi_hist"] == {"AB": [], "BC": [], "CA": []}
    assert warmed["R3_mix"] == warmed["R2_time"] == 0
    restored = runner.load_model(warmed, 19)
    for text in ("abcdefgh", "日本語です", "中文测试"):
        assert restored.segment_text(text, include_confidence=True) == model.segment_text(text, include_confidence=True)


def test_development_halves_are_document_disjoint_and_channel_check_never_trains():
    rows = [ExternalSentence(language, f"d{index:02d}", f"s{index:02d}", f"p{index:02d}",
                             f"{language}-{index}", (language, "-", str(index)))
            for index in range(16) for language in ("en", "ja", "zh")]
    corpus = runner.group_parallel_corpus(rows, folds=4)
    partition, _ = corpus.partition(0)
    fitted, tuning, receipt = runner.split_development(corpus, partition, 19)
    assert len(fitted) == len(tuning) == 6
    for left in ("channel_fit", "cost_dev", "segmenter_train", "test"):
        for right in ("channel_fit", "cost_dev", "segmenter_train", "test"):
            if left != right:
                assert set(receipt[left]["groups"]).isdisjoint(receipt[right]["groups"])
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy")
    settings = config()
    settings["channel"].update(min_per_label=1)
    settings["active"].update(initial_chars=2)
    channel = ContextAcquisitionModel(**settings["channel"]).freeze()
    original = channel.export_state()
    result = runner.evaluate_channel(model, tuning, channel, settings)
    assert result["method"] == "heldout_one_step_prefix_extension_channel_check"
    assert channel.export_state() == original


def _offline_sources(tmp_path):
    sources = []
    for language in ("en", "ja", "zh"):
        revision = "a" * 40
        directory = tmp_path / "cache" / language / revision
        directory.mkdir(parents=True)
        blocks = []
        for index in range(12):
            forms = [language, "-", str(index)]
            lines = [f"# newdoc id = doc{index}", f"# sent_id = s{index}",
                     f"# parallel_id = own/s{index}", "# text = " + "".join(forms)]
            lines += ["\t".join([str(i + 1), form] + ["_"] * 8) for i, form in enumerate(forms)]
            blocks.append("\n".join(lines))
        files = {f"{language}_pud-ud-test.conllu": "\n\n".join(blocks),
                 "README.md": "Own synthetic test fixture", "LICENSE.txt": "Own synthetic test fixture"}
        metadata = {}
        for name, text in files.items():
            payload = text.encode()
            (directory / name).write_bytes(payload)
            metadata[name] = {"bytes": len(payload), "sha256": sha256(payload), "url": "unused-offline"}
        sources.append({"language": language, "revision": revision, "files": metadata})
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"sources": sources}))
    return manifest


def test_offline_study_freezes_choice_before_test_and_resumes_without_rerunning(tmp_path, monkeypatch):
    settings = config()
    settings.update(fold_count=3, char_cost_candidates=[0.01])
    settings["active"].update(max_window_chars=24, context_chars=8, trace_limit=16)
    settings["fixed_stream"].update(max_window_chars=24, context_chars=8, lookahead_chars=4)
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19)
    model.train_student(["en-1", "ja-1", "zh-1"], [["en", "-", "1"], ["ja", "-", "1"], ["zh", "-", "1"]],
                        StudentTrainingConfig(epochs=1, validation_split=0, phase_lr=0, dtype="float64", lexical_buckets=16))
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps(model.state_dict()))
    settings["checkpoint_sha256"] = sha256(checkpoint.read_bytes())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(settings))
    manifest = _offline_sources(tmp_path)
    output = tmp_path / "output"
    args = Namespace(config=config_path, checkpoint=checkpoint, sources=manifest, cache_dir=tmp_path / "cache",
                     output=output, publish=tmp_path / "published", download=False, prepare_only=True)
    runner.run(args)
    assert not (output / "development_selection.json").exists()
    plan = json.loads((output / "plan.json").read_text())
    test_ids = set(plan["contract"]["selection_partition"]["test"]["parallel_ids"])
    args.prepare_only = False
    original = runner.evaluate_rows
    def checked_evaluation(state, rows, *args, **kwargs):
        if any(row.parallel_id in test_ids for row in rows):
            # All test modes are requested only after the saved decision exists.
            assert (output / "development_selection.json").exists()
        return original(state, rows, *args, **kwargs)
    monkeypatch.setattr(runner, "evaluate_rows", checked_evaluation)
    runner.run(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["selection"]["char_cost"] == 0.01
    assert set(summary["evaluation_receipt_sha256"]) == set(runner.MODES)
    assert all(set(language) == set(runner.MODES) for language in summary["languages"].values())
    monkeypatch.setattr(runner, "evaluate_rows", lambda *a, **kw: pytest.fail("unexpected completed episode rerun"))
    args.publish = None
    runner.run(args)
    assert json.loads((output / "summary.json").read_text()) == summary
    # The resumed channel is immutable and its exact learned counts are bound.
    channel = json.loads((output / "channel.json").read_text())
    channel["observations"] += 1
    (output / "channel.json").write_text(json.dumps(channel))
    with pytest.raises(ValueError, match="channel integrity"):
        runner.run(args)
