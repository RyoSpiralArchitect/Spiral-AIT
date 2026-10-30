from argparse import Namespace
from collections import Counter
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import benchmark_slavic_learning as runner
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def config():
    return json.loads((runner.ROOT / "configs/slavic-learning-pilot.json").read_text())


def row(language, text, role="test", number=1, segments=None):
    source = SimpleNamespace(language=language, document_id="document-" + str(number), sentence_id="sentence-" + str(number),
                             text=text, segments=tuple(segments or [text]), upstream_split="test", source_id="authored_fixture",
                             parallel_id=None if language == "uk" else "parallel-" + str(number))
    return runner.row_record(source, role, "group-" + str(number))


def examples():
    return [row(language, language + " abc", segments=[language, " ", "abc"]) for language in (*runner.SLAVIC, *runner.LANGUAGES)]


def payload():
    all_rows = examples()
    return {"slavic_train": [{**item, "role": "train"} for item in all_rows[:3]],
            "rehearsal": [{**item, "role": "train"} for item in all_rows[3:]],
            "evaluation": [{**item, "evaluation_role": "slavic_test" if item["language"] in runner.SLAVIC else "retention_test"}
                           for item in all_rows]}


def tiny_state():
    return OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19).state_dict()


def test_cross_role_guard_uses_legacy_exposure_and_keeps_original_texts():
    original = [row("uk", "Привіт.", "train", 1), row("uk", " ПРИВІТ. ", "test", 2),
                row("pl", "Old text", "train", 3), row("cs", "Jedinečný text.", "test", 4),
                row("uk", "обʼєкт і пам’ять", "train", 5)]
    kept, excluded = runner.guard_slavic_roles(original, [row("en", "old TEXT", "test", 6)])
    assert [item["text"] for item in kept] == ["Jedinečný text.", "обʼєкт і пам’ять"] or {
        item["text"] for item in kept} == {"Jedinečný text.", "обʼєкт і пам’ять"}
    assert len(excluded) == 3
    assert {item["reason"] for item in excluded} == {"normalized_duplicate_cross_role_including_legacy"}
    assert all("text" not in item and "segments" not in item for item in excluded)
    assert original[0]["text"] == "Привіт."


def test_duplicate_and_ambiguous_labels_are_removed_before_caps():
    candidates = [row("pl", "Dziękuję.", "train", 1), row("pl", "DZIĘKUJĘ.", "train", 2),
                  row("uk", "ab", "train", 3, ["a", "b"]), row("uk", "ab", "train", 4, ["ab"])]
    kept, excluded = runner.guard_slavic_roles(candidates, [])
    assert len(kept) == 1 and kept[0]["language"] == "pl"
    assert Counter(item["reason"] for item in excluded) == {
        "ambiguous_exact_surface_supervision": 2, "normalized_duplicate_within_role_language": 1}


def test_selection_is_deterministic_and_does_not_consult_quality_outcomes():
    candidates = [row("uk", "case " + str(index), "train", index) for index in range(12)]
    expected = [item["sample_id"] for item in runner.select_rows(candidates, 4, 17, "train")]
    for index, item in enumerate(candidates):
        item["test_f1"] = 1 if index % 2 else 0
    assert [item["sample_id"] for item in runner.select_rows(list(reversed(candidates)), 4, 17, "train")] == expected


def test_architecture_is_derived_and_only_training_controls_are_declared():
    settings, state = config(), tiny_state()
    cfg = runner.training_settings(state, settings)
    for name in ("hidden_dim", "emb_dim", "window", "dtype"):
        assert cfg[name] == state["student"]["architecture"][name]
    assert cfg["warm_start"] is True and cfg["context_hidden_dim"] == state["student"]["context_hidden_dim"]
    settings["training"]["hidden_dim"] = 999
    with pytest.raises(ValueError, match="Architecture"):
        runner.validate_config(settings)


def test_probability_and_task_results_reject_missing_bins_or_wrong_identity(monkeypatch):
    model = runner.load_model(tiny_state(), 19)
    expected = payload()["evaluation"]
    result = runner.tree(runner.evaluate_rows(model, expected))
    runner.validate_results(result, expected)
    for mutate in (lambda r: r.pop(), lambda r: r[0].update(sample_id="wrong"),
                   lambda r: r[0]["boundary"].update(f1=0.2),
                   lambda r: r[0]["calibration"]["bin_label_sums"].pop(),
                   lambda r: r[0]["calibration"]["bin_probability_sums"].__setitem__(0, float("nan"))):
        damaged = copy.deepcopy(result)
        mutate(damaged)
        with pytest.raises(ValueError):
            runner.validate_results(damaged, expected)


def test_json_roundtrip_and_download_transport_do_not_change_identity(tmp_path):
    value = {"histogram": {2: 1, 10: 3}, "signed": -0.2}
    path = tmp_path / "receipt.json"
    runner.immutable_json(path, value)
    assert runner.digest(json.loads(path.read_text())) == runner.digest(value)
    runner.immutable_json(path, value)
    with pytest.raises(ValueError, match="Immutable"):
        runner.immutable_json(path, {"changed": True})
    before = {"sources": [{"files": [{"sha256": "same", "downloaded": True}]}]}
    after = {"sources": [{"files": [{"sha256": "same", "downloaded": False}]}]}
    assert runner.canonical_acquisition(before) == runner.canonical_acquisition(after)


def fake_study(tmp_path, monkeypatch):
    settings = config()
    settings["seeds"] = [19]
    settings["training"]["epochs"] = 1
    state, data = tiny_state(), payload()
    checkpoint = tmp_path / "initial.json"
    runner.immutable_json(checkpoint, state)
    settings["checkpoint_sha256"] = runner.file_hash(checkpoint)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(settings))
    monkeypatch.setattr(runner, "prepare_inputs", lambda *args: (data, {"fixture": "only authored cases"}))
    calls = []
    def execute(path):
        task = json.loads(Path(path).read_text())
        calls.append(task["run_id"])
        model = runner.load_model(state, task["seed"])
        initial = runner.parameter_state(model)
        model.student.W_out[0] += 0.01
        model.encoder.Wq[0][0, 0] += 0.001
        final = runner.parameter_state(model)
        runner.immutable_json(task["trained_checkpoint"], model.state_dict())
        result = runner.tree(runner.evaluate_rows(model, data["evaluation"]))
        receipt = {key: task[key] for key in ("contract_sha256", "run_id", "arm", "seed", "input_sha256", "training")}
        receipt.update(checkpoint_sha256=runner.file_hash(task["trained_checkpoint"]), results=result,
                       results_sha256=runner.digest(result), initial_parameter_sha256=task["initial_parameter_sha256"],
                       zero_epoch_parameter_sha256=task["initial_parameter_sha256"], training_wall_seconds=0.01,
                       final_parameter_sha256=runner.digest(final),
                       initial_component_sha256={k: runner.digest(v) for k, v in initial.items()},
                       final_component_sha256={k: runner.digest(v) for k, v in final.items()},
                       checkpoint_bytes=Path(task["trained_checkpoint"]).stat().st_size,
                       training_examples=len(data["slavic_train"]) + (len(data["rehearsal"]) if task["arm"] == "slavic_rehearsal" else 0),
                       processed_examples=len(data["slavic_train"]) + (len(data["rehearsal"]) if task["arm"] == "slavic_rehearsal" else 0),
                       training_examples_per_language=dict(Counter(r["language"] for r in data["slavic_train"] + (data["rehearsal"] if task["arm"] == "slavic_rehearsal" else []))),
                       history=[{"epoch": 1.0, "train_loss": 1.0}], whole_worker_peak_rss_bytes=100,
                       runtime={**task["runtime"], "thread_environment": runner.THREADS})
        runner.immutable_json(task["receipt"], receipt)
    monkeypatch.setattr(runner, "execute_worker", execute)
    args = Namespace(config=config_path, checkpoint=checkpoint, output=tmp_path / "study", prepare_only=True,
                     publish=None, download=False, legacy_sources="fixture", legacy_cache="fixture", slavic_cache="fixture")
    return args, calls


def test_prepare_then_complete_publish_and_resume_without_training(tmp_path, monkeypatch):
    args, calls = fake_study(tmp_path, monkeypatch)
    runner.run(args)
    assert not calls and not (args.output / "input.json").exists()
    args.prepare_only, args.publish = False, tmp_path / "public"
    runner.run(args)
    assert len(calls) == 2
    assert len((args.publish / "paired_scores.jsonl").read_text().splitlines()) == 7
    assert json.loads((args.publish / "summary.json").read_text())["total_evaluations"] == 18
    args.publish = None
    monkeypatch.setattr(runner, "execute_worker", lambda *args: pytest.fail("resume retrained"))
    runner.run(args)
    path = next((args.output / "training").glob("*.json"))
    value = json.loads(path.read_text())
    value["results"].pop()
    value["results_sha256"] = runner.digest(value["results"])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Incomplete"):
        runner.run(args)


def test_incomplete_training_attempt_is_preserved_without_an_implicit_retry(tmp_path, monkeypatch):
    args, calls = fake_study(tmp_path, monkeypatch)
    args.prepare_only = False
    def interrupted(*args):
        raise RuntimeError("training interrupted")
    monkeypatch.setattr(runner, "execute_worker", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run(args)
    monkeypatch.setattr(runner, "execute_worker", lambda *args: pytest.fail("implicit retry"))
    with pytest.raises(ValueError, match="Incomplete training attempt"):
        runner.run(args)
    assert len(list((args.output / "attempts").glob("*.json"))) == 1


def test_fresh_training_worker_preserves_initial_parameters_and_saves_reloadable_checkpoint(tmp_path):
    # Isolated code snapshots prevent other agents' concurrent source edits
    # from changing this subprocess's integrity contract during the test.
    implementation = tmp_path / "implementation"
    shutil.copytree(runner.ROOT / runner.PACKAGE, implementation / runner.PACKAGE,
                    ignore=shutil.ignore_patterns("__pycache__"))
    script = implementation / "scripts/benchmark_slavic_learning.py"
    script.parent.mkdir()
    shutil.copyfile(runner.__file__, script)
    code = {str(path.relative_to(implementation)): runner.file_hash(path)
            for path in sorted((implementation / runner.PACKAGE).rglob("*.py")) + [script]}
    initial, data, settings = tiny_state(), payload(), config()
    settings["training"]["epochs"] = 1
    start, input_path = tmp_path / "initial.json", tmp_path / "input.json"
    runner.immutable_json(start, initial)
    runner.immutable_json(input_path, data)
    task = {"contract_sha256": "authored-worker-fixture", "arm": "slavic_only", "seed": 19, "run_id": "fixture",
            "code_sha256": code, "checkpoint": str(start), "checkpoint_sha256": runner.file_hash(start),
            "input": str(input_path), "input_sha256": runner.digest(data),
            "initial_parameter_sha256": runner.digest(runner.parameter_state(runner.load_model(initial, 19))),
            "training": runner.training_settings(initial, settings),
            "runtime": {"python": runner.platform.python_version(), "numpy": np.__version__, "platform": runner.platform.platform()},
            "trained_checkpoint": str(tmp_path / "trained.json"), "receipt": str(tmp_path / "receipt.json")}
    task_path = tmp_path / "task.json"
    runner.immutable_json(task_path, task)
    environment = {**os.environ, **runner.THREADS, "PYTHONHASHSEED": "0"}
    environment.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, str(script), "--worker", str(task_path)], cwd=implementation, env=environment, check=True)
    receipt = json.loads(Path(task["receipt"]).read_text())
    assert receipt["initial_parameter_sha256"] == receipt["zero_epoch_parameter_sha256"] == task["initial_parameter_sha256"]
    assert receipt["final_parameter_sha256"] != receipt["initial_parameter_sha256"]
    assert receipt["training_examples"] == 3 and receipt["processed_examples"] == 3
    restored = runner.load_model(json.loads(Path(task["trained_checkpoint"]).read_text()), 19)
    result = runner.evaluate_rows(restored, data["evaluation"])
    assert [item["prediction_sha256"] for item in result] == [item["prediction_sha256"] for item in receipt["results"]]
