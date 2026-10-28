"""Continuation preserves a loaded Python head instead of initializing it again."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig


TEXTS = ["Український текст.", "Пам’ять і об'єкт."]
SEGMENTS = [["Український", " ", "текст", "."], ["Пам’ять", " ", "і", " ", "об'єкт", "."]]


def configured():
    model = OnePassAIT(latent_dim=8, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19)
    model.student.julia_backend = model.student.compiled_backend = None
    cfg = StudentTrainingConfig(epochs=1, validation_split=0, phase_lr=0, train_context_encoder=True,
                                hidden_dim=6, emb_dim=4, context_hidden_dim=4, lexical_buckets=32,
                                dtype="float64", shuffle_train=False, batch_size=1, encoder_lr=0.01)
    model.train_student(["old tokens"], [["old", " ", "tokens"]], cfg)
    return model, cfg


def arrays(model):
    head = model.student
    result = {name: np.asarray(getattr(head, name)).copy() for name in (
        "embeddings", "W_window", "b_window", "W_out", "b_out", "gate_w", "gate_b",
        "transitions", "lexical_weights", "ctx_W1", "ctx_b1", "ctx_w", "ctx_b")}
    result.update({"encoder/" + name: value.copy() for name, value in model.encoder.trainable_parameters().items()})
    return result


def assert_arrays_equal(model, expected):
    actual = arrays(model)
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def test_zero_epochs_preserve_loaded_weights_phase_predictions_and_parameter_handles():
    source, cfg = configured()
    saved = json.loads(json.dumps(source.state_dict()))
    model, _ = configured()
    model.load_state_dict(saved)
    probes = {text: model.segment_text(text, include_confidence=True)
              for text in (*TEXTS, "English text.", "日本語です。", "中文测试。")}
    before = copy.deepcopy(model.state_dict())
    handles = {name: getattr(model.student, name) for name in ("embeddings", "W_window", "lexical_weights", "ctx_W1")}
    rng_before = model.student.rng.getstate()
    summary = model.train_student(TEXTS, SEGMENTS, replace(cfg, epochs=0, warm_start=True))
    assert summary["warm_start"] is True and summary["epochs_completed"] == 0
    assert model.state_dict() == before
    assert model.student.rng.getstate() == rng_before
    assert all(getattr(model.student, name) is handle for name, handle in handles.items())
    for text, expected in probes.items():
        assert model.segment_text(text, include_confidence=True) == expected


def test_first_gradient_sees_loaded_head_encoder_and_phase_and_actual_learning_updates(monkeypatch):
    model, cfg = configured()
    expected = arrays(model)
    phase = copy.deepcopy(model.phase.export_state().basis)
    original = model.student._sequence_gradients
    calls = 0
    def gradients(sequence, config):
        nonlocal calls
        if calls == 0:
            assert_arrays_equal(model, expected)
            assert model.phase.export_state().basis == phase
        calls += 1
        return original(sequence, config)
    monkeypatch.setattr(model.student, "_sequence_gradients", gradients)
    summary = model.train_student(TEXTS, SEGMENTS, replace(cfg, warm_start=True))
    assert summary["warm_start"] is True and calls == len(TEXTS)
    assert not np.array_equal(model.student.lexical_weights, expected["lexical_weights"])
    assert any(not np.array_equal(value, expected["encoder/" + name])
               for name, value in model.encoder.trainable_parameters().items())
    assert model.phase.export_state().basis == phase


def test_checkpoint_continuation_matches_uninterrupted_sgd_when_order_is_fixed():
    initial, cfg = configured()
    uninterrupted = copy.deepcopy(initial)
    uninterrupted.train_student(TEXTS, SEGMENTS, replace(cfg, epochs=2, warm_start=True))
    initial.train_student(TEXTS, SEGMENTS, replace(cfg, epochs=1, warm_start=True))
    checkpoint = json.loads(json.dumps(initial.state_dict()))
    restored, _ = configured()
    restored.load_state_dict(checkpoint)
    restored.train_student(TEXTS, SEGMENTS, replace(cfg, epochs=1, warm_start=True))
    assert restored.state_dict() == uninterrupted.state_dict()
    for text in TEXTS:
        assert restored.segment_text(text, include_confidence=True) == uninterrupted.segment_text(text, include_confidence=True)


@pytest.mark.parametrize("changes", [
    {"hidden_dim": 7}, {"emb_dim": 5}, {"window": 3}, {"context_hidden_dim": 5},
    {"dtype": "float32"}, {"dtype": "unexpected"}, {"lexical_buckets": 0},
    {"use_encoder_context": False, "train_context_encoder": False}, {"use_encoder_context": 1}, {"hidden_dim": True},
])
def test_incompatible_continuation_rejects_before_any_live_state_changes(changes):
    model, cfg = configured()
    before = copy.deepcopy(model.state_dict())
    before_history, before_rng = copy.deepcopy(model.student.history), model.student.rng.getstate()
    model.student.best_state = {"unchanged": True}
    with pytest.raises(ValueError, match="warm_start"):
        model.train_student(TEXTS, SEGMENTS, replace(cfg, warm_start=True, **changes))
    assert model.state_dict() == before
    assert model.student.history == before_history and model.student.best_state == {"unchanged": True}
    assert model.student.rng.getstate() == before_rng


@pytest.mark.parametrize("value", [1, "yes", None, np.bool_(True)])
def test_warm_start_requires_explicit_boolean(value):
    model, cfg = configured()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="warm_start must be a boolean"):
        model.train_student(TEXTS, SEGMENTS, replace(cfg, warm_start=value))
    assert model.state_dict() == before


@pytest.mark.parametrize("backend", [None, "julia", "compiled"])
def test_warm_start_requires_existing_python_ownership(backend):
    model, cfg = configured()
    model.student._fitted_backend = backend
    before = copy.deepcopy(model.student._capture_state())
    with pytest.raises(ValueError, match="initialized Python"):
        model.train_student(TEXTS, SEGMENTS, replace(cfg, warm_start=True))
    assert model.student._capture_state() == before
    assert model.student._fitted_backend == backend


@pytest.mark.parametrize("parameter", ["ctx_W1", "lexical_weights", "W_out"])
def test_damaged_parameter_arrays_reject_before_reconfiguration(parameter):
    model, cfg = configured()
    damaged = getattr(model.student, parameter).copy()
    damaged.flat[0] = np.nan
    setattr(model.student, parameter, damaged)
    before = arrays(model)
    with pytest.raises(ValueError, match="valid existing"):
        model.train_student(TEXTS, SEGMENTS, replace(cfg, warm_start=True))
    assert_arrays_equal(model, before)


def test_cold_training_keeps_its_previous_reinitialization_behavior():
    model, cfg = configured()
    model.student.b_out = 99
    model.student.lexical_weights[:] = 9
    expected = copy.deepcopy(model)
    expected.student.configure(replace(cfg, epochs=0))
    summary = model.train_student(TEXTS, SEGMENTS, replace(cfg, epochs=0))
    assert "warm_start" not in summary
    assert model.state_dict() == expected.state_dict()
    assert model.student.b_out == 0 and np.all(model.student.lexical_weights == 0)


def test_warm_start_can_freeze_encoder_and_change_optimization_controls():
    model, cfg = configured()
    before = copy.deepcopy(model.encoder.export_state())
    head = model.student.lexical_weights.copy()
    summary = model.train_student(TEXTS, SEGMENTS, replace(cfg, warm_start=True, train_context_encoder=False, encoder_lr=0,
                                                          lr=0.01, context_lr=0.01, max_grad_norm=1.0))
    assert summary["warm_start"] is True and not summary["train_context_encoder"]
    assert model.encoder.export_state() == before
    assert not np.array_equal(model.student.lexical_weights, head)
    assert model.student.max_grad_norm == 1.0


def test_native_config_schema_is_unchanged_and_warm_start_bypasses_available_native_heads():
    model, cfg = configured()
    # A no-context Python head can coexist with installed native alternatives.
    cfg = replace(cfg, use_encoder_context=False, train_context_encoder=False, lexical_buckets=0)
    model.student.configure(cfg)
    class StrictNative:
        device = "cpu"
        def preferred_device(self): return "cpu"
        def to_device(self, device): return True
        def available_devices(self): return ["cpu"]
        def configure(self, values): assert "warm_start" not in values
        def train(self, texts, segments, values):
            assert "warm_start" not in values
            return {"history": []}
    native = StrictNative()
    model.student.julia_backend = native
    model.student.configure(cfg)
    model.student.b_out = 12
    # Any native invocation during warm continuation must be a test failure.
    native.train = lambda *args: pytest.fail("native warm-start route was used")
    native.configure = lambda *args: pytest.fail("native warm-start configure was used")
    summary = model.train_student(TEXTS, SEGMENTS, replace(cfg, epochs=0, warm_start=True))
    assert summary["warm_start"] is True and model.student.b_out == 12
    model.student.configure = lambda *args: pytest.fail("native cold path should return before Python configure")
    native.train = StrictNative().train
    summary = model.train_student(TEXTS, SEGMENTS, replace(cfg, warm_start=False))
    assert summary["backend"].startswith("julia")
