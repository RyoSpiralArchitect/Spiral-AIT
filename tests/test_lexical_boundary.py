"""Optional identity-sensitive residual: mechanics, not corpus-quality claims."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.boundary import (
    BoundaryStudent,
    StudentTrainingConfig,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.phase import PhaseBasisLearner


def make_student(buckets=0, **options):
    student = BoundaryStudent(PhaseBasisLearner(dim=8), seed=19)
    student.julia_backend = student.compiled_backend = None
    cfg = StudentTrainingConfig(
        lexical_buckets=buckets, use_encoder_context=False, dtype="float64", **options,
    )
    student.configure(cfg)
    return student, cfg


def test_zero_initial_residual_preserves_existing_emissions():
    baseline, _ = make_student()
    lexical, _ = make_student(257)
    text = "東京から京都、hello 🌀"
    np.testing.assert_array_equal(baseline._python_logits(text), lexical._python_logits(text))
    assert not np.any(lexical.lexical_weights)
    assert baseline.lexical_weights.shape == (0,)


@pytest.mark.parametrize("buckets", [1, 17])
def test_sparse_lexical_crf_gradient_matches_finite_difference(buckets, monkeypatch):
    student, cfg = make_student(buckets, phase_lr=0, reg=0.03)
    monkeypatch.setattr(student.phase, "apply_error", lambda *args, **kwargs: None)
    student.lexical_weights[:] = np.linspace(-0.2, 0.3, buckets)
    seq = student.build_sequences(["東京都大阪"], [["東京", "都", "大阪"]])[0]
    _, gradients, _ = student._sequence_gradients(seq, cfg)
    assert isinstance(gradients["lexical_weights"], dict)

    def objective():
        logits, _ = student._forward_sequence(seq)
        nll = student._crf_loss(logits, student._labels_to_int(seq.labels))[0]
        return nll + 0.5 * cfg.reg * student._l2_norm()

    for bucket in range(buckets):
        original = student.lexical_weights[bucket]
        student.lexical_weights[bucket] = original + 1e-6
        upper = objective()
        student.lexical_weights[bucket] = original - 1e-6
        lower = objective()
        student.lexical_weights[bucket] = original
        analytic = gradients["lexical_weights"].get(bucket, 0.0) + cfg.reg * original
        assert analytic == pytest.approx((upper - lower) / 2e-6, abs=1e-7)
    if buckets == 1:
        assert any(abs(count) > 1 for features in seq.lexical_features for _, count in features)


def test_lexical_hashes_are_stable_across_process_hash_seeds():
    code = """
import json
from spiralreality_AIT_onepass_aifcore_integrated.integrated.boundary import BoundaryStudent, StudentTrainingConfig
from spiralreality_AIT_onepass_aifcore_integrated.integrated.phase import PhaseBasisLearner
s = BoundaryStudent(PhaseBasisLearner(dim=8))
s.julia_backend = s.compiled_backend = None
s.configure(StudentTrainingConfig(lexical_buckets=257, use_encoder_context=False))
print(json.dumps(s._lexical_features('日本語 🌀 e\\u0301')))
"""
    outputs = [subprocess.check_output(
        [sys.executable, "-c", code], text=True,
        env={**os.environ, "PYTHONHASHSEED": value},
    ) for value in ("1", "9876")]
    assert json.loads(outputs[0]) == json.loads(outputs[1])


def test_cached_lexical_features_survive_phase_updates_and_rebuild_for_new_bucket_count():
    student, _ = make_student(257)
    seq = student.build_sequences(["東京都"], [["東京", "都"]])[0]
    features = seq.lexical_features
    student.phase.apply_error(seq.text, 1, 3.0)
    student._forward_sequence(seq)
    assert seq.lexical_features is features
    student.configure(StudentTrainingConfig(lexical_buckets=1, use_encoder_context=False))
    student._forward_sequence(seq)
    assert seq.lexical_buckets == 1
    assert seq.lexical_features == student._lexical_features(seq.text)


def test_lexical_gradient_participates_in_clipping_and_regularisation():
    student, cfg = make_student(4, lexical_lr=0.05, max_grad_norm=5.0, reg=0.1)
    for parameter in student._regularized_parameters().values():
        parameter.fill(0)
    student.lexical_weights[:] = [3, 4, 1, 2]
    initial = student.lexical_weights.copy()
    gradients = student._zero_grad()
    gradients["lexical_weights"] = {0: 6.0, 1: 8.0}
    assert student._grad_norm(gradients) == 10.0
    student._apply_gradients(gradients, cfg, batch_size=2)
    objective_gradient = np.array([3.3, 4.4, 0.1, 0.2])
    scale = cfg.lexical_lr * (5 / (np.linalg.norm(objective_gradient) + 1e-9))
    np.testing.assert_allclose(
        student.lexical_weights, initial - scale * objective_gradient,
    )


def test_duplicate_batch_keeps_head_crf_and_lexical_updates_identical(monkeypatch):
    single, cfg = make_student(257, reg=0, max_grad_norm=None, phase_lr=0)
    doubled, _ = make_student(257, reg=0, max_grad_norm=None, phase_lr=0)
    monkeypatch.setattr(single.phase, "apply_error", lambda *args, **kwargs: None)
    seq = single.build_sequences(["東京都大阪"], [["東京", "都", "大阪"]])[0]
    _, gradients, _ = single._sequence_gradients(seq, cfg)
    accumulated = doubled._zero_grad()
    doubled._accumulate(accumulated, gradients)
    doubled._accumulate(accumulated, gradients)
    single._apply_gradients(gradients, cfg, batch_size=1)
    doubled._apply_gradients(accumulated, cfg, batch_size=2)
    for key in ("embeddings", "W_window", "W_out", "transitions", "lexical_weights"):
        np.testing.assert_array_equal(getattr(single, key), getattr(doubled, key))


def test_explicit_cjk_identity_supervision_is_learnable_with_categories_and_gate_frozen(monkeypatch):
    # All inputs have the same length and category sequence. With context and
    # gate features disabled the original head cannot distinguish these labels.
    texts = ["東京", "京都", "大阪", "奈良"]
    gold = [["東京"], ["京", "都"], ["大阪"], ["奈", "良"]]
    baseline, _ = make_student()
    lexical, _ = make_student()
    for student in (baseline, lexical):
        monkeypatch.setattr(student, "_gate_features", lambda seq, idx: np.zeros(3))
    common = dict(
        epochs=40, batch_size=4, validation_split=0, shuffle_train=False,
        lr=0, crf_lr=0, phase_lr=0, encoder_lr=0, reg=0,
        use_encoder_context=False, dtype="float64", max_grad_norm=None,
    )
    baseline.train(texts, gold, StudentTrainingConfig(**common))
    summary = lexical.train(texts, gold, StudentTrainingConfig(lexical_buckets=257, **common))
    baseline_probs = [float(baseline.boundary_probs(text)[0]) for text in texts]
    assert len(set(baseline_probs)) == 1
    assert [lexical.decode(text)["tokens"] for text in texts] == gold
    assert [baseline.decode(text)["tokens"] for text in texts] != gold
    assert summary["history"][-1]["train_loss"] < summary["history"][0]["train_loss"]
    assert summary["lexical_buckets"] == 257


def test_lexical_checkpoint_roundtrip_and_legacy_disable():
    student, _ = make_student(257)
    student.lexical_weights[:] = np.linspace(-0.8, 0.9, 257)
    state = json.loads(json.dumps(student.export_state()))
    restored, _ = make_student()
    restored.load_state(state)
    for text in ("東京都大阪", "English text", "", "字"):
        np.testing.assert_array_equal(student.boundary_probs(text), restored.boundary_probs(text))
        assert student.decode(text) == restored.decode(text)
    legacy = copy.deepcopy(state)
    legacy.pop("lexical")
    restored.load_state(legacy)
    assert restored.lexical_buckets == 0
    assert restored.lexical_weights.shape == (0,)
    student.configure(StudentTrainingConfig(lexical_buckets=0, use_encoder_context=False))
    assert student.lexical_weights.shape == (0,)


def test_sequence_cache_does_not_change_lexical_training():
    cached, _ = make_student()
    uncached, _ = make_student()
    texts = ["東京都", "大阪府", "京都市"]
    segments = [["東京", "都"], ["大阪", "府"], ["京都", "市"]]
    common = dict(lexical_buckets=257, epochs=3, validation_split=0.34, use_encoder_context=False)
    first = cached.train(texts, segments, StudentTrainingConfig(cache_sequences=True, **common))
    second = uncached.train(texts, segments, StudentTrainingConfig(cache_sequences=False, **common))
    assert first["history"] == second["history"]
    assert cached.export_state() == uncached.export_state()


def test_early_stopping_restores_lexical_weights(monkeypatch):
    student, _ = make_student(4)
    steps = 0

    def step(seq, cfg):
        nonlocal steps
        steps += 1
        student.lexical_weights[:] = steps
        return float(steps), student._zero_grad(), []

    monkeypatch.setattr(student, "_sequence_gradients", step)
    monkeypatch.setattr(student, "evaluate", lambda sequences: (float(student.lexical_weights[0]), 0.0))
    summary = student.train(
        ["東京", "京都"], [["東京"], ["京", "都"]],
        StudentTrainingConfig(
            lexical_buckets=4, use_encoder_context=False, epochs=4,
            validation_split=0.5, patience=1, reg=0,
        ),
    )
    assert summary["best_epoch"] == 1
    assert summary["epochs_completed"] == 2
    np.testing.assert_array_equal(student.lexical_weights, np.ones(4))


def test_lexical_path_never_delegates_to_native_backends():
    calls = []

    class UnsupportedNative:
        device = "cpu"

        def available_devices(self):
            return ["cpu"]

        def __getattr__(self, name):
            def unsupported(*args, **kwargs):
                calls.append(name)
                raise AssertionError("Native backend does not implement lexical features")
            return unsupported

    student, _ = make_student()
    student.julia_backend = student.compiled_backend = UnsupportedNative()
    summary = student.train(
        ["東京"], [["東", "京"]],
        StudentTrainingConfig(lexical_buckets=32, epochs=1, use_encoder_context=False),
    )
    assert summary["backend_used"] == "numpy"
    assert student.decode("東京")["backend_used"] == "python"
    assert student.boundary_probs("東京").shape == (1,)
    assert calls == []


def test_disabling_lexical_on_native_retrain_cannot_reuse_the_old_residual():
    class Native:
        device = "cpu"

        def preferred_device(self):
            return "cpu"

        def to_device(self, target):
            return True

        def available_devices(self):
            return ["cpu"]

        def train(self, *args):
            return {"history": []}

        def decode(self, text):
            return [text]

    student, _ = make_student(16)
    student.lexical_weights[:] = 100
    student.compiled_backend = Native()
    student.train(["東京"], [["東京"]], StudentTrainingConfig(use_encoder_context=False))
    assert student.lexical_buckets == 0
    assert student.lexical_weights.shape == (0,)
    assert student.decode("東京")["backend_used"] == "compiled:cpu"


@pytest.mark.parametrize("mutation", [
    {"version": "unknown"}, {"buckets": -1}, {"weights": [1.0]},
    {"weights": [float("nan")] * 4},
])
def test_malformed_lexical_checkpoints_are_rejected(mutation):
    student, _ = make_student(4)
    state = student.export_state()
    state["lexical"].update(mutation)
    with pytest.raises(ValueError, match="lexical"):
        student.load_state(state)


@pytest.mark.parametrize("options", [
    {"lexical_buckets": -1}, {"lexical_buckets": 1.5},
    {"lexical_lr": 0}, {"lexical_lr": -1}, {"lexical_lr": float("nan")},
])
def test_invalid_lexical_configuration_is_rejected_before_mutation(options):
    student, _ = make_student(16)
    before = student.export_state()
    with pytest.raises(ValueError, match="lexical"):
        student.train(["東京"], [["東京"]], StudentTrainingConfig(**options))
    assert student.export_state() == before


def test_empty_input_has_no_empty_tokens():
    student, _ = make_student(16)
    assert student.decode("")["tokens"] == []
    assert student.decode_with_logit_bias("")["tokens"] == []
