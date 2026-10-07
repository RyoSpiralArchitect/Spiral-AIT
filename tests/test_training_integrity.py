"""Regression evidence for training/inference state and supervision integrity."""

import copy
import json

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.boundary import (
    BoundaryStudent,
    StudentTrainingConfig,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.encoder import SpectralTransformerAdapter
from spiralreality_AIT_onepass_aifcore_integrated.integrated.phase import PhaseBasisLearner


def make_student(*, context=False):
    student = BoundaryStudent(PhaseBasisLearner(dim=8), seed=41)
    student.julia_backend = student.compiled_backend = None
    if context:
        student.bind_encoder(SpectralTransformerAdapter(d_model=8, n_layers=1, n_heads=2))
    return student


TEXTS = ["ab cd", "ef gh", "ij kl"]
SEGMENTS = [["ab ", "cd"], ["ef ", "gh"], ["ij ", "kl"]]


def test_cached_features_follow_phase_updates():
    student = make_student(context=True)
    cached = student.build_sequences(TEXTS[:1], SEGMENTS[:1])[0]
    old_phases = cached.phases.copy()
    student.phase.apply_error(cached.text, 2, error=4.0)
    cached_logits, _ = student._forward_sequence(cached)
    fresh = student.build_sequences(TEXTS[:1], SEGMENTS[:1])[0]
    fresh_logits, _ = student._forward_sequence(fresh)
    assert not np.allclose(old_phases, cached.phases)
    np.testing.assert_allclose(cached.phases, fresh.phases)
    np.testing.assert_allclose(cached_logits, fresh_logits)


def test_cache_does_not_change_training_or_validation_results():
    cached, uncached = make_student(context=True), make_student(context=True)
    common = dict(epochs=3, batch_size=1, validation_split=0.34, patience=3)
    a = cached.train(TEXTS, SEGMENTS, StudentTrainingConfig(cache_sequences=True, **common))
    b = uncached.train(TEXTS, SEGMENTS, StudentTrainingConfig(cache_sequences=False, **common))
    assert a["history"] == b["history"]
    json.dumps(a)
    assert a["val_loss"] == b["val_loss"]
    assert cached.export_state() == uncached.export_state()
    assert a["processed_train_tokens"] == a["train_tokens"] * a["epochs_completed"]
    assert set(a["train_indices"]).isdisjoint(a["validation_indices"])
    assert sorted(a["train_indices"] + a["validation_indices"]) == list(range(len(TEXTS)))


def test_early_stopping_restores_phase_encoder_and_returned_metrics(monkeypatch):
    student = make_student(context=True)
    captures = []
    capture = student._capture_state
    steps = 0

    def controlled_step(seq, cfg):
        nonlocal steps
        steps += 1
        student.b_out = float(steps)
        student.phase.apply_error(seq.text, 1, error=float(steps))
        student.encoder_adapter.gate_bias[0] += float(steps)
        return float(steps), student._zero_grad(), []

    def capture_step():
        state = capture()
        captures.append(copy.deepcopy(state))
        return state

    monkeypatch.setattr(student, "_sequence_gradients", controlled_step)
    monkeypatch.setattr(student, "_capture_state", capture_step)
    monkeypatch.setattr(student, "evaluate", lambda sequences: (student.b_out ** 2, 1 / student.b_out))
    summary = student.train(
        TEXTS[:2], SEGMENTS[:2],
        StudentTrainingConfig(epochs=4, batch_size=1, validation_split=0.5, patience=1),
    )
    assert summary["best_epoch"] == 1
    assert summary["epochs_completed"] == 2
    assert summary["history"][-1]["val_loss"] == 4.0
    assert summary["val_loss"] == 1.0
    assert summary["val_f1"] == 1.0
    assert capture() == captures[0]


def test_retraining_without_validation_does_not_restore_previous_fit():
    student = make_student(context=True)
    student.train(TEXTS, SEGMENTS, StudentTrainingConfig(epochs=2, validation_split=0.34))
    previous = student.export_state()
    assert student.best_state is not None
    summary = student.train(
        ["x yz"], [["x ", "yz"]],
        StudentTrainingConfig(epochs=1, validation_split=0, hidden_dim=7, emb_dim=5, window=3),
    )
    assert student.best_state is None
    assert len(student.history) == 1
    assert summary["best_epoch"] is None
    assert student.W_window.shape == (7, 30)
    assert student.export_state() != previous
    assert "".join(student.decode("x yz")["tokens"]) == "x yz"


def test_configure_resets_context_parameters():
    student = make_student(context=True)
    cfg = StudentTrainingConfig(context_hidden_dim=6)
    student.configure(cfg)
    initial = student.ctx_W1.copy()
    student.ctx_W1 += 10.0
    student.ctx_b = 5.0
    student.configure(cfg)
    np.testing.assert_array_equal(student.ctx_W1, initial)
    assert student.ctx_b == 0.0


@pytest.mark.parametrize("legacy", [False, True])
def test_custom_architecture_checkpoint_roundtrip(legacy):
    student = make_student(context=True)
    student.train(
        TEXTS, SEGMENTS,
        StudentTrainingConfig(epochs=2, validation_split=0, hidden_dim=7, emb_dim=5, window=3, dtype="float64"),
    )
    state = student.export_state()
    restored = make_student(context=True)
    if legacy:
        state.pop("architecture")
        # Old outer checkpoints supplied the dtype separately.
        restored.dtype = np.float64
    restored.load_state(state)
    assert (restored.hidden_dim, restored.emb_dim, restored.window) == (7, 5, 3)
    np.testing.assert_allclose(restored.boundary_probs("new text"), student.boundary_probs("new text"))
    assert restored.decode("new text")["tokens"] == student.decode("new text")["tokens"]


@pytest.mark.parametrize("texts,segments", [
    (["ab", "cd"], [["ab"]]),
    (["ab cd"], [["ab", "cd"]]),
    (["abc"], [["ab", "x"]]),
    (["abc"], [["", "abc"]]),
    (["abc"], ["abc"]),
    (["abc"], [[1, "bc"]]),
])
def test_invalid_supervision_fails_before_state_or_backend_mutation(texts, segments):
    student = make_student(context=True)
    before = student.export_state()

    class BackendMustNotRun:
        def train(self, *args, **kwargs):
            raise AssertionError("invalid supervision reached a backend")

    student.julia_backend = BackendMustNotRun()
    with pytest.raises(ValueError):
        student.train(texts, segments, StudentTrainingConfig(use_encoder_context=False))
    student.julia_backend = None
    assert student.export_state() == before


def test_zero_epochs_validation_is_reported_without_index_error():
    student = make_student()
    summary = student.train(TEXTS, SEGMENTS, StudentTrainingConfig(epochs=0, validation_split=0.34))
    assert summary["epochs_completed"] == 0
    assert np.isfinite(summary["val_loss"])


def test_precomputed_logits_reuse_encoder_and_preserve_bias_results(monkeypatch):
    student = make_student(context=True)
    text = "abc def"
    logits = student._python_logits(text)
    expected_probs = student.boundary_probs_with_logit_bias(text, 0.3)
    expected_tokens = student.decode_with_logit_bias(text, 0.3)["tokens"]

    def unexpected_forward(text):
        raise AssertionError("recomputed supplied logits")

    monkeypatch.setattr(student, "_python_logits", unexpected_forward)
    np.testing.assert_allclose(student.boundary_probs_with_logit_bias(text, 0.3, logits=logits), expected_probs)
    assert student.decode_with_logit_bias(text, 0.3, logits=logits)["tokens"] == expected_tokens
    assert logits == student._resolve_logits(text, logits)
    for method in (student.boundary_probs_with_logit_bias, student.decode_with_logit_bias):
        with pytest.raises(ValueError, match="one score per character boundary"):
            method(text, logits=logits[:-1])
