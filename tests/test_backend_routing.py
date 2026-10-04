"""Checkpoint and inference routes must retain the actually fitted head."""

import copy
import json

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.boundary import BoundaryStudent, StudentTrainingConfig
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT
from spiralreality_AIT_onepass_aifcore_integrated.integrated.phase import PhaseBasisLearner


class Native:
    device = "cpu"
    backend = "test"

    def __init__(self):
        self.width = 2
        self.calls = []

    def preferred_device(self):
        return "cpu"

    def to_device(self, device):
        return True

    def available_devices(self):
        return ["cpu"]

    def train(self, *args):
        self.width = 1
        return {"history": []}

    def decode(self, text):
        self.calls.append("decode")
        return [text[index:index + self.width] for index in range(0, len(text), self.width)]

    def boundary_probs(self, text):
        self.calls.append("probs")
        return np.ones(max(0, len(text) - 1))

    def export_state(self):
        return {"width": self.width}

    def load_state(self, state):
        self.width = state["width"]


def student():
    model = BoundaryStudent(PhaseBasisLearner(dim=8), seed=19)
    model.julia_backend = model.compiled_backend = None
    return model


@pytest.mark.parametrize("legacy", [False, True])
def test_python_checkpoint_disables_unrelated_native_handles(legacy):
    source = student()
    source.configure(StudentTrainingConfig(use_encoder_context=False))
    source.b_out = -50
    state = source.export_state()
    if legacy:
        state.pop("fitted_backend")
    target = student()
    native = Native()
    target.julia_backend = target.compiled_backend = native
    target.load_state(state)
    assert source.decode("abcd")["tokens"] == ["abcd"]
    assert target.decode("abcd")["tokens"] == ["abcd"]
    np.testing.assert_array_equal(target.boundary_probs("abcd"), source.boundary_probs("abcd"))
    assert native.calls == []
    assert target.julia_backend is target.compiled_backend is None


@pytest.mark.parametrize("backend", ["compiled", "julia"])
def test_native_fit_and_checkpoint_retain_the_native_route(backend):
    source = student()
    setattr(source, f"{backend}_backend", Native())
    source.train(["abcd"], [["a", "b", "c", "d"]], StudentTrainingConfig(use_encoder_context=False))
    assert source.use_encoder_context is False
    assert source._fitted_backend == backend
    assert source.decode("abcd")["tokens"] == list("abcd")
    state = json.loads(json.dumps(source.export_state()))
    assert state["fitted_backend"] == backend
    assert f"_{backend}" in state
    for legacy in (False, True):
        payload = copy.deepcopy(state)
        if legacy:
            payload.pop("fitted_backend")
        restored = student()
        setattr(restored, f"{backend}_backend", Native())
        # An unrelated available backend must not steal the route after load.
        setattr(restored, "julia_backend" if backend == "compiled" else "compiled_backend", Native())
        restored.load_state(payload)
        assert restored.decode("abcd")["tokens"] == list("abcd")
        assert restored._fitted_backend == backend
        np.testing.assert_array_equal(restored.boundary_probs("abcd"), np.ones(3))


@pytest.mark.parametrize("backend", ["compiled", "julia"])
def test_native_fit_rejects_unsynchronised_python_logits(backend):
    model = OnePassAIT(latent_dim=8, encoder_layers=1, encoder_heads=2)
    model.student.julia_backend = model.student.compiled_backend = None
    setattr(model.student, f"{backend}_backend", Native())
    model.train_student(["abcd"], [list("abcd")], StudentTrainingConfig(use_encoder_context=False))
    assert model.segment_text("abcd") == list("abcd")
    for options in ({"include_confidence": True}, {"use_aif": True}):
        with pytest.raises(ValueError, match="synchronized Python weights"):
            model.segment_text("abcd", **options)
    for method in (model.student.decode_with_logit_bias, model.student.boundary_probs_with_logit_bias):
        for provided in (None, [0.0, 0.0, 0.0]):
            with pytest.raises(ValueError, match="synchronized Python weights"):
                method("abcd", logits=provided)


def test_native_checkpoint_requires_its_saved_weights_and_matching_backend():
    source = student()
    source.compiled_backend = Native()
    source.train(["abcd"], [list("abcd")], StudentTrainingConfig(use_encoder_context=False))
    state = source.export_state()
    with pytest.raises(ValueError, match="requires its fitted compiled backend"):
        student().load_state(state)
    del state["_compiled"]
    target = student()
    target.compiled_backend = Native()
    with pytest.raises(ValueError, match="serialized weights"):
        target.load_state(state)


def test_legacy_checkpoint_with_two_native_heads_is_explicitly_ambiguous():
    model = student()
    model.configure(StudentTrainingConfig(use_encoder_context=False))
    state = model.export_state()
    del state["fitted_backend"]
    state["_compiled"] = state["_julia"] = {"state": {"width": 1}}
    with pytest.raises(ValueError, match="ambiguous"):
        model.load_state(state)


@pytest.mark.parametrize("operation", ["decode", "boundary_probs", "export_state"])
def test_fitted_native_failure_never_falls_back_to_untrained_python(operation, monkeypatch):
    model = student()
    model.compiled_backend = Native()
    model.train(["abcd"], [list("abcd")], StudentTrainingConfig(use_encoder_context=False))

    def fail(*args):
        raise OSError("native unavailable")

    monkeypatch.setattr(model.compiled_backend, operation, fail)
    args = () if operation == "export_state" else ("abcd",)
    with pytest.raises(RuntimeError, match="[Ff]itted"):
        getattr(model, operation)(*args)
