import copy
import json

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.encoder import SpectralTransformerAdapter
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig


def test_transformer_backward_matches_finite_differences_for_every_parameter_family():
    encoder = SpectralTransformerAdapter(d_model=4, n_layers=2, n_heads=2, ff_multiplier=1, seed=10)
    rng = np.random.default_rng(11)
    inputs = rng.normal(size=(3, 4))
    gates = np.array([0.2, 0.7, 0.4])
    direction = rng.normal(size=inputs.shape)
    output, tape = encoder.forward_with_cache(inputs, gates)
    np.testing.assert_array_equal(output, encoder.forward(inputs, gates))
    dx, gradients = encoder.backward(direction, tape)
    epsilon = 1e-5

    def objective():
        return float(np.sum(encoder.forward(inputs, gates) * direction))

    for name, parameter in encoder.trainable_parameters().items():
        for index in (tuple(0 for _ in parameter.shape), tuple(size - 1 for size in parameter.shape)):
            old = parameter[index]
            parameter[index] = old + epsilon
            plus = objective()
            parameter[index] = old - epsilon
            minus = objective()
            parameter[index] = old
            assert gradients[name][index] == pytest.approx((plus - minus) / (2 * epsilon), abs=2e-6, rel=2e-5), name
    for index in np.ndindex(inputs.shape):
        old = inputs[index]
        inputs[index] = old + epsilon
        plus = objective()
        inputs[index] = old - epsilon
        minus = objective()
        inputs[index] = old
        assert dx[index] == pytest.approx((plus - minus) / (2 * epsilon), abs=2e-6, rel=2e-5)


def make_model(seed=17):
    return OnePassAIT(latent_dim=8, seed=seed, encoder_layers=1, encoder_heads=2, encoder_backend="numpy")


def test_crf_loss_gradient_reaches_context_transformer():
    model = make_model()
    cfg = StudentTrainingConfig(train_context_encoder=True, phase_lr=0, reg=0, dtype="float64")
    model.student.configure(cfg)
    seq = model.student.build_sequences(["日本語 test"], [["日本", "語", " ", "test"]])[0]
    _, grads, _ = model.student._sequence_gradients(seq, cfg)
    before = copy.deepcopy(model.state_dict())
    epsilon = 1e-4
    for name, parameter in model.encoder.trainable_parameters().items():
        index = np.unravel_index(np.abs(grads["encoder"][name]).argmax(), parameter.shape)
        old = parameter[index]
        losses = []
        for delta in (epsilon, -epsilon):
            parameter[index] = old + delta
            logits, _ = model.student._forward_sequence(seq)
            losses.append(model.student._crf_loss(logits, model.student._labels_to_int(seq.labels))[0])
        parameter[index] = old
        assert grads["encoder"][name][index] == pytest.approx((losses[0] - losses[1]) / (2 * epsilon), abs=2e-7, rel=2e-3), name
    assert model.state_dict() == before


def test_supervised_encoder_updates_roundtrip_and_remains_opt_in():
    texts = ["alpha beta", "日本語です", "中文测试"]
    segments = [["alpha", " ", "beta"], ["日本", "語", "です"], ["中文", "测试"]]
    for trainable in (False, True):
        model = make_model()
        before = copy.deepcopy(model.encoder.export_state())
        phase = copy.deepcopy(model.phase.export_state().basis)
        cfg = StudentTrainingConfig(epochs=3, validation_split=0, phase_lr=0,
                                    train_context_encoder=trainable, encoder_lr=0.05 if trainable else 0)
        model.train_student(texts, segments, cfg)
        assert (model.encoder.export_state() != before) == trainable
        assert model.phase.export_state().basis == phase
        state = json.loads(json.dumps(model.state_dict()))
        restored = make_model(seed=99)
        restored.load_state_dict(state)
        for text in texts:
            assert restored.segment_text(text, include_confidence=True) == model.segment_text(text, include_confidence=True)
        assert restored.student.train_context_encoder == trainable


def test_encoder_batch_gradient_is_averaged_once_and_included_in_clipping():
    single, double = make_model(), make_model()
    cfg = StudentTrainingConfig(train_context_encoder=True, phase_lr=0, reg=0, max_grad_norm=None)
    for model in (single, double):
        model.student.configure(cfg)
    seq = single.student.build_sequences(["a b"], [["a", " ", "b"]])[0]
    _, grads, _ = single.student._sequence_gradients(seq, cfg)
    doubled = double.student._zero_grad()
    double.student._accumulate(doubled, grads)
    double.student._accumulate(doubled, grads)
    single.student._apply_gradients(grads, cfg, 1)
    double.student._apply_gradients(doubled, cfg, 2)
    assert single.encoder.export_state() == double.encoder.export_state()
    zeros = single.student._zero_grad()
    zeros["encoder"]["Wq.0"][0, 0] = 5
    assert single.student._grad_norm(zeros) == 5


def test_unsupported_trainable_encoder_is_rejected_before_mutation():
    model = make_model()
    model.student.encoder_adapter = None
    before = copy.deepcopy(model.student.export_state())
    with pytest.raises(ValueError, match="attached context encoder"):
        model.student.train(["ab"], [["a", "b"]], StudentTrainingConfig(train_context_encoder=True))
    assert model.student.export_state() == before


def test_empty_backward_and_invalid_tape():
    encoder = SpectralTransformerAdapter(d_model=4, n_layers=1, n_heads=2)
    output, tape = encoder.forward_with_cache(np.zeros((0, 4)), [])
    gradient, parameters = encoder.backward(output, tape)
    assert gradient.shape == (0, 4)
    assert all(np.count_nonzero(value) == 0 for value in parameters.values())
    with pytest.raises(ValueError, match="does not match"):
        encoder.backward(np.zeros((2, 4)), tape)
