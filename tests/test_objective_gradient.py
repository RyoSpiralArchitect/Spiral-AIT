import numpy as np
import pytest
from dataclasses import replace

from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig


def configured_model():
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2,
                       encoder_backend="numpy", seed=19)
    cfg = StudentTrainingConfig(train_context_encoder=True, phase_lr=0,
                                reg=0.1, max_grad_norm=0.01, lexical_buckets=16,
                                dtype="float64", hidden_dim=4, emb_dim=4, context_hidden_dim=4)
    model.student.configure(cfg)
    model.student.transitions[:] = [[0.1, -0.2], [0.3, -0.1]]
    model.student.lexical_weights[:] = np.linspace(-0.2, 0.2, 16)
    return model, cfg


def parameters(model):
    student = model.student
    result = {name: getattr(student, name) for name in (
        "embeddings", "W_window", "b_window", "W_out", "b_out", "gate_w", "gate_b",
        "transitions", "ctx_W1", "ctx_b1", "ctx_w", "ctx_b", "lexical_weights")}
    result.update({"encoder/" + name: value for name, value in model.encoder.trainable_parameters().items()})
    return result


@pytest.mark.parametrize("changes", [{"reg": -1}, {"reg": float("nan")},
                                   {"max_grad_norm": -1}, {"max_grad_norm": float("inf")}])
def test_invalid_regularization_and_clipping_fail_before_reconfiguration(changes):
    model, cfg = configured_model()
    before = model.state_dict()
    with pytest.raises(ValueError):
        model.train_student(["ab"], [["a", "b"]], replace(cfg, **changes))
    assert model.state_dict() == before


def test_duplicate_batch_preserves_clipped_regularized_step_in_every_parameter():
    single, cfg = configured_model()
    duplicate, _ = configured_model()
    seq = single.student.build_sequences(["ab cd"], [["ab", " ", "cd"]])[0]
    _, gradients, _ = single.student._sequence_gradients(seq, cfg)
    doubled = duplicate.student._zero_grad()
    duplicate.student._accumulate(doubled, gradients)
    duplicate.student._accumulate(doubled, gradients)
    first = single.student._apply_gradients(gradients, cfg, 1)
    second = duplicate.student._apply_gradients(doubled, cfg, 2)
    assert 0 < first["gradient_clip_scale"] < 1
    assert first == second
    for name, value in parameters(single).items():
        np.testing.assert_array_equal(value, parameters(duplicate)[name], err_msg=name)


def test_mean_regularized_gradient_matches_reported_objective():
    model, cfg = configured_model()
    student = model.student
    examples = student.build_sequences(["ab cd", "猫です"], [["ab", " ", "cd"], ["猫", "です"]])
    accumulated = student._zero_grad()
    for seq in examples:
        _, grads, _ = student._sequence_gradients(seq, cfg)
        student._accumulate(accumulated, grads)
    gradients = student._mean_objective_gradient(accumulated, cfg, len(examples))

    def objective():
        nll = 0.0
        for seq in examples:
            logits, _ = student._forward_sequence(seq)
            nll += student._crf_loss(logits, student._labels_to_int(seq.labels))[0]
        return nll / len(examples) + 0.5 * cfg.reg * student._l2_norm()

    epsilon = 1e-5
    for name, parameter in parameters(model).items():
        if name.startswith("encoder/"):
            gradient = gradients["encoder"][name.split("/", 1)[1]]
        else:
            gradient = gradients[name]
        if np.ndim(parameter) == 0:
            old = getattr(student, name)
            setattr(student, name, old + epsilon)
            plus = objective()
            setattr(student, name, old - epsilon)
            minus = objective()
            setattr(student, name, old)
            actual = gradient
        else:
            index = np.unravel_index(np.abs(gradient).argmax(), parameter.shape)
            old = parameter[index]
            parameter[index] = old + epsilon
            plus = objective()
            parameter[index] = old - epsilon
            minus = objective()
            parameter[index] = old
            actual = gradient[index]
        assert actual == pytest.approx((plus - minus) / (2 * epsilon), rel=2e-4, abs=2e-6), name
