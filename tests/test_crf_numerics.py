"""CRF probability and gradient checks at sharp and long-sequence settings."""

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.boundary import BoundaryStudent, StudentTrainingConfig
from spiralreality_AIT_onepass_aifcore_integrated.integrated.phase import PhaseBasisLearner


@pytest.mark.parametrize("length,scale", [(20, 5), (500, 100), (2000, 1000)])
def test_sharp_long_float32_marginals_remain_normalized(length, scale):
    student = BoundaryStudent(PhaseBasisLearner(dim=8), seed=19)
    rng = np.random.default_rng(11)
    student.transitions = rng.normal(0, scale, (2, 2)).astype(np.float32)
    logits = rng.normal(0, scale, length).astype(np.float32)
    labels = rng.integers(0, 2, length).tolist()
    loss, emissions, transitions, marginals = student._crf_loss(logits, labels)
    probabilities = np.asarray(marginals)
    assert np.isfinite(loss)
    assert np.all(np.isfinite(emissions))
    assert np.all(np.isfinite(transitions))
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-7)


def test_normalized_transition_expectations_match_loss_gradient():
    student = BoundaryStudent(PhaseBasisLearner(dim=8), seed=19)
    student.julia_backend = student.compiled_backend = None
    student.configure(StudentTrainingConfig(dtype="float64", use_encoder_context=False))
    student.transitions[:] = [[0.1, -0.2], [0.4, 0.3]]
    logits = [0.2, -0.7, 1.1, -0.3]
    labels = [0, 1, 1, 0]
    _, _, gradient, _ = student._crf_loss(logits, labels)
    for previous in (0, 1):
        for current in (0, 1):
            original = student.transitions[previous, current]
            student.transitions[previous, current] = original + 1e-6
            upper = student._crf_loss(logits, labels)[0]
            student.transitions[previous, current] = original - 1e-6
            lower = student._crf_loss(logits, labels)[0]
            student.transitions[previous, current] = original
            assert gradient[previous, current] == pytest.approx((upper - lower) / 2e-6, abs=1e-8)
