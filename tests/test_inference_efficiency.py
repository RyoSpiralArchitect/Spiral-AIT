"""Inference keeps the same emissions/marginals without backward allocations."""
import copy
from unittest.mock import patch

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig
from spiralreality_AIT_onepass_aifcore_integrated.integrated.encoder import SpectralTransformerAdapter


def model(*, context=True, dtype="float32", lexical_buckets=0):
    result = OnePassAIT(latent_dim=8, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=17)
    result.student.configure(StudentTrainingConfig(use_encoder_context=context, dtype=dtype,
                                                    lexical_buckets=lexical_buckets))
    result.student.julia_backend = result.student.compiled_backend = None
    return result


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("text", ["", "x", "A boundary in context.", "文脈を見て判断する。", "观察上下文，再更新判断。"])
def test_public_inference_exactly_matches_retained_training_forward(context, dtype, text):
    ait = model(context=context, dtype=dtype, lexical_buckets=32)
    student = ait.student
    before = copy.deepcopy(ait.state_dict())
    sequence = student.build_sequences([text], [[text]] if text else [[]])[0]
    logits, caches = student._forward_sequence(sequence, include_context_summary=True)
    _, _, _, marginals = student._crf_loss(logits, student._labels_to_int(sequence.labels))
    expected_probabilities = np.array([float(row[1]) for row in marginals])
    expected_tokens = student.decode_with_logit_bias(text, logits=logits)["tokens"]
    observed = student.observe_sequence(text)
    np.testing.assert_array_equal(observed["boundary_probabilities"], expected_probabilities)
    assert observed["tokens"] == expected_tokens == student.decode(text)["tokens"]
    np.testing.assert_array_equal(student.boundary_probs(text), expected_probabilities)
    np.testing.assert_array_equal(student._python_logits(text), logits)
    if caches and "context_summary" in caches[0]:
        np.testing.assert_array_equal(observed["context_vector"], caches[0]["context_summary"])
    else:
        np.testing.assert_array_equal(observed["context_vector"], np.zeros(ait.latent_dim))
    assert ait.state_dict() == before


def test_fresh_inference_reads_current_phase_once_and_does_not_keep_backward_tensors():
    ait = model(lexical_buckets=32)
    student = ait.student
    text = "abc def ghi"
    with patch.object(student, "_refresh_phase_features", wraps=student._refresh_phase_features) as refresh:
        student.observe_sequence(text)
    assert refresh.call_count == 1
    reference = student.build_sequences([text], [[text]])[0]
    logits, caches = student._forward_sequence(reference, include_context_summary=True)
    fresh = student._inference_sequence(text)
    fast_logits, fast_caches = student._forward_sequence(
        fresh, retain_boundary_cache=False, include_context_summary=True)
    assert logits == fast_logits
    assert len(fast_caches) == 1 and set(fast_caches[0]) == {"context_summary"}
    np.testing.assert_array_equal(fast_caches[0]["context_summary"], caches[0]["context_summary"])
    _, discarded = student._forward_sequence(fresh, retain_boundary_cache=False)
    assert discarded == []
    assert any("window" in cache and "hidden" in cache for cache in caches)


@pytest.mark.parametrize("length,scale", [(0, 1), (5, 2), (500, 100), (2000, 1000)])
def test_inference_crf_matches_sharp_training_marginals_and_loss_without_gradients(length, scale):
    student = model().student
    rng = np.random.default_rng(83)
    student.transitions = rng.normal(0, scale, (2, 2)).astype(np.float32)
    logits = rng.normal(0, scale, length).astype(np.float32)
    labels = rng.integers(0, 2, length).tolist()
    expected = student._crf_loss(logits, labels)
    actual = student._crf_loss(logits, labels, compute_gradients=False)
    assert actual[0] == expected[0]
    assert actual[1] is None and actual[2] is None
    np.testing.assert_array_equal(actual[3], expected[3])


def test_inference_does_not_stale_cached_training_features_or_encoder_backward_tapes():
    ait = model(dtype="float64", lexical_buckets=32)
    student = ait.student
    cfg = StudentTrainingConfig(train_context_encoder=True, phase_lr=0, reg=0,
                                dtype="float64", lexical_buckets=32)
    student.configure(cfg)
    text = "日本語 test"
    cached = student.build_sequences([text], [["日本", "語", " ", "test"]])[0]
    old_phases = cached.phases.copy()
    student.phase.apply_error(text, 2, error=3.0)
    student.observe_sequence(text)
    fresh = student.build_sequences([text], [["日本", "語", " ", "test"]])[0]
    old_result = student._sequence_gradients(cached, cfg)
    new_result = student._sequence_gradients(fresh, cfg)
    assert old_result[0] == new_result[0]
    assert not np.array_equal(old_phases, cached.phases)
    for key in old_result[1]:
        if key == "encoder":
            for name in old_result[1][key]:
                np.testing.assert_array_equal(old_result[1][key][name], new_result[1][key][name])
        else:
            np.testing.assert_array_equal(old_result[1][key], new_result[1][key])
    np.testing.assert_array_equal(old_result[2], new_result[2])
    _, caches = student._forward_sequence(fresh, retain_encoder_cache=True)
    assert "encoder_tape" in caches[0]
    with pytest.raises(ValueError, match="require boundary"):
        student._forward_sequence(fresh, retain_encoder_cache=True, retain_boundary_cache=False)


def _allocating_attention_reference(scores, bias):
    """Original allocating arithmetic, retained only as an equivalence oracle."""
    scores = scores + bias[None, :, :]
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    return weights / (weights.sum(axis=-1, keepdims=True) + 1e-12)


def _assert_tapes_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if key == "layers":
            assert len(left[key]) == len(right[key])
            for a, b in zip(left[key], right[key]):
                _assert_tapes_equal(a, b)
        else:
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize("dimension,heads,layers,length,dtype", [
    (4, 1, 0, 0, np.float32),
    (4, 1, 0, 5, np.float32),
    (4, 2, 1, 1, np.float32),
    (8, 2, 3, 17, np.float64),
    (16, 4, 2, 8, np.float16),
])
@pytest.mark.parametrize("mask_kind", ["none", "valid", "wrong_shape"])
def test_attention_workspace_matches_allocating_forward_diagnostics_tape_and_gradients(
        dimension, heads, layers, length, dtype, mask_kind):
    encoder = SpectralTransformerAdapter(d_model=dimension, n_layers=layers, n_heads=heads, seed=93)
    rng = np.random.default_rng(31)
    inputs = rng.normal(size=(length, dimension)).astype(dtype)
    gates = rng.uniform(size=max(0, length - 2)).astype(dtype)
    mask = (None if mask_kind == "none" else rng.normal(size=(length, length)).T
            if mask_kind == "valid" else rng.normal(size=(2, 3)))
    inputs_before, gates_before = inputs.copy(), gates.copy()
    mask_before = None if mask is None else mask.copy()
    # Read-only inputs catch accidental reuse of caller storage as workspace.
    inputs.setflags(write=False)
    gates.setflags(write=False)
    if mask is not None:
        mask.setflags(write=False)
    actual, actual_tape = encoder.forward_with_cache(inputs, gates, mask)
    actual_attention = [row.copy() for row in encoder.last_attn]
    actual_mask = encoder.last_gate_mask.copy()
    direction = rng.normal(size=actual.shape)
    actual_dx, actual_gradients = encoder.backward(direction, actual_tape)
    with patch.object(encoder, "_attention_weights", side_effect=_allocating_attention_reference):
        reference, reference_tape = encoder.forward_with_cache(inputs, gates, mask)
    reference_dx, reference_gradients = encoder.backward(direction, reference_tape)
    np.testing.assert_array_equal(actual, reference)
    _assert_tapes_equal(actual_tape, reference_tape)
    np.testing.assert_array_equal(actual_dx, reference_dx)
    for key in actual_gradients:
        np.testing.assert_array_equal(actual_gradients[key], reference_gradients[key])
    np.testing.assert_array_equal(actual_mask, encoder.last_gate_mask)
    for row, expected in zip(actual_attention, encoder.last_attn):
        np.testing.assert_array_equal(row, expected)
    np.testing.assert_array_equal(inputs, inputs_before)
    np.testing.assert_array_equal(gates, gates_before)
    if mask is not None:
        np.testing.assert_array_equal(mask, mask_before)
    np.testing.assert_array_equal(actual, encoder.forward(inputs, gates, mask))
