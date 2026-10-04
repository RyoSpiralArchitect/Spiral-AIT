import numpy as np

from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def test_cached_encoding_restores_matching_diagnostics_and_metadata(monkeypatch):
    model = OnePassAIT(latent_dim=16, encoder_layers=2, seed=7)
    first = model.encode("Hello world.")
    expected_diagnostics = model.gate_diagnostics()
    expected_attention = [attention.copy() for attention in model.last_attention]
    model.encode("Another, much longer, input changes the diagnostic trace.")
    assert model.gate_diagnostics().gate_trace != expected_diagnostics.gate_trace

    def unexpected_forward(*args, **kwargs):
        raise AssertionError("A cache hit must not recompute the encoder")

    monkeypatch.setattr(model.encoder, "forward", unexpected_forward)
    cached = model.encode("Hello world.")
    assert cached.keys() == first.keys()
    for key, value in first.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(cached[key], value)
        else:
            assert cached[key] == value
    assert model.gate_diagnostics() == expected_diagnostics
    for actual, expected in zip(model.last_attention, expected_attention):
        np.testing.assert_array_equal(actual, expected)
    assert len(model.last_attention) == len(expected_attention)


def test_cached_and_fresh_results_cannot_mutate_diagnostics_or_cache():
    model = OnePassAIT(latent_dim=16, encoder_layers=1, seed=7)
    first = model.encode("Hello world.")
    expected_diagnostics = model.gate_diagnostics()
    expected_mask = first["gate_mask"].copy()
    expected_attention = model.last_attention[0].copy()
    first["gate_mask"][:] = 99
    first["gate_pos"][:] = 99
    assert model.gate_diagnostics() == expected_diagnostics

    cached = model.encode("Hello world.")
    np.testing.assert_array_equal(cached["gate_mask"], expected_mask)
    cached["gate_mask"][:] = -99
    cached["gate_pos"][:] = -99
    assert model.gate_diagnostics() == expected_diagnostics
    model.last_attention[0][:] = 99

    model.encode("Hello world.")
    assert model.gate_diagnostics() == expected_diagnostics
    np.testing.assert_array_equal(model.last_attention[0], expected_attention)


def test_empty_encoding_clears_previous_diagnostics_and_keeps_metadata():
    model = OnePassAIT(latent_dim=16, encoder_layers=1, seed=7)
    first = model.encode("Hello world.")
    empty = model.encode("")
    diagnostics = model.gate_diagnostics()
    assert diagnostics.gate_trace == []
    assert diagnostics.attention_strength == []
    assert diagnostics.mask_energy == 0.0
    assert empty.keys() == first.keys()
    assert empty["H"].shape == (0, 16)
    assert empty["gate_mask"].shape == (0, 0)
    for key, value in model.student.backend_metadata().items():
        assert empty[key] == value
