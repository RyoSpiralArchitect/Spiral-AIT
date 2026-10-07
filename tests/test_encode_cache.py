import numpy as np
import pytest

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


def test_encode_cache_evicts_least_recently_used_entries(monkeypatch):
    model = OnePassAIT(latent_dim=8, encoder_layers=1, encoder_heads=2, encode_cache_max_entries=2)
    model.student.use_encoder_context = False
    original = model.encoder.forward
    calls = []

    def counted_forward(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.encoder, "forward", counted_forward)
    first = model.encode("first")
    model.encode("other")
    model.encode("first")  # Refresh the first entry's recency.
    model.encode("third")
    assert list(model._encode_cache) == ["first", "third"]
    assert len(calls) == 3
    model.encode("other")  # Evicted entries are computed again.
    assert len(calls) == 4
    np.testing.assert_array_equal(model.encode("first")["H"], first["H"])


def test_encode_cache_byte_budget_includes_attention_and_skips_oversized_entries():
    probe = OnePassAIT(latent_dim=8, encoder_layers=2, encoder_heads=2)
    probe.encode("alpha")
    entry = probe._encode_cache["alpha"]
    payload_bytes = sum(array.nbytes for array in entry.values())
    attention_bytes = sum(array.nbytes for key, array in entry.items() if key.startswith("_attention_"))
    assert attention_bytes > 0
    model = OnePassAIT(latent_dim=8, encoder_layers=2, encoder_heads=2,
                       encode_cache_max_bytes=2 * payload_bytes - 1)
    for text in ("alpha", "bravo", "third"):
        model.encode(text)
        # Omitting attention from accounting would incorrectly admit two entries.
        assert list(model._encode_cache) == [text]
        assert sum(array.nbytes for arrays in model._encode_cache.values() for array in arrays.values()) == payload_bytes
    oversized = "too long to retain in this small cache"
    uncached = model.encode(oversized)
    assert uncached["H"].shape == (len(oversized), 8)
    assert list(model._encode_cache) == ["third"]


@pytest.mark.parametrize("option", ["encode_cache_max_entries", "encode_cache_max_bytes"])
def test_zero_cache_budget_disables_retention(option):
    model = OnePassAIT(latent_dim=8, encoder_layers=1, encoder_heads=2, **{option: 0})
    first = model.encode("recompute")
    second = model.encode("recompute")
    assert not model._encode_cache
    np.testing.assert_array_equal(first["H"], second["H"])


@pytest.mark.parametrize("option", ["encode_cache_max_entries", "encode_cache_max_bytes"])
@pytest.mark.parametrize("invalid", [-1, 0.5, True])
def test_cache_budgets_must_be_nonnegative_integers(option, invalid):
    with pytest.raises(ValueError, match="non-negative integer"):
        OnePassAIT(**{option: invalid})
