import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import (
    OnePassAIT, StudentTrainingConfig,
)


def test_loading_replaces_cached_encoding_and_diagnostics():
    text = "Changed weights require fresh encodings."
    current = OnePassAIT(latent_dim=16, seed=10, encoder_layers=1)
    replacement = OnePassAIT(latent_dim=16, seed=20, encoder_layers=1)
    old = current.encode(text)["H"].copy()
    expected = replacement.encode(text)["H"]
    assert not np.allclose(old, expected)
    current.load_state_dict(replacement.state_dict())
    assert not current._encode_cache
    assert current.gate_diagnostics().gate_trace == []
    np.testing.assert_allclose(current.encode(text)["H"], expected)


def test_explicit_empty_training_data_does_not_train_on_default_corpus():
    model = OnePassAIT(latent_dim=16, encoder_layers=1)
    with pytest.raises(ValueError):
        model.train_student(texts=[], segments=[], cfg=StudentTrainingConfig(epochs=1))
