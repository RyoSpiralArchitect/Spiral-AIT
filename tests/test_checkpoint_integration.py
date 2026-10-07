import copy
import json
from pathlib import Path

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


@pytest.mark.parametrize("backend", ["compiled", "julia"])
@pytest.mark.parametrize("rejection", ["legacy_ownership", "missing_backend"])
def test_rejected_checkpoint_preserves_public_model_state_and_cached_outputs(backend, rejection):
    text = "abcd efgh"
    current = OnePassAIT(latent_dim=8, encoder_layers=1, seed=11)
    current.student.julia_backend = current.student.compiled_backend = None
    replacement = OnePassAIT(latent_dim=8, encoder_layers=1, seed=23)
    replacement.phase.apply_error(text, 2, error=10.0)
    checkpoint = json.loads(json.dumps(replacement.state_dict()))
    # The inner legacy payload is an actual export from the base revision.
    fixture = json.loads((Path(__file__).parent / "fixtures" / "legacy_native_8840fee.json").read_text())
    checkpoint["student"] = fixture["states"][backend]
    if rejection == "missing_backend":
        checkpoint["student"]["fitted_backend"] = backend
        checkpoint["student"]["use_encoder_context"] = False
    expected_tokens = current.segment_text(text)
    expected_encoding = current.encode(text)
    before = copy.deepcopy(current.state_dict())
    before_cache = copy.deepcopy(current._encode_cache)
    before_diagnostics = current.gate_diagnostics()
    before_segment_metadata = copy.deepcopy(current._last_segment_metadata)
    assert not np.array_equal(checkpoint["goal_vec"], before["goal_vec"])
    assert checkpoint["phase"] != before["phase"]

    with pytest.raises(ValueError, match="ambiguous|requires its fitted"):
        current.load_state_dict(checkpoint)

    assert current.state_dict() == before
    assert current.gate_diagnostics() == before_diagnostics
    assert current._last_segment_metadata == before_segment_metadata
    assert list(current._encode_cache) == list(before_cache)
    for key, array in before_cache[text].items():
        np.testing.assert_array_equal(current._encode_cache[text][key], array)
    np.testing.assert_array_equal(current.encode(text)["H"], expected_encoding["H"])
    current._encode_cache.clear()
    np.testing.assert_array_equal(current.encode(text)["H"], expected_encoding["H"])
    assert current.segment_text(text) == expected_tokens
