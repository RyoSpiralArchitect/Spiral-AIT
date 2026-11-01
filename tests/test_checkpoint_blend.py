from copy import deepcopy
import json

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.checkpoint_blend import blend_checkpoints
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig


def pair():
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=21)
    cfg = StudentTrainingConfig(epochs=1, validation_split=0, phase_lr=0,
                                lexical_buckets=16, train_context_encoder=True)
    model.train_student(["ab cd"], [["ab", " ", "cd"]], cfg)
    reference = deepcopy(model.state_dict())
    cfg.warm_start = True
    model.train_student(["аб вг"], [["аб", " ", "вг"]], cfg)
    return reference, model.state_dict()


@pytest.mark.parametrize("weight", [0, 1])
def test_blend_endpoints_are_exact_independent_copies(weight):
    reference, adapted = pair()
    result = blend_checkpoints(reference, adapted, adaptation_weight=weight)
    assert result == (reference if weight == 0 else adapted)
    result["student"]["W_out"][0] += 3
    assert result != (reference if weight == 0 else adapted)


def test_blend_interpolates_trained_weights_and_reloads_without_inference_ensemble():
    reference, adapted = pair()
    snapshots = deepcopy((reference, adapted))
    result = blend_checkpoints(reference, adapted, adaptation_weight=.75)
    assert (reference, adapted) == snapshots
    dtype = reference["student"]["architecture"]["dtype"]
    np.testing.assert_allclose(result["student"]["W_out"], .25 * np.array(reference["student"]["W_out"], dtype=dtype) + .75 * np.array(adapted["student"]["W_out"], dtype=dtype), rtol=0, atol=0)
    np.testing.assert_allclose(result["encoder"]["Wq"], .25 * np.array(reference["encoder"]["Wq"]) + .75 * np.array(adapted["encoder"]["Wq"]), rtol=0, atol=0)
    assert result["encoder"] == result["student"]["_encoder"]
    assert result["phase"] == reference["phase"] == adapted["phase"]
    assert result["dynamics"] == adapted["dynamics"]
    models = [OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy") for _ in range(2)]
    models[0].load_state_dict(result)
    models[1].load_state_dict(json.loads(json.dumps(result)))
    for text in ("Пам’ять і звʼязок.", "中文分词。", "日本語の文。", "English text."):
        assert "".join(models[0].segment_text(text)) == text
        assert models[0].segment_text(text, include_confidence=True) == models[1].segment_text(text, include_confidence=True)


@pytest.mark.parametrize("weight", [-.1, 1.1, True, float("nan"), float("inf"), "0.75"])
def test_invalid_weight_rejected_before_loading(weight):
    with pytest.raises(ValueError, match="adaptation_weight"):
        blend_checkpoints({}, {}, adaptation_weight=weight)


@pytest.mark.parametrize("kind", ["owner", "shape", "phase", "encoder_copy", "nonfinite", "lexical"])
def test_blend_rejects_incompatible_checkpoints_without_mutating_inputs(kind):
    reference, adapted = pair()
    if kind == "owner":
        adapted["student"]["fitted_backend"] = "compiled"
    elif kind == "shape":
        adapted["student"]["W_out"].append(0)
    elif kind == "phase":
        adapted["phase"][next(iter(adapted["phase"]))][0][0] += .1
    elif kind == "encoder_copy":
        adapted["encoder"]["Wq"][0][0][0] += 1
    elif kind == "nonfinite":
        adapted["student"]["b_out"] = float("inf")
    else:
        adapted["student"]["lexical"]["version"] = "unsupported"
    before = deepcopy((reference, adapted))
    with pytest.raises((ValueError, TypeError)):
        blend_checkpoints(reference, adapted, adaptation_weight=.75)
    assert (reference, adapted) == before
