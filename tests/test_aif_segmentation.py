from __future__ import annotations

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import TRAIN_TEXTS
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT

import numpy as np
import pytest
from unittest.mock import patch


def test_segment_text_with_aif_emits_policy_metadata() -> None:
    ait = OnePassAIT(latent_dim=16, seed=11)
    text = TRAIN_TEXTS[0]

    tokens = ait.segment_text(text, use_aif=True)
    assert isinstance(tokens, list)
    assert tokens, "segmentation should return at least one token"

    result = ait.segment_text(text, return_metadata=True, use_aif=True)
    assert isinstance(result, dict)
    assert "tokens" in result
    assert "chosen_policy" in result
    assert result["chosen_policy"] in ait.policies
    assert "aif" in result
    assert "candidates" in result["aif"]


def test_aif_reuses_one_encoder_pass_without_changing_policy_scores():
    ait = OnePassAIT(latent_dim=16, seed=11, encoder_layers=1)
    text = "One pass, several policies."
    selection = ait.select_policy_aif(text)
    expected = ait.student.decode_with_logit_bias(
        text,
        next(row["logit_bias"] for row in selection["candidates"]
             if row["policy"] == selection["chosen_policy"]),
    )
    with patch.object(ait.encoder, "forward", wraps=ait.encoder.forward) as forward:
        actual = ait.segment_text(text, return_metadata=True, use_aif=True)
        assert forward.call_count == 1
    assert actual["tokens"] == expected["tokens"]
    assert actual["aif"] == selection
    for row in selection["candidates"]:
        probs = ait.student.boundary_probs_with_logit_bias(text, row["logit_bias"])
        np.testing.assert_allclose(row["stats"], ait._segmentation_stats(text, probs))


@pytest.mark.parametrize("text", ["", "猫", "Short text."])
def test_supplied_logits_match_recomputed_crf_paths(text):
    ait = OnePassAIT(latent_dim=16, encoder_layers=1)
    logits = ait.student._python_logits(text)
    original = list(logits)
    for bias in (-0.35, 0.0, 0.35):
        np.testing.assert_allclose(
            ait.student.boundary_probs_with_logit_bias(text, bias),
            ait.student.boundary_probs_with_logit_bias(text, bias, logits=logits),
        )
        assert ait.student.decode_with_logit_bias(text, bias) == (
            ait.student.decode_with_logit_bias(text, bias, logits=logits)
        )
    assert logits == original
    with pytest.raises(ValueError):
        ait.student.decode_with_logit_bias(text, logits=logits + [1.0])
