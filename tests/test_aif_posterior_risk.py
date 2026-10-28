from dataclasses import replace
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, SegmentationAIFConfig


@pytest.mark.parametrize("belief,expected", [([0.8, 0.2], "ProbeReliability"),
                                           ([0.2, 0.2], "ProbeMotivation"),
                                           ([0.9, 0.9], "DecideNow")])
def test_candidates_share_one_belief_and_match_enumerated_expected_error(belief, expected):
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy")
    tokens = {0.0: ["abc"], -0.35: ["a", "bc"], -0.10: ["ab", "c"], 0.35: ["a", "b", "c"]}
    decisions = {0.0: [0, 0], -0.35: [1, 0], -0.10: [0, 1], 0.35: [1, 1]}
    def marginals(text, bias, **kwargs):
        # Candidate-specific confidence is deliberately misleading. Only the
        # unchanged posterior is evidence for choosing between their decodes.
        return np.array(belief if bias == 0 else [0.99, 0.01])
    with patch.object(model.student, "boundary_probs_with_logit_bias", side_effect=marginals), \
         patch.object(model.student, "decode_with_logit_bias", side_effect=lambda text, bias, **kwargs: {"tokens": tokens[bias]}):
        result = model._select_policy_from_logits("abc", [0, 0], SegmentationAIFConfig())
        misleading_precision = replace(SegmentationAIFConfig(), obs_sigma_by_policy={"SeekEvidence": 1e-20},
                                       target_boundary_rate=0.99, efe_epistemic_w0=1000)
        assert model._select_policy_from_logits("abc", [0, 0], misleading_precision) == result
    assert result["chosen_policy"] == expected
    assert result["selected_risk"] <= result["baseline_risk"]
    assert result["evidence"] == "shared_crf_posterior_no_new_observation"
    for candidate in result["candidates"]:
        expected_error = 0.0
        for gold in product((0, 1), repeat=2):
            probability = np.prod([p if label else 1 - p for p, label in zip(belief, gold)])
            error = np.mean(np.array(gold) != decisions[candidate["logit_bias"]])
            expected_error += probability * error
        assert candidate["risk"] == pytest.approx(expected_error)
        assert candidate["epistemic"] == candidate["w"] == 0


def test_identical_decodes_prefer_zero_bias_and_legacy_is_explicit():
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy")
    with patch.object(model.student, "decode_with_logit_bias", return_value={"tokens": ["abc"]}):
        selected = model._select_policy_from_logits("abc", [0, 0], None)
        assert selected["chosen_policy"] == "ProbeMotivation"
    legacy = model.select_policy_aif("abc", SegmentationAIFConfig(selection_mode="legacy_efe"))
    assert legacy["selection_mode"] == "legacy_efe"
    assert all(row["epistemic"] > 0 for row in legacy["candidates"])


@pytest.mark.parametrize("options", [{"selection_mode": "unknown"}, {"marginal_temperature": 0},
                                    {"false_positive_cost": float("nan")}, {"false_negative_cost": -1},
                                    {"logit_bias_by_policy": {name: 0.1 for name in
                                      ("ProbeMotivation", "ProbeReliability", "SeekEvidence", "DecideNow")}}])
def test_invalid_scoring_contract_is_rejected(options):
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy")
    with pytest.raises(ValueError):
        model.select_policy_aif("abc", SegmentationAIFConfig(**options))
