import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.evaluation_metrics import (
    aggregate_probability_sums, boundary_counts, fit_temperature, paired_document_interval,
    probability_sums, temperature_scale,
)


def test_nonwhitespace_metric_does_not_reward_easy_space_boundaries():
    text = "a b!"
    result = boundary_counts(text, ["a", " ", "b", "!"], ["a", " ", "b!"], exclude_whitespace=True)
    assert result == {"tp": 0, "fp": 0, "fn": 1, "f1": 0.0, "positions": 1}
    with pytest.raises(ValueError, match="reconstruct"):
        boundary_counts(text, [text], ["ab!"])


def test_calibration_uses_development_nll_and_pooled_bins():
    probabilities = [0.01, 0.99, 0.01, 0.99]
    labels = [0, 1, 1, 0]
    temperature = fit_temperature(probabilities, labels, [0.5, 1, 2, 4])
    assert temperature == 4
    raw = probability_sums(probabilities, labels)
    calibrated = probability_sums(temperature_scale(probabilities, temperature), labels)
    assert calibrated["negative_log_likelihood_sum"] < raw["negative_log_likelihood_sum"]
    combined = aggregate_probability_sums([probability_sums(probabilities[:2], labels[:2]), probability_sums(probabilities[2:], labels[2:])])
    expected = aggregate_probability_sums([raw])
    assert combined["bin_counts"] == expected["bin_counts"]
    for key in ("positions", "nll", "brier", "ece"):
        assert combined[key] == pytest.approx(expected[key])
    np.testing.assert_array_equal(temperature_scale(probabilities, 1), probabilities)


@pytest.mark.parametrize("invalid", [-1, 0, float("nan")])
def test_invalid_temperature_is_rejected(invalid):
    with pytest.raises(ValueError):
        temperature_scale([0.5], invalid)


def test_paired_interval_resamples_document_units():
    result = paired_document_interval([("doc-a", 0.1), ("doc-a", 0.1), ("doc-b", 0.1)])
    assert result["document_groups"] == 2
    assert result["delta_mean_f1"] == pytest.approx(0.1)
    assert result["document_bootstrap_95_percent"] == pytest.approx([0.1, 0.1])
