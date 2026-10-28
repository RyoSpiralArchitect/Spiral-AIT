import copy
import json
import math

import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel


def trained_channel():
    model = ContextAcquisitionModel(action_chars=(4, 16), probability_edges=(0, 1), age_edges=(0,),
                                    outcome_edges=(0, 0.5, 1), alpha=1, min_per_label=2)
    # The real context extension is informative for action 4, and uninformative
    # for action 16. More acquired text is not assigned an automatic reward.
    for label in (0, 1):
        for _ in range(9):
            model.record(0.5, 0.9 if label else 0.1, label, 2, 4)
        for after in (0.1, 0.9):
            for _ in range(5):
                model.record(0.5, after, label, 2, 16)
    return model


def test_channel_matches_analytical_binary_observation_model():
    model = trained_channel()
    result = model.estimate(0.5, 2, 4)
    # Symmetric smoothing of [9, 0] / [0, 9] gives error 1/11.
    error = 1 / 11
    posterior_entropy = -error * math.log(error) - (1 - error) * math.log(1 - error)
    assert result["eligible"]
    assert result["support_per_label"] == [9, 9]
    assert result["expected_risk"] == pytest.approx(error)
    assert result["predicted_posterior_entropy"] == pytest.approx(posterior_entropy)
    assert result["information_gain"] == pytest.approx(math.log(2) - posterior_entropy)
    assert result["outcome_probabilities"] == pytest.approx([0.5, 0.5])
    assert result["posterior_probabilities"] == pytest.approx([1 / 11, 10 / 11])
    uninformative = model.estimate(0.5, 2, 16)
    assert uninformative["expected_risk"] == pytest.approx(0.5)
    assert uninformative["information_gain"] == pytest.approx(0)


@pytest.mark.parametrize("prior", [0, 0.01, 0.25, 0.5, 0.9, 1])
def test_estimate_obeys_bayes_martingale_risk_and_information_bounds(prior):
    result = trained_channel().estimate(prior, 2, 4)
    assert sum(result["outcome_probabilities"]) == pytest.approx(1)
    assert sum(q * p for q, p in zip(result["outcome_probabilities"], result["posterior_probabilities"])) == pytest.approx(prior)
    assert 0 <= result["expected_risk"] <= min(prior, 1 - prior)
    assert 0 <= result["information_gain"] <= result["current_entropy"]
    assert result["predicted_posterior_entropy"] + result["information_gain"] == pytest.approx(result["current_entropy"])


@pytest.mark.parametrize("prior", [0, 0.01, 0.25, 0.5, 0.9, 1])
def test_filter_prior_is_distinct_from_the_observed_crf_conditioning_state(prior):
    model = ContextAcquisitionModel(action_chars=(4,), probability_edges=(0, 0.5, 1),
                                    age_edges=(0,), outcome_edges=(0, 0.5, 1), alpha=1, min_per_label=2)
    for label in (0, 1):
        for _ in range(9):
            model.record(0.9, 0.9 if label else 0.1, label, 2, 4)
    result = model.estimate(prior, 2, 4, conditioning_probability=0.9)
    assert result["eligible"]
    assert result["state_bin"]["probability"] == 1
    assert result["prior_probability"] == prior
    assert result["conditioning_probability"] == 0.9
    assert sum(result["outcome_probabilities"]) == pytest.approx(1)
    assert sum(q * p for q, p in zip(result["outcome_probabilities"], result["posterior_probabilities"])) == pytest.approx(prior)
    assert 0 <= result["expected_risk"] <= min(prior, 1 - prior)
    assert 0 <= result["information_gain"] <= result["current_entropy"]
    # A low latent prior must not accidentally route the future step into the
    # unobserved low-CRF-marginal state.
    if prior < 0.5:
        assert not model.estimate(prior, 2, 4)["eligible"]


def test_invalid_conditioning_probability_is_rejected_without_state_changes():
    model = trained_channel()
    before = model.export_state()
    for value in (float("nan"), float("inf"), -0.1, 1.1, True):
        with pytest.raises(ValueError, match="conditioning_probability"):
            model.estimate(0.5, 2, 4, conditioning_probability=value)
    assert model.export_state() == before


def test_unsupported_pair_has_zero_gain_and_requires_both_actual_labels():
    model = ContextAcquisitionModel(min_per_label=2)
    for _ in range(10):
        model.record(0.5, 0.95, 1, 2, 4)
    result = model.estimate(0.5, 2, 4)
    assert not result["eligible"]
    assert result["support_per_label"] == [0, 10]
    assert result["expected_risk"] == result["current_risk"] == 0.5
    assert result["information_gain"] == 0
    assert result["likelihoods"] == []
    model.record(0.5, 0.05, 0, 2, 4)
    assert not model.estimate(0.5, 2, 4)["eligible"]
    model.record(0.5, 0.05, 0, 2, 4)
    assert model.estimate(0.5, 2, 4)["eligible"]
    assert not model.estimate(0.5, 8, 4)["eligible"]
    assert not model.estimate(0.1, 2, 4)["eligible"]
    assert not model.estimate(0.5, 2, 16)["eligible"]


def test_endpoint_bins_snapshot_and_json_roundtrip_preserve_independent_state():
    model = ContextAcquisitionModel(min_per_label=1)
    model.record(1, 1, 1, 1000, 4)
    model.record(1, 0, 0, 1000, 4)
    before = model.export_state()
    snapshot = model.snapshot()
    restored = ContextAcquisitionModel.from_state(json.loads(json.dumps(before)))
    assert snapshot.frozen
    assert not model.frozen
    assert restored.export_state() == before
    assert restored.estimate(1, 1000, 4)["state_bin"] == {"probability": 4, "age": 4, "action_chars": 4}
    assert restored.estimate(1, 1000, 4)["information_gain"] == 0
    model.record(1, 0, 0, 1000, 4)
    assert snapshot.observations == 2
    assert model.observations == 3
    state = snapshot.export_state()
    state["counts"][0] = 100
    assert snapshot.export_state()["counts"][0] == 0
    with pytest.raises(RuntimeError, match="frozen"):
        snapshot.record(0.5, 0.9, 1, 2, 4)
    with pytest.raises(RuntimeError, match="frozen"):
        snapshot.load_state(before)


@pytest.mark.parametrize("mutation", [
    lambda state: state.update(version=True),
    lambda state: state.update(version=1.0),
    lambda state: state.update(version=999),
    lambda state: state.update(counts=state["counts"][:-1]),
    lambda state: state["counts"].__setitem__(0, -1),
    lambda state: state["counts"].__setitem__(0, float("nan")),
    lambda state: state["counts"].__setitem__(0, 0.0),
    lambda state: state["counts"].__setitem__(0, True),
    lambda state: state.update(observations=0),
    lambda state: state.update(frozen=1),
    lambda state: state["config"].update(alpha=float("inf")),
    lambda state: state["config"].update(min_per_label=True),
    lambda state: state.update(extra="unsupported"),
])
def test_malformed_checkpoints_are_rejected_atomically(mutation):
    model = trained_channel()
    before = model.export_state()
    invalid = copy.deepcopy(before)
    mutation(invalid)
    with pytest.raises(ValueError):
        model.load_state(invalid)
    assert model.export_state() == before


@pytest.mark.parametrize("arguments", [
    {"action_chars": ()}, {"action_chars": (4, 4)}, {"action_chars": (True,)},
    {"probability_edges": (0, 0.5, 0.4, 1)}, {"probability_edges": (0.1, 1)},
    {"age_edges": (1, 4)}, {"age_edges": (0, -1)}, {"alpha": float("nan")},
    {"alpha": 0}, {"min_per_label": 0},
])
def test_invalid_model_configuration_is_rejected(arguments):
    with pytest.raises(ValueError):
        ContextAcquisitionModel(**arguments)


@pytest.mark.parametrize("arguments", [
    (float("nan"), 0.5, 1, 2, 4), (0.5, 1.1, 1, 2, 4),
    (0.5, 0.9, 2, 2, 4), (0.5, 0.9, True, 2, 4),
    (0.5, 0.9, 1, -1, 4), (0.5, 0.9, 1, 2, 8),
])
def test_invalid_records_do_not_change_counts(arguments):
    model = ContextAcquisitionModel()
    before = model.export_state()
    with pytest.raises(ValueError):
        model.record(*arguments)
    assert model.export_state() == before
