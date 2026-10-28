import copy
from fractions import Fraction
import json
import math

import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.observation_world import BayesianObservationWorld


def trained_channel():
    channel = ContextAcquisitionModel(action_chars=(4, 16), probability_edges=(0, 1),
                                      age_edges=(0,), outcome_edges=(0, 0.5, 1), alpha=1, min_per_label=2)
    for label in (0, 1):
        for _ in range(9):
            channel.record(0.5, 0.9 if label else 0.1, label, 2, 4)
        for outcome in (0.1, 0.9):
            for _ in range(5):
                channel.record(0.5, outcome, label, 2, 16)
    return channel


def entropy(probability):
    if probability in (0, 1):
        return 0.0
    return -probability * math.log(probability) - (1 - probability) * math.log1p(-probability)


def test_mixture_matches_exact_joint_enumeration_and_not_a_scalar_trust_update():
    world = BayesianObservationWorld(trained_channel())
    initial = world.initial_belief(0.25)
    result = world.estimate(initial, 0.5, 2, 4)
    assert result["outcome_probabilities"] == pytest.approx([53 / 88, 35 / 88])
    assert result["posterior_probabilities"] == pytest.approx([13 / 106, 31 / 70])
    assert [world.informative_probability(joint) for joint in result["joint_posteriors"]] == pytest.approx([31 / 53, 13 / 35])
    assert result["joint_likelihoods"][0] == result["joint_likelihoods"][1] == [0.5, 0.5]
    assert result["regime_information_gain"] > 0

    # A persistent hypothesis after two observed low outcomes can be computed
    # directly as prior(R,Y) * P(low|R,Y)^2. Sequential Bayes must match it.
    first = result["joint_posteriors"][0]
    second = world.estimate(first, 0.1, 6, 4)["joint_posteriors"][0]
    fractions = [Fraction(3, 8) * Fraction(1, 2) ** 2,
                 Fraction(1, 8) * Fraction(1, 2) ** 2,
                 Fraction(3, 8) * Fraction(10, 11) ** 2,
                 Fraction(1, 8) * Fraction(1, 11) ** 2]
    expected = [float(item / sum(fractions)) for item in fractions]
    assert second == pytest.approx(expected)
    label, regime = world.label_probability(first), world.informative_probability(first)
    falsely_independent = ((1 - regime) * (1 - label), (1 - regime) * label,
                          regime * (1 - label), regime * label)
    naive_second = world.estimate(falsely_independent, 0.1, 6, 4)["joint_posteriors"][0]
    assert abs(world.label_probability(naive_second) - world.label_probability(second)) > 0.01


@pytest.mark.parametrize("belief", [
    (0.5, 0.0, 0.5, 0.0), (0.0, 0.5, 0.0, 0.5),
    (0.25, 0.25, 0.25, 0.25), (0.0, 0.0, 0.8, 0.2),
    (0.01, 0.09, 0.89, 0.01), (0.2, 0.3, 0.4, 0.1),
])
def test_both_marginals_are_martingales_and_all_information_is_coherent(belief):
    world = BayesianObservationWorld(trained_channel())
    result = world.estimate(belief, 0.9, 7, 4)
    weights, joints = result["outcome_probabilities"], result["joint_posteriors"]
    assert math.fsum(weights) == pytest.approx(1)
    for cell in range(4):
        assert math.fsum(weight * posterior[cell] for weight, posterior in zip(weights, joints)) == pytest.approx(belief[cell])
    label, regime = world.label_probability(belief), world.informative_probability(belief)
    assert sum(weights[i] * world.label_probability(joints[i]) for i in range(2)) == pytest.approx(label)
    assert sum(weights[i] * world.informative_probability(joints[i]) for i in range(2)) == pytest.approx(regime)
    assert 0 <= result["expected_risk"] <= min(label, 1 - label)
    expected_hy = sum(weight * entropy(world.label_probability(posterior)) for weight, posterior in zip(weights, joints))
    expected_hr = sum(weight * entropy(world.informative_probability(posterior)) for weight, posterior in zip(weights, joints))
    assert result["information_gain"] == pytest.approx(entropy(label) - expected_hy, abs=1e-14)
    assert result["regime_information_gain"] == pytest.approx(entropy(regime) - expected_hr, abs=1e-14)
    # The retained two-row likelihood API must reproduce the same outcomes.
    for index, outcome in enumerate(weights):
        assert ((1 - label) * result["likelihoods"][0][index]
                + label * result["likelihoods"][1][index]) == pytest.approx(outcome)


@pytest.mark.parametrize("prior", [0, 0.01, 0.25, 0.5, 0.9, 1])
def test_certain_informative_regime_recovers_original_channel(prior):
    channel = trained_channel()
    world = BayesianObservationWorld(channel, informative_prior=1)
    result = world.estimate(world.initial_belief(prior), 0.9, 2, 4)
    baseline = channel.estimate(prior, 2, 4, conditioning_probability=0.9)
    for name in ("prior_probability", "current_risk", "current_entropy", "expected_risk", "information_gain",
                 "predicted_posterior_entropy", "outcome_probabilities", "posterior_probabilities"):
        assert result[name] == pytest.approx(baseline[name])
    for row, expected in zip(result["likelihoods"], baseline["likelihoods"]):
        assert row == pytest.approx(expected)
    assert result["regime_information_gain"] == 0


def test_null_is_label_independent_and_equal_label_balanced_not_count_pooled():
    channel = ContextAcquisitionModel(action_chars=(4,), probability_edges=(0, 1), age_edges=(0,),
                                      outcome_edges=(0, 0.5, 1), alpha=1, min_per_label=2)
    for _ in range(99):
        channel.record(0.5, 0.1, 0, 2, 4)
    for _ in range(9):
        channel.record(0.5, 0.9, 1, 2, 4)
    world = BayesianObservationWorld(channel, informative_prior=0)
    belief = world.initial_belief(0.2)
    result = world.estimate(belief, 0.5, 2, 4)
    expected_null = [(100 / 101 + 1 / 11) / 2, (1 / 101 + 10 / 11) / 2]
    assert result["outcome_probabilities"] == pytest.approx(expected_null)
    assert result["posterior_probabilities"] == pytest.approx([0.2, 0.2])
    assert result["information_gain"] == pytest.approx(0, abs=1e-14)
    assert result["expected_risk"] == pytest.approx(0.2)


def test_balanced_label_prior_does_not_identify_regime_in_one_observation():
    world = BayesianObservationWorld(trained_channel())
    result = world.estimate(world.initial_belief(0.5), 0.5, 2, 4)
    assert result["information_gain"] > 0
    assert result["regime_information_gain"] == pytest.approx(0, abs=1e-14)
    assert [world.informative_probability(joint) for joint in result["joint_posteriors"]] == pytest.approx([0.5, 0.5])
    # This is uncertainty, not an automatic correctness detector. Correlation
    # with the unknown label survives even though the regime marginal is fixed.
    joint = result["joint_posteriors"][0]
    assert joint[2] != pytest.approx(world.label_probability(joint) * 0.5)


def test_identical_label_rows_have_no_information_even_after_correlated_history():
    world = BayesianObservationWorld(trained_channel())
    joint = (0.05, 0.4, 0.45, 0.1)
    result = world.estimate(joint, 0.5, 2, 16)
    assert result["expected_risk"] == pytest.approx(0.5)
    assert result["information_gain"] == pytest.approx(0, abs=1e-14)
    assert result["regime_information_gain"] == pytest.approx(0, abs=1e-14)
    for posterior in result["joint_posteriors"]:
        assert posterior == pytest.approx(joint)


def test_unsupported_cells_have_no_forecast_and_snapshot_is_independent():
    channel = ContextAcquisitionModel(min_per_label=2)
    for _ in range(3):
        channel.record(0.5, 0.9, 1, 2, 4)
    world = BayesianObservationWorld(channel)
    before = copy.deepcopy(world.channel.export_state())
    belief = world.initial_belief(0.4)
    result = world.estimate(belief, 0.5, 2, 4)
    assert not result["eligible"]
    assert result["likelihoods"] == result["joint_likelihoods"] == result["joint_posteriors"] == []
    assert result["expected_risk"] == result["current_risk"] == 0.4
    assert result["information_gain"] == result["regime_information_gain"] == 0
    for _ in range(3):
        channel.record(0.5, 0.1, 0, 2, 4)
    assert not world.estimate(belief, 0.5, 2, 4)["eligible"]
    assert channel.estimate(0.4, 2, 4)["eligible"]
    assert world.channel.export_state() == before


def test_planning_and_json_roundtrip_cannot_mutate_world_or_belief():
    world = BayesianObservationWorld(trained_channel())
    before = world.channel.export_state()
    belief = list(world.initial_belief(0.3))
    saved = belief.copy()
    result = world.estimate(belief, 0.9, 4, 4)
    copied = json.loads(json.dumps(result))
    assert world.estimate(copied["joint_posteriors"][0], 0.1, 8, 4) == world.estimate(result["joint_posteriors"][0], 0.1, 8, 4)
    result["joint_likelihoods"][2][0] = 0
    result["likelihoods"][0][0] = 0
    assert world.channel.export_state() == before
    assert belief == saved
    assert json.loads(json.dumps(world.estimate(belief, 0.9, 4, 4))) == copied
    with pytest.raises(RuntimeError, match="frozen"):
        world.channel.record(0.5, 0.9, 1, 2, 4)


@pytest.mark.parametrize("belief", [None, {}, "0001", (0, 1), (0, 0, 0, 0), (0.2, 0.2, 0.2, 0.2),
                                        (True, 0, 0, 0), (-0.1, 0.1, 0.5, 0.5),
                                        (float("nan"), 0, 0, 1), (float("inf"), 0, 0, 0)])
def test_invalid_joint_states_reject_without_world_changes(belief):
    world = BayesianObservationWorld(trained_channel())
    before = world.channel.export_state()
    for operation in (world.label_probability, world.informative_probability,
                      lambda value: world.estimate(value, 0.5, 2, 4)):
        with pytest.raises(ValueError):
            operation(belief)
    assert world.channel.export_state() == before


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_initial_probabilities_reject(value):
    with pytest.raises(ValueError, match="informative_prior"):
        BayesianObservationWorld(trained_channel(), informative_prior=value)
    with pytest.raises(ValueError, match="probability"):
        BayesianObservationWorld(trained_channel()).initial_belief(value)


def test_channel_input_validation_remains_in_force():
    world = BayesianObservationWorld(trained_channel())
    belief = world.initial_belief(0.5)
    for raw, age, action in ((True, 2, 4), (float("nan"), 2, 4), (0.5, -1, 4), (0.5, 2, 8)):
        with pytest.raises(ValueError):
            world.estimate(belief, raw, age, action)
    with pytest.raises(TypeError, match="channel"):
        BayesianObservationWorld({})
