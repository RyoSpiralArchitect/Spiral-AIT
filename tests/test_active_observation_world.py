"""Causal integration of per-boundary world hypotheses with actual acquisition."""
import copy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import (
    ActiveContextConfig, ActiveTextSession, StringTextSource,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def setup_session(**values):
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=17)
    def inspect(text):
        p = 0.5 if len(text) == 8 else 0.9
        return {"tokens": [text] if text else [], "boundary_probabilities": [p] * max(0, len(text) - 1),
                "context_vector": np.array([len(text) / 96, 0.2, 0.3, 0.4])}
    model.student.observe_sequence = inspect
    channel = ContextAcquisitionModel(min_per_label=1)
    # The same genuinely labeled calibration relation is available in all bins.
    for raw in (0.1, 0.3, 0.5, 0.7, 0.9):
        for age in channel.age_edges:
            for count in channel.action_chars:
                for _ in range(30):
                    channel.record(raw, 0.1, 0, age, count)
                    channel.record(raw, 0.9, 1, age, count)
    settings = dict(observation_model="bayesian_regime", phase_gain=0, surprise_gain=0,
                    char_cost=0.0001, learn_dynamics=False, horizon=2)
    settings.update(values)
    return ActiveTextSession(model, channel, ActiveContextConfig(**settings))


def test_joint_world_plans_are_pure_and_future_suffix_cannot_change_them():
    sessions = [setup_session() for _ in range(2)]
    sources = [StringTextSource("abcdefgh" + suffix * 100) for suffix in ("x", "z")]
    for session, source in zip(sessions, sources):
        session.step(source)
    before_model = copy.deepcopy(sessions[0].model.state_dict())
    before_observation = copy.deepcopy(sessions[0]._observation)
    before_metrics = sessions[0].metrics
    a, b = sessions[0].plan(), sessions[1].plan()
    assert a == b == sessions[0].plan()
    assert a.action == "Acquire" and a.informative_probability == pytest.approx(0.5)
    assert sessions[0].model.state_dict() == before_model
    np.testing.assert_equal(sessions[0]._observation, before_observation)
    assert sessions[0].metrics == before_metrics


def test_real_deliveries_retain_label_regime_correlation_and_score_before_update():
    session = setup_session()
    source = StringTextSource("x" * 100)
    session.step(source)
    decision = session.plan()
    prior = session._observation["world_beliefs"][decision.focus_index]
    expected = session.observation_world.estimate(prior, decision.conditioning_probability,
                                                 decision.age, decision.requested_chars)
    session.execute(decision, source)
    outcome = session.trace[-1]["outcome_forecast"]
    index = outcome["outcome_bin"]
    joint = session._observation["world_beliefs"][decision.focus_index]
    assert joint == expected["joint_posteriors"][index]
    assert outcome["outcome_probability"] == pytest.approx(expected["outcome_probabilities"][index])
    assert outcome["outcome_nll"] == pytest.approx(-np.log(outcome["outcome_probability"]))
    probability = session.observation_world.label_probability(joint)
    # First balanced outcome can leave q(R) unchanged while correlating R and Y.
    independent = session.observation_world.initial_belief(probability)
    assert not np.allclose(joint, independent)
    assert session.metrics["observation_world_updates"] > 0
    assert session.metrics["outcome_forecasts"] == 1
    assert session.metrics["world_model_updates"] == 0
    assert session.metrics["outcome_nll_sum"] == pytest.approx(outcome["outcome_nll"])


def test_reset_regime_ablation_forgets_correlation_after_real_and_planned_observations():
    session = setup_session(learn_observation_regime=False)
    source = StringTextSource("x" * 100)
    session.step(source)
    decision = session.plan()
    session.execute(decision, source)
    for p, joint in zip(session._observation["boundary_probabilities"], session._observation["world_beliefs"]):
        assert joint == session.observation_world.initial_belief(p)
    assert session.metrics["observation_world_updates"] == 0
    assert session.metrics["channel_boundary_updates"] > 0
    observed = []
    original = session.observation_world.estimate
    def estimate(belief, *args):
        observed.append(belief)
        return original(belief, *args)
    with patch.object(session.observation_world, "estimate", side_effect=estimate):
        session.plan()
    assert observed
    for belief in observed:
        assert belief == session.observation_world.initial_belief(session.observation_world.label_probability(belief))


def test_world_regime_has_a_direct_path_to_action_value_and_null_has_no_label_gain():
    informative = setup_session(informative_prior=1.0)
    null = setup_session(informative_prior=0.0)
    for session in (informative, null):
        session.step(StringTextSource("x" * 100))
    assert informative.plan().action == "Acquire"
    assert null.plan().action == "Commit"
    assert all(candidate.expected_information_gain == 0 for candidate in null.plan().candidates)
    assert all(candidate.expected_model_information_gain == 0 for candidate in null.plan().candidates)


def test_failed_inspection_and_failed_world_learning_do_not_consume_joint_evidence_twice():
    session = setup_session()
    source = StringTextSource("x" * 100)
    session.step(source)
    decision = session.plan()
    before = copy.deepcopy(session._observation)
    metrics = session.metrics
    with patch.object(session.model, "observe_transition", side_effect=RuntimeError("learning failure")):
        with pytest.raises(RuntimeError, match="learning failure"):
            session.execute(decision, source)
    np.testing.assert_equal(session._observation, before)
    assert session.metrics == metrics
    assert session.plan() == decision
    session.execute(decision, source)
    assert session.metrics["outcome_forecasts"] == 1
    assert session.metrics["received_chars"] == 8 + decision.requested_chars
    assert session._source_observation_id == decision.observation_id


def test_partial_eof_does_not_update_regimes_or_score_unmatched_predictions():
    session = setup_session()
    source = StringTextSource("x" * 9)
    session.step(source)
    assert session.plan().action == "Acquire"
    session.step(source)
    assert session.metrics["observation_world_updates"] == 0
    assert session.metrics["outcome_forecasts"] == 0
    for p, joint in zip(session._observation["boundary_probabilities"], session._observation["world_beliefs"]):
        assert joint == session.observation_world.initial_belief(p)
    assert not session.trace[-1]["matched_forecast"]


def test_certain_informative_regime_reproduces_original_controller():
    sessions = [setup_session(informative_prior=1), setup_session(observation_model="fixed_channel")]
    results = [s.run(StringTextSource("x" * 36)) for s in sessions]
    assert results[0]["tokens"] == results[1]["tokens"]
    assert results[0]["metrics"]["policy_counts"] == results[1]["metrics"]["policy_counts"]
    for a, b in zip(results[0]["trace"], results[1]["trace"]):
        assert (a["action"], a["requested_chars"]) == (b["action"], b["requested_chars"])
        if a.get("observed_probability") is None:
            assert b.get("observed_probability") is None
        else:
            assert a["observed_probability"] == pytest.approx(b["observed_probability"])


@pytest.mark.parametrize("observation_model", ["fixed_channel", "bayesian_regime"])
def test_planning_cannot_spend_an_acquisition_beyond_the_execution_budget(observation_model):
    original = setup_session()
    delayed = ContextAcquisitionModel(action_chars=(4,), min_per_label=1)
    for raw in (0.1, 0.3, 0.5, 0.7, 0.9):
        for _ in range(50):
            # First read contains no label information; only the SECOND helps.
            for label in (0, 1):
                delayed.record(raw, 0.5, label, 4, 4)
                for age in (8, 16, 32):
                    delayed.record(raw, 0.1 if label == 0 else 0.9, label, age, 4)
    config = ActiveContextConfig(observation_model=observation_model, informative_prior=1,
                                 max_acquisitions_per_commit=2, action_chars=(4,), horizon=2,
                                 char_cost=0.01, call_cost=0, epistemic_weight=0,
                                 phase_gain=0, surprise_gain=0, learn_dynamics=False)
    session = ActiveTextSession(original.model, delayed, config)
    session.step(StringTextSource("x" * 100))
    # The initial delivery consumed one read, so only one acquisition remains.
    decision = session.plan()
    assert decision.action == "Commit"
    acquisition = next(candidate for candidate in decision.candidates if candidate.requested_chars == 4)
    assert acquisition.expected_risk == pytest.approx(0.5)
    assert acquisition.total == pytest.approx(0.54)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True, "0.5"])
def test_invalid_regime_priors_reject(value):
    with pytest.raises(ValueError, match="informative_prior"):
        ActiveContextConfig(informative_prior=value)


def test_configuration_normalizes_scalars_and_rejects_unknown_models():
    config = ActiveContextConfig(informative_prior=np.float64(0.25), model_epistemic_weight=np.float64(0.1))
    assert type(config.informative_prior) is float
    assert type(config.model_epistemic_weight) is float
    with pytest.raises(ValueError, match="observation_model"):
        replace(config, observation_model="unknown")
    with pytest.raises(ValueError, match="learn_observation_regime"):
        replace(config, learn_observation_regime=1)
