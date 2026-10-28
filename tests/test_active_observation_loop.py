from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.aif_core import (
    ActionSpace, ActiveInferenceAgent, AgentConfig, ExecutionObservation, GaussianBelief,
    CEMPlanner, EFEEngine, QuadraticPreference, SimpleTreePlanner,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.dynamics import DynamicsConfig, LatentDynamicsModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.gwm_bridge import AITGWMBridge
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def model():
    return OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19)


@pytest.mark.parametrize("planner", ["tree", "cem"])
def test_planning_and_prediction_do_not_train_assimilate_or_advance_phases(planner):
    ait = model()
    ait.dynamics.cfg = replace(ait.dynamics.cfg, min_observations=1, train_interval=1, epochs=1)
    ait.observe_transition(np.zeros(4), "SeekEvidence", np.ones(4), np.array([0.1, 0.2, -0.1, 0.3]),
                           observation_id=1, source="delivered_test_observation")
    assert ait.dynamics.is_ready()
    bridge = AITGWMBridge(ait, ait.encode("新しい観測を待つ"))
    agent = ActiveInferenceAgent(4, ActionSpace(ait.policies, ait.policy_vecs), ait.goal_vec,
                                bridge.step_fn, bridge.ctx_from_prefix, bridge.r3_from_prefix,
                                AgentConfig(horizon=2, planner=planner, cem_K=3, cem_iters=1))
    before = copy.deepcopy(ait.state_dict())
    agent_before = agent.state_dict()
    first, second = agent.plan(), agent.plan()
    assert first == second
    assert ait.state_dict() == before
    assert agent.state_dict() == agent_before
    assert all(term["evidence"] == "forecast_only" for term in first.terms)


def test_untrained_forecast_is_persistence_with_no_new_evidence():
    ait = model()
    mean = np.array([0.1, 0.2, 0.3, 0.4])
    before = copy.deepcopy(ait.state_dict())
    result = ait.forecast_transition(mean, np.eye(4), "SeekEvidence", np.ones(4))
    np.testing.assert_array_equal(result["mean"], mean)
    np.testing.assert_allclose(result["covariance"], 1.05 * np.eye(4))
    assert result["source"] == "untrained_persistence_forecast_only"
    assert result["is_observation"] is False
    assert ait.state_dict() == before


def simple_agent(**kwargs):
    def forecast(prior, action, context):
        return GaussianBelief(np.array([1.0]), np.array([[2.0]])), np.array([1.0]), np.array([[0.5]])
    return ActiveInferenceAgent(1, ActionSpace(["look"], {"look": np.ones(1)}), np.zeros(1),
                                forecast, lambda prefix: np.zeros(1), lambda prefix: 0,
                                AgentConfig(horizon=1, planner="tree"), **kwargs)


def test_execution_installs_actual_next_state_posterior_and_correct_entropy():
    agent = simple_agent()
    old_entropy = agent.belief.entropy()
    observation = ExecutionObservation(1, np.array([3.0]), np.array([[0.5]]), "sensor")
    receipt = agent.act_and_update("look", observation=observation)
    # Predicted prior (1, 2) combined with observation (3, .5): K=.8.
    np.testing.assert_allclose(agent.belief.mu, [2.6])
    np.testing.assert_allclose(agent.belief.Sigma, [[0.4]])
    assert receipt["prior_H"] == old_entropy
    assert receipt["predicted_H"] > receipt["prior_H"] > receipt["post_H"]
    assert receipt["innovation_squared"] == pytest.approx(4 / 2.5)
    assert receipt["information_gain_nats"] > 0
    state = copy.deepcopy(agent.state_dict())
    with pytest.raises(ValueError, match="strictly increase"):
        agent.act_and_update("look", observation=observation)
    assert agent.state_dict() == state
    restored = simple_agent()
    restored.load_state_dict(json.loads(json.dumps(state)))
    assert restored.state_dict() == state


def test_observation_callback_is_never_executed_during_planning():
    calls = []
    def observe(prior, action, context, predicted):
        calls.append(action)
        return ExecutionObservation(1, np.array([3.0]), np.array([[0.5]]), "test_environment")
    agent = simple_agent(observation_fn=observe)
    action = agent.plan().actions[0]
    assert calls == []
    agent.act_and_update(action)
    assert calls == ["look"]
    with pytest.raises(ValueError, match="requires a delivered observation"):
        simple_agent().act_and_update("look")


def test_failed_execution_hook_retains_delivery_and_original_forecast_without_rereading():
    calls = {name: 0 for name in ("context", "forecast", "observation", "hook")}
    def observe(prior, action, context, predicted):
        calls["observation"] += 1
        return ExecutionObservation(1, np.array([3.0]), np.array([[0.5]]), "sensor")
    def hook(prior, action, context, observation, posterior):
        calls["hook"] += 1
        np.testing.assert_array_equal(context, [0.0])
        if calls["hook"] == 1:
            raise RuntimeError("learning temporarily failed")
        return {"accepted": observation.observation_id}
    agent = simple_agent(observation_fn=observe, on_observation=hook)
    context_provider, forecast = agent.ctx_provider, agent.step_fn
    def context(prefix):
        calls["context"] += 1
        return context_provider(prefix)
    def predict(*args):
        calls["forecast"] += 1
        return forecast(*args)
    agent.ctx_provider, agent.step_fn = context, predict
    before = copy.deepcopy(agent.state_dict())
    with pytest.raises(RuntimeError, match="temporarily failed"):
        agent.act_and_update("look")
    assert agent.state_dict()["belief"] == before["belief"]
    assert agent.last_observation_id == 0 and agent.pending_action == "look"
    owned = agent.pending_observation
    owned.mean[:] = -99
    assert agent.pending_observation.mean.tolist() == [3.0]
    # A pending external delivery cannot be replaced, or credited to another act.
    agent.action_space.names.append("other")
    with pytest.raises(ValueError, match="pending execution"):
        agent.act_and_update("other")
    with pytest.raises(ValueError, match="different observation"):
        agent.act_and_update("look", observation=ExecutionObservation(1, np.array([4.0]), np.array([[0.5]]), "sensor"))
    receipt = agent.act_and_update("look")
    assert calls == {"context": 1, "forecast": 1, "observation": 1, "hook": 2}
    assert receipt["learning"] == {"accepted": 1}
    assert agent.last_observation_id == 1 and agent.pending_action is None and agent.pending_observation is None
    np.testing.assert_allclose(agent.belief.mu, [2.6])


@pytest.mark.parametrize("retry", ["same_delivery", "implicit", "checkpoint"])
def test_mutating_failed_hook_cannot_change_the_retained_delivery_or_retry_posterior(retry):
    calls = {"source": 0, "hook": 0}
    original = ExecutionObservation(1, np.array([3.0]), np.array([[0.5]]), "sensor")
    def source(*args):
        calls["source"] += 1
        return original
    def hook(prior, action, context, observation, posterior):
        calls["hook"] += 1
        observation.mean[:] = 99
        observation.covariance[:] = 999
        prior.mu[:] = -99
        context[:] = -99
        posterior.mu[:] = -99
        if calls["hook"] == 1:
            raise RuntimeError("mutated hook failure")
        return {"accepted": observation.observation_id}
    agent = simple_agent(observation_fn=source, on_observation=hook)
    with pytest.raises(RuntimeError, match="mutated hook failure"):
        agent.act_and_update("look")
    assert calls == {"source": 1, "hook": 1}
    np.testing.assert_array_equal(original.mean, [3.0])
    np.testing.assert_array_equal(agent.pending_observation.mean, [3.0])
    np.testing.assert_array_equal(agent.pending_observation.covariance, [[0.5]])
    assert agent.last_observation_id == 0
    if retry == "checkpoint":
        saved = json.loads(json.dumps(agent.state_dict()))
        restored = simple_agent(observation_fn=lambda *args: pytest.fail("source reread"), on_observation=hook)
        restored.load_state_dict(saved)
        agent = restored
    result = agent.act_and_update("look", **({"observation": original} if retry == "same_delivery" else {}))
    assert result["learning"] == {"accepted": 1}
    assert result["obs_mu_norm"] == 3.0
    assert calls == {"source": 1, "hook": 2}
    np.testing.assert_allclose(agent.belief.mu, [2.6])
    np.testing.assert_allclose(agent.belief.Sigma, [[0.4]])
    assert agent.last_observation_id == 1 and agent.pending_action is None


def test_successful_mutating_hook_also_receives_an_independent_delivery():
    original = ExecutionObservation(1, np.array([3.0]), np.array([[0.5]]), "sensor")
    def hook(prior, action, context, observation, posterior):
        observation.mean[:] = -99
        observation.covariance[:] = 999
        return {"accepted": observation.observation_id}
    agent = simple_agent(on_observation=hook)
    result = agent.act_and_update("look", observation=original)
    np.testing.assert_array_equal(original.mean, [3.0])
    np.testing.assert_array_equal(original.covariance, [[0.5]])
    np.testing.assert_allclose(agent.belief.mu, [2.6])
    np.testing.assert_allclose(agent.belief.Sigma, [[0.4]])
    assert result["obs_mu_norm"] == 3.0 and agent.last_observation_id == 1


def test_pending_execution_checkpoint_resumes_without_callback_or_reforecast():
    observation = ExecutionObservation(1, np.array([3.0]), np.array([[0.5]]), "sensor")
    def fail(*args):
        raise RuntimeError("retry required")
    agent = simple_agent(on_observation=fail)
    with pytest.raises(RuntimeError):
        agent.act_and_update("look", observation=observation)
    saved = json.loads(json.dumps(agent.state_dict()))
    restored = simple_agent(observation_fn=lambda *args: pytest.fail("observation acquired again"))
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    damaged = copy.deepcopy(saved)
    damaged["pending_execution"]["prior"]["mean"] = [99.0]
    with pytest.raises(ValueError, match="prior/context mismatch"):
        restored.load_state_dict(damaged)
    assert restored.state_dict() == saved
    restored.step_fn = lambda *args: pytest.fail("pending transition reforecast")
    restored.ctx_provider = lambda *args: pytest.fail("pending context recomputed")
    restored.act_and_update("look")
    assert restored.last_observation_id == 1 and restored.pending_action is None
    np.testing.assert_allclose(restored.belief.mu, [2.6])


@pytest.mark.parametrize("version", [0, 3, True, 1.0])
def test_agent_checkpoint_rejects_unknown_or_untyped_versions_atomically(version):
    agent = simple_agent()
    state = agent.state_dict()
    invalid = copy.deepcopy(state)
    invalid["version"] = version
    invalid["belief"]["mean"] = [100.0]
    with pytest.raises(ValueError, match="checkpoint version"):
        agent.load_state_dict(invalid)
    assert agent.state_dict() == state


def test_cem_keeps_best_evaluated_sequence_when_coordinate_mode_is_worse(monkeypatch):
    # The two sampled good paths are AB and BA. Coordinate modes form AA,
    # which was not sampled and incurs a large second-step cost.
    class Sampling:
        def __init__(self):
            self.indices = iter((0, 1, 1, 0))
        def choice(self, count, p):
            return next(self.indices)
    monkeypatch.setattr(np.random, "default_rng", lambda seed: Sampling())
    space = ActionSpace(["A", "B"], {"A": np.ones(1), "B": -np.ones(1)})
    planner = CEMPlanner(EFEEngine(QuadraticPreference(np.zeros(1), np.eye(1))), space,
                         horizon=2, K=2, elite_frac=1, iters=1)
    def context(prefix):
        return np.array([10.0 if prefix == ["A", "A"] else 0.0])
    def forecast(prior, action, value):
        return GaussianBelief(value.copy(), np.eye(1)), value.copy(), np.eye(1)
    result = planner.plan(GaussianBelief(np.zeros(1), np.eye(1)), forecast, context, lambda prefix: 0)
    assert result.actions in (["A", "B"], ["B", "A"])


def test_planner_quadratic_risk_integrates_state_and_sensor_uncertainty():
    space = ActionSpace(["look"], {"look": np.ones(1)})
    planner = SimpleTreePlanner(EFEEngine(QuadraticPreference(np.zeros(1), np.eye(1))),
                                space, horizon=1, branch_limit=1)
    def forecast(prior, action, context):
        return GaussianBelief(np.zeros(1), np.array([[2.0]])), np.zeros(1), np.array([[0.5]])
    result = planner.plan(GaussianBelief(np.zeros(1), np.eye(1)), forecast,
                           lambda prefix: np.zeros(1), lambda prefix: 0)
    # E[(o-0)^2]/2 = (P+R)/2, but posterior variance is P*R/(P+R).
    assert result.terms[0]["risk"] == pytest.approx(0.5 * (2.0 + 0.5))
    assert result.terms[0]["epistemic"] == pytest.approx(0.5 * np.log(1 + 2.0 / 0.5))


def test_real_transition_is_once_and_freeze_prevents_parameter_updates():
    ait = model()
    ait.dynamics.cfg = replace(ait.dynamics.cfg, min_observations=1, train_interval=1, epochs=1)
    before = copy.deepcopy(ait.dynamics.export_state())
    observed = np.array([0.1, 0.2, 0.3, 0.4])
    receipt = ait.observe_transition(np.zeros(4), "SeekEvidence", np.ones(4), observed,
                                    observation_id=1, source="incoming_text", train=False)
    assert receipt["recorded"] and not receipt["training"]["trained"]
    for key in ("W1", "b1", "W_mu", "b_mu", "W_sigma", "b_sigma"):
        assert ait.dynamics.export_state()[key] == before[key]
    assert ait.dynamics.recorded_observations == 1
    np.testing.assert_array_equal(ait.dynamics.buffer[0][1], observed)
    assert all(len(history) == 1 for history in ait._phi_hist.values())
    state = copy.deepcopy(ait.state_dict())
    with pytest.raises(ValueError, match="strictly increase"):
        ait.observe_transition(np.zeros(4), "SeekEvidence", np.ones(4), observed,
                               observation_id=1, source="incoming_text")
    assert ait.state_dict() == state
    restored = model()
    restored.load_state_dict(json.loads(json.dumps(state)))
    assert restored.state_dict() == state
    with pytest.raises(ValueError, match="strictly increase"):
        restored.observe_transition(np.zeros(4), "SeekEvidence", np.ones(4), observed,
                                    observation_id=1, source="incoming_text")
    snapshot = ait.state_dict()
    ait.observe_transition(np.zeros(4), "SeekEvidence", np.ones(4), observed,
                           observation_id=2, source="incoming_text", train=False)
    assert snapshot["transition_observation_id"] == 1
    assert all(len(history) == 1 for history in snapshot["phi_hist"].values())


def test_simulated_delivery_never_enters_real_learning_or_phase_history():
    ait = model()
    before = copy.deepcopy(ait.state_dict())
    receipt = ait.observe_transition(np.zeros(4), "SeekEvidence", np.ones(4), np.ones(4),
                                    observation_id=1, source="explicit_simulation", simulated=True)
    assert receipt["simulated"] and not receipt["recorded"]
    assert ait.dynamics.export_state() == before["dynamics"]
    assert ait._phi_hist == before["phi_hist"]
    assert ait.last_observation_id == 1


def test_failed_transition_learning_rolls_back_without_consuming_delivery(monkeypatch):
    ait = model()
    ait.dynamics.cfg = replace(ait.dynamics.cfg, min_observations=1)
    before = copy.deepcopy(ait.state_dict())
    handle = ait.dynamics
    def fail():
        ait.dynamics.W1[:] = 0
        raise RuntimeError("training failed")
    monkeypatch.setattr(ait.dynamics, "train", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        ait.observe_transition(np.zeros(4), "SeekEvidence", np.ones(4), np.ones(4),
                               observation_id=1, source="incoming_text")
    assert ait.dynamics is handle
    assert ait.state_dict() == before


def test_dynamics_gaussian_nll_gradients_and_duplicate_batch_scaling():
    cfg = DynamicsConfig(hidden_dim=4, min_observations=1, epochs=1, batch_size=1, max_grad_norm=0.01)
    one, two = LatentDynamicsModel(2, 2, seed=8, cfg=cfg), LatentDynamicsModel(2, 2, seed=8, cfg=replace(cfg, batch_size=2))
    args = (np.array([0.1, -0.2]), np.array([0.4, 0.3]), np.array([0.2, -0.1]), np.array([0.4, -0.3]))
    one.record(*args)
    two.record(*args)
    two.record(*args)
    loss, gradients = one._loss_and_gradients(one.buffer)
    epsilon = 1e-6
    for name, gradient in gradients.items():
        parameter = getattr(one, name)
        index = np.unravel_index(np.abs(gradient).argmax(), parameter.shape)
        old = parameter[index]
        parameter[index] = old + epsilon
        plus = one._loss_and_gradients(one.buffer)[0]
        parameter[index] = old - epsilon
        minus = one._loss_and_gradients(one.buffer)[0]
        parameter[index] = old
        assert gradient[index] == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-6, rel=1e-4), name
    one.train(); two.train()
    for name in gradients:
        np.testing.assert_allclose(getattr(one, name), getattr(two, name), atol=1e-12, rtol=1e-12)
    restored = LatentDynamicsModel(2, 2, seed=80)
    restored.load_state(json.loads(json.dumps(one.export_state())))
    assert restored.export_state() == one.export_state()


def test_legacy_dynamics_are_not_claimed_as_real_observation_training():
    ait = model()
    state = ait.state_dict()
    for key in ("version", "config", "recorded_observations", "trained_observations", "variance_link", "buffer"):
        state["dynamics"].pop(key)
    state["dynamics"]["trained_steps"] = 10
    for key in ("transition_observation_id", "R3_mix", "R2_time"):
        state.pop(key)
    state["phi_hist"] = {key: [0.1, 0.2, 0.3] for key in ("AB", "BC", "CA")}
    ait.load_state_dict(state)
    assert not ait.dynamics.is_ready()
    assert ait.dynamics.variance_link == "legacy_sigmoid_proxy_v1"
    assert not any(ait._phi_hist.values())


@pytest.mark.parametrize("corruption", ["observation_counter", "phase_history", "dynamics_counter"])
def test_observation_checkpoint_preflight_preserves_existing_model(corruption):
    ait = model()
    before = copy.deepcopy(ait.state_dict())
    damaged = copy.deepcopy(before)
    if corruption == "observation_counter":
        damaged["transition_observation_id"] = True
    elif corruption == "phase_history":
        damaged["phi_hist"]["AB"] = [float("nan")]
    else:
        damaged["dynamics"]["recorded_observations"] = 1.5
    damaged["goal_vec"] = [9.0] * 4
    with pytest.raises(ValueError):
        ait.load_state_dict(damaged)
    assert ait.state_dict() == before
