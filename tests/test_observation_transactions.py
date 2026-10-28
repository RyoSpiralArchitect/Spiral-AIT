"""Transaction rollback must preserve delivered-observation semantics."""
import copy
from dataclasses import replace
import math

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.dynamics import LatentDynamicsModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT
from spiralreality_AIT_onepass_aifcore_integrated.integrated.utils import unit


def model():
    result = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19)
    result.dynamics.cfg = replace(result.dynamics.cfg, min_observations=2, train_interval=2,
                                  max_buffer=4, epochs=1, batch_size=1)
    return result


def arguments(index=0):
    before = np.array([0.1, -0.2, 0.3, -0.4]) + index * 0.001
    context = np.array([-0.2, 0.1, 0.4, -0.3])
    after = before + np.array([0.01, 0.02, -0.01, -0.02])
    return before, "SeekEvidence", context, after


def observe(ait, *, train, index=0):
    return ait.observe_transition(*arguments(index), observation_id=ait.last_observation_id + 1,
                                  source="delivered_numeric_test_fixture", train=train)


def apply_without_transaction(ait, *, train, index=0):
    """Independent successful-path oracle: the specified three state updates."""
    before, policy, context, after = arguments(index)
    ait.dynamics.record(before, unit(ait.policy_vecs[policy]), context, after)
    training = {"trained": False}
    if train and ait.dynamics.should_train():
        training = ait.dynamics.train()
    ait.update_global_phases(after)
    ait.last_observation_id += 1
    return training


@pytest.mark.parametrize("train", [False, True])
def test_success_matches_direct_learning_across_training_cadence_and_replay_eviction(train):
    actual, expected = model(), model()
    for index in range(9):
        # Existing replay vectors are immutable inputs to recording and fitting.
        retained_rows = actual.dynamics.buffer.copy()
        for inputs, target in retained_rows:
            inputs.setflags(write=False)
            target.setflags(write=False)
        training = apply_without_transaction(expected, train=train, index=index)
        receipt = observe(actual, train=train, index=index)
        assert receipt["training"] == training
        assert actual.state_dict() == expected.state_dict()
        for actual_row, old_row in zip(actual.dynamics.buffer[:-1], retained_rows[-3:]):
            assert actual_row[0] is old_row[0]
            assert actual_row[1] is old_row[1]
    assert len(actual.dynamics.buffer) == 4
    assert actual.dynamics.recorded_observations == 9
    assert actual.dynamics.trained_steps == (4 if train else 0)


def test_stock_fit_failure_after_an_optimizer_update_restores_state_and_allows_same_id(monkeypatch):
    actual = model()
    observe(actual, train=False)
    expected = copy.deepcopy(actual)
    before = copy.deepcopy(actual.state_dict())
    handle, replay = actual.dynamics, actual.dynamics.buffer
    original_outer = np.outer
    calls = 0
    def fail_in_second_training_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        # Two replay rows make six initial loss outers, then three in the first
        # batch. Fail after those first optimizer assignments have happened.
        if calls == 10:
            assert not np.array_equal(actual.dynamics.W1, np.asarray(before["dynamics"]["W1"]))
            raise RuntimeError("second batch failed")
        return original_outer(*args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(np, "outer", fail_in_second_training_batch)
        with pytest.raises(RuntimeError, match="second batch failed"):
            observe(actual, train=True, index=1)
    assert calls == 10
    assert actual.dynamics is handle and actual.dynamics.buffer is replay
    assert actual.state_dict() == before
    assert observe(actual, train=True, index=1) == observe(expected, train=True, index=1)
    assert actual.state_dict() == expected.state_dict()


@pytest.mark.parametrize("train", [False, True])
def test_stock_phase_failure_restores_full_buffer_and_prior_training_state(monkeypatch, train):
    actual = model()
    for index in range(4):
        observe(actual, train=False, index=index)
    before = copy.deepcopy(actual.state_dict())
    handle, replay, first = actual.dynamics, actual.dynamics.buffer, actual.dynamics.buffer[0]
    original_atan2 = math.atan2
    calls = 0
    def fail_after_first_phase(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("phase calculation failed")
        return original_atan2(*args)
    with monkeypatch.context() as scoped:
        scoped.setattr(math, "atan2", fail_after_first_phase)
        with pytest.raises(RuntimeError, match="phase calculation failed"):
            observe(actual, train=train, index=5)
    assert calls == 2
    assert actual.dynamics is handle and actual.dynamics.buffer is replay
    assert actual.dynamics.buffer[0] is first
    assert actual.state_dict() == before


@pytest.mark.parametrize("class_hook", [False, True])
def test_custom_record_hooks_keep_deep_rollback_for_in_place_mutations(monkeypatch, class_hook):
    actual = model()
    observe(actual, train=False)
    before = copy.deepcopy(actual.state_dict())
    handle = actual.dynamics
    def corrupt_and_fail(*args, **kwargs):
        actual.dynamics.buffer[0][0][:] = 999
        actual.dynamics.W1[:] = 0
        actual.dynamics.cfg.lr = 1.5
        raise RuntimeError("custom record failed")
    with monkeypatch.context() as scoped:
        scoped.setattr(LatentDynamicsModel if class_hook else actual.dynamics, "record", corrupt_and_fail)
        with pytest.raises(RuntimeError, match="custom record failed"):
            observe(actual, train=False, index=1)
    assert actual.dynamics is handle
    assert actual.state_dict() == before


@pytest.mark.parametrize("class_hook", [False, True])
def test_custom_phase_hooks_keep_deep_rollback_for_arbitrary_dynamics_mutations(monkeypatch, class_hook):
    actual = model()
    observe(actual, train=False)
    before = copy.deepcopy(actual.state_dict())
    handle = actual.dynamics
    def corrupt_and_fail(*args, **kwargs):
        actual.dynamics.buffer[0][1][:] = -999
        actual.dynamics.W_mu[:] = 0
        actual.dynamics.trained_steps = 10
        actual._phi_hist["AB"].append(1.234)
        raise RuntimeError("custom phase failed")
    with monkeypatch.context() as scoped:
        scoped.setattr(OnePassAIT if class_hook else actual, "update_global_phases", corrupt_and_fail)
        with pytest.raises(RuntimeError, match="custom phase failed"):
            observe(actual, train=False, index=1)
    assert actual.dynamics is handle
    assert actual.state_dict() == before
