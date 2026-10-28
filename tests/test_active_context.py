import copy
from dataclasses import replace
import json
from unittest.mock import patch

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import (
    ActiveContextConfig, ActiveSessionError, ActiveTextSession, StringTextSource, TextObservation, marginal_tokens,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def model_with_observer(*, whole_token=False):
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=17)
    def observe(text):
        probabilities = [0.2 if whole_token else (0.5 if len(text) == 8 else 0.9)] * max(0, len(text) - 1)
        return {"tokens": [text] if text else [], "boundary_probabilities": probabilities,
                "context_vector": np.array([np.sin(len(text)), np.cos(len(text)), len(text) / 96, 0.2]),
                "mean_curvature": 0.0}
    model.student.observe_sequence = observe
    return model


def informative_channel():
    channel = ContextAcquisitionModel(min_per_label=1)
    for _ in range(30):
        # Deliberately reversed raw confidence shows a genuine likelihood update
        # rather than simply trusting the newly computed classifier probability.
        channel.record(0.5, 0.9, 0, 7, 16)
        channel.record(0.5, 0.1, 1, 7, 16)
        channel.record(0.5, 0.5, 0, 7, 4)
        channel.record(0.5, 0.5, 1, 7, 4)
    return channel


class CountedSource(StringTextSource):
    def __init__(self, text):
        super().__init__(text)
        self.calls = 0
    def __len__(self):
        pytest.fail("The actor must not inspect the hidden source length")
    def read(self, *args):
        self.calls += 1
        return super().read(*args)


def test_observation_uses_one_encoder_pass_and_interior_features_without_learning():
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy")
    first = np.zeros((4, 4))
    second = first.copy()
    second[1:3] = 2
    before = copy.deepcopy(model.state_dict())
    with patch.object(model.encoder, "forward", return_value=first) as forward:
        a = model.student.observe_sequence("abcd")
        assert forward.call_count == 1
    with patch.object(model.encoder, "forward", return_value=second):
        b = model.student.observe_sequence("abcd")
    assert not np.array_equal(a["context_vector"], b["context_vector"])
    assert model.state_dict() == before
    actual = model.student.observe_sequence("Hello.")
    expected = model.segment_text("Hello.", include_confidence=True)
    assert actual["tokens"] == expected["tokens"]
    np.testing.assert_array_equal(actual["boundary_probabilities"], expected["boundary_probabilities"])


def test_pure_plans_cannot_see_hidden_suffix_and_bayes_filter_uses_the_channel():
    models = [model_with_observer(), model_with_observer()]
    sessions = [ActiveTextSession(m, informative_channel(), ActiveContextConfig(char_cost=0.001)) for m in models]
    sources = [CountedSource("abcdefgh" + suffix * 70) for suffix in ("a", "z")]
    for session, source in zip(sessions, sources):
        session.step(source)
    before = copy.deepcopy(models[0].state_dict())
    channel_before = sessions[0].channel.export_state()
    decision = sessions[0].plan()
    assert sessions[0].plan() == decision == sessions[1].plan()
    assert decision.action == "Acquire" and decision.requested_chars == 16
    assert models[0].state_dict() == before
    assert sessions[0].channel.export_state() == channel_before
    assert sources[0].calls == 1
    sessions[0].execute(decision, sources[0])
    event = sessions[0].trace[-1]
    assert event["raw_observed_probability"] == 0.9
    assert event["observed_probability"] < 0.1
    assert event["matched_forecast"]
    assert event["channel_boundary_updates"] > 0
    assert models[0].last_observation_id == 2
    assert sessions[0].channel.export_state() == channel_before


def test_multistep_forecasts_keep_observed_state_separate_from_label_prior():
    session = ActiveTextSession(model_with_observer(), informative_channel(), ActiveContextConfig(char_cost=0.001))
    session.step(CountedSource("a" * 100))
    with patch.object(session.channel, "estimate", wraps=session.channel.estimate) as estimate:
        session.plan()
    assert any(abs(call.args[0] - call.kwargs["conditioning_probability"]) > 0.01 for call in estimate.call_args_list)
    for call in estimate.call_args_list[1:]:
        assert "conditioning_probability" in call.kwargs


def test_acquired_receipt_survives_inference_failure_without_rereading_or_learning():
    model = model_with_observer()
    session = ActiveTextSession(model, informative_channel(), ActiveContextConfig(char_cost=0.001))
    source = CountedSource("a" * 60)
    session.step(source)
    decision = session.plan()
    before = copy.deepcopy(model.state_dict())
    original = model.student.observe_sequence
    with patch.object(model.student, "observe_sequence", side_effect=RuntimeError("temporary inspection failure")):
        with pytest.raises(RuntimeError, match="temporary"):
            session.execute(decision, source)
    assert source.calls == 2
    assert model.state_dict() == before
    assert session.metrics["received_chars"] == 8
    model.student.observe_sequence = original
    session.execute(decision, source)
    assert source.calls == 2
    assert session.metrics["received_chars"] == 24
    assert model.last_observation_id == 2
    with pytest.raises(ValueError, match="stale"):
        session.execute(decision, source)


def test_truncated_eof_is_not_a_matched_information_forecast_and_short_non_eof_rejects():
    session = ActiveTextSession(model_with_observer(), informative_channel(), ActiveContextConfig(char_cost=0.001))
    source = CountedSource("a" * 11)
    session.step(source)
    assert session.plan().requested_chars == 16
    session.step(source)
    assert not session.trace[-1]["forecast_full_extension"]
    assert not session.trace[-1]["matched_forecast"]
    assert session.metrics["forecast_evaluations"] == 0
    assert session.metrics["truncated_acquisitions"] == 1
    second = ActiveTextSession(model_with_observer(), ContextAcquisitionModel())
    decision = second.plan()
    with pytest.raises(ValueError, match="Source must deliver"):
        second.execute(decision, observation=TextObservation(decision.observation_id, "a", False))


def test_forecast_precedes_read_and_inspection_and_only_arrivals_update_the_world():
    model = model_with_observer()
    model.dynamics.cfg.min_observations = 1
    model.dynamics.cfg.train_interval = 1
    model.dynamics.cfg.epochs = 1
    calls = []
    original_predict, original_inspect = model.predict_next, model.student.observe_sequence
    def predict(*args):
        calls.append("forecast")
        return original_predict(*args)
    def inspect(*args):
        calls.append("inspect")
        return original_inspect(*args)
    class Source(CountedSource):
        def read(self, *args):
            calls.append("read")
            return super().read(*args)
    model.predict_next, model.student.observe_sequence = predict, inspect
    before = copy.deepcopy(model.student.export_state())
    session = ActiveTextSession(model, ContextAcquisitionModel())
    result = session.run(Source("a" * 22))
    assert calls[:3] == ["forecast", "read", "inspect"]
    assert model.dynamics.recorded_observations == result["metrics"]["observations"]
    assert result["metrics"]["world_model_updates"] > 0
    assert model.student.export_state() == before
    assert "".join(result["tokens"]) == "a" * 22


def test_run_step_budget_and_errors_return_already_committed_tokens_for_continuation():
    session = ActiveTextSession(model_with_observer(), ContextAcquisitionModel())
    source = CountedSource("a" * 22)
    partial = session.run(source, max_steps=3)
    assert not partial["done"] and partial["tokens"]
    final = session.run(source)
    assert final["done"]
    assert "".join(partial["tokens"] + final["tokens"]) == "a" * 22
    second = ActiveTextSession(model_with_observer(), ContextAcquisitionModel())
    class BrokenSource(CountedSource):
        broken = True
        def read(self, *args):
            if self.calls == 1 and self.broken:
                self.broken = False
                raise OSError("temporarily unavailable")
            return super().read(*args)
    source = BrokenSource("a" * 22)
    with pytest.raises(ActiveSessionError) as failure:
        second.run(source)
    assert isinstance(failure.value.__cause__, OSError)
    recovered = second.run(source)
    assert "".join(failure.value.partial_result["tokens"] + recovered["tokens"]) == "a" * 22


def test_capacity_and_acquisition_budgets_bound_the_window_and_report_separate_forced_cuts():
    for budget, expected in ((20, "window_forced_cuts"), (1, "budget_forced_cuts")):
        cfg = ActiveContextConfig(initial_chars=8, action_chars=(4,), max_window_chars=12,
                                  context_chars=0, max_acquisitions_per_commit=budget, trace_limit=2)
        session = ActiveTextSession(model_with_observer(whole_token=True), ContextAcquisitionModel(action_chars=(4,)), cfg)
        result = session.run(CountedSource("a" * 50))
        assert "".join(result["tokens"]) == "a" * 50
        assert result["metrics"]["max_window_chars_seen"] <= 12
        assert result["metrics"][expected] > 0
        assert result["metrics"]["forced_splits"] == result["metrics"]["window_forced_cuts"] + result["metrics"]["budget_forced_cuts"]
        assert len(result["trace"]) <= 2 and result["metrics"]["trace_truncated"] > 0


def test_empty_source_done_delivery_validation_and_numpy_config_scalars_are_serializable():
    cfg = ActiveContextConfig(initial_chars=np.int64(8), context_chars=np.int64(2),
                              char_cost=np.float32(.002), trace_limit=np.int32(2))
    session = ActiveTextSession(model_with_observer(), ContextAcquisitionModel(), cfg)
    result = session.run(StringTextSource(""))
    assert result["tokens"] == [] and result["done"]
    assert session.model.last_observation_id == 0
    json.dumps(result)
    with pytest.raises(ValueError, match="completed"):
        session.execute(session.plan(), observation=TextObservation(1, "lost", True))
    assert marginal_tokens("abcd", [.4, .5, .9]) == ["ab", "c", "d"]
    assert marginal_tokens("", []) == []
