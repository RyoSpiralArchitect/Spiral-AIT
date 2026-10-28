"""The immediate step path must retain the validated execution contract."""
from dataclasses import replace

import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import ActiveContextConfig, ActiveTextSession, StringTextSource
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def make_session(mode):
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=17)
    channel = ContextAcquisitionModel(min_per_label=1)
    for raw in (0.1, 0.3, 0.5, 0.7, 0.9):
        for age in channel.age_edges:
            for action in channel.action_chars:
                for _ in range(10):
                    channel.record(raw, 0.1, 0, age, action)
                    channel.record(raw, 0.9, 1, age, action)
    return ActiveTextSession(model, channel, ActiveContextConfig(
        observation_model=mode, learn_dynamics=False, max_window_chars=32, context_chars=8))


@pytest.mark.parametrize("mode", ["fixed_channel", "bayesian_regime"])
def test_immediate_steps_equal_separately_validated_plans_through_reframes_and_eof(mode):
    direct, explicit = make_session(mode), make_session(mode)
    text = "Observe, then act. 文脈を読んで判断する。观察之后再行动。" * 2
    first = direct.run(StringTextSource(text))
    source, tokens = StringTextSource(text), []
    for _ in range(1000):
        result = explicit.execute(explicit.plan(), source)
        tokens.extend(result["tokens"])
        if result["done"]:
            break
    else:
        pytest.fail("Explicit replay did not finish")
    assert first["done"] and first["tokens"] == tokens
    assert first["metrics"] == explicit.metrics
    assert first["trace"] == explicit.trace
    assert direct.model.state_dict() == explicit.model.state_dict()
    assert "".join(tokens) == text


def test_public_execution_still_rejects_old_and_forged_plans_without_reading():
    session = make_session("bayesian_regime")
    decision = session.plan()
    source = StringTextSource("An observation.")
    with pytest.raises(ValueError, match="stale"):
        session.execute(replace(decision, requested_chars=1), source)
    session.step(source)
    state = session.model.state_dict()
    with pytest.raises(ValueError, match="stale"):
        session.execute(decision, source)
    assert session.model.state_dict() == state


@pytest.mark.parametrize("hook", ["instance", "class", "subclass"])
def test_steps_still_dispatch_custom_execution_hooks(monkeypatch, hook):
    session = make_session("bayesian_regime")
    calls = []
    original = ActiveTextSession.execute
    if hook == "instance":
        def execute(decision, source=None, **kwargs):
            calls.append(decision)
            return original(session, decision, source, **kwargs)
        monkeypatch.setattr(session, "execute", execute)
    elif hook == "class":
        def execute(self, decision, source=None, **kwargs):
            calls.append(decision)
            return original(self, decision, source, **kwargs)
        monkeypatch.setattr(ActiveTextSession, "execute", execute)
    else:
        class InstrumentedSession(ActiveTextSession):
            def execute(self, decision, source=None, **kwargs):
                calls.append(decision)
                return super().execute(decision, source, **kwargs)
        session = InstrumentedSession(session.model, session.channel, session.config)
    text = "Read and update. 判断を更新する。"
    result = session.run(StringTextSource(text))
    assert result["done"] and "".join(result["tokens"]) == text
    assert len(calls) == result["metrics"]["events"] + 1  # Includes final Done.
    assert calls[-1].action == "Done"
