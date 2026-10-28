from dataclasses import asdict

import pytest

from scripts import demo_observation_world as demo
from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import ActiveContextConfig
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def test_demo_replays_provided_text_with_frozen_weights_and_explicit_new_code(monkeypatch):
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy")
    state = model.state_dict()
    channel = ContextAcquisitionModel()
    parent = {"contract_sha256": "historical-source-contract"}
    monkeypatch.setattr(demo, "load_parent_study", lambda path: (state, channel, parent, asdict(ActiveContextConfig())))
    text = "Read evidence. 文脈を見る。观察证据。"
    result = demo.replay("unused-fixture", text)
    assert result["done"] and "".join(result["tokens"]) == text
    assert result["parent_study"] == parent
    assert "scripts/demo_observation_world.py" in result["code_sha256"]
    assert result["config"]["observation_model"] == "bayesian_regime"
    assert not result["config"]["learn_dynamics"]
    assert result["metrics"]["world_model_updates"] == 0


def test_demo_preserves_existing_output_receipts(tmp_path, monkeypatch):
    destination = tmp_path / "receipt.json"
    destination.write_text("original receipt")
    monkeypatch.setattr(demo, "replay", lambda *args, **kwargs: {"new": "receipt"})
    with pytest.raises(SystemExit):
        demo.main(["--output", str(destination)])
    assert destination.read_text() == "original receipt"
