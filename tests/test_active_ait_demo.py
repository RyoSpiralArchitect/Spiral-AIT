import json

import pytest

from scripts import demo_active_ait as demo
from scripts.benchmark_active_context import fingerprints
from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import json_hash, sha256
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def study(tmp_path):
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19)
    state = model.state_dict()
    channel = ContextAcquisitionModel().freeze().export_state()
    config = {"model_seed": 19, "char_cost_candidates": [0.002], "active": {"trace_limit": 64}}
    contract = {"config": config, "code_sha256": fingerprints()}
    identity = json_hash(contract)
    plan = {"contract_sha256": identity, "contract": contract}
    selected = {"contract_sha256": identity, "char_cost": 0.002}
    fitted = {"contract_sha256": identity, "channel_sha256": json_hash(channel), "warm_transitions_sha256": "own-fixture"}
    checkpoint = json.dumps(state).encode()
    protected = {key: state[key] for key in ("student", "encoder", "phase")}
    warm = {"contract_sha256": identity, "checkpoint_sha256": sha256(checkpoint),
            "warm_transitions_sha256": "own-fixture", "protected_segmenter_sha256": json_hash(protected)}
    files = {"plan.json": plan, "channel.json": channel, "channel_fit.json": fitted,
             "development_selection.json": selected, "warm_start.json": warm,
             "summary.json": {"contract_sha256": identity, "selection": selected, "channel_fit": fitted, "world_warmup": warm}}
    for name, data in files.items():
        (tmp_path / name).write_text(json.dumps(data))
    (tmp_path / "warm_started_model.json").write_bytes(checkpoint)
    return tmp_path


def test_replay_uses_verified_frozen_artifacts_and_prints_actual_events(tmp_path, capsys):
    directory = study(tmp_path)
    receipt = demo.replay(directory, "Test text. 日本語。中文。")
    assert receipt["kind"] == "provided_text_replay"
    assert receipt["world_adaptation_enabled"]
    assert receipt["metrics"]["world_model_updates"] == 0
    assert "".join(receipt["tokens"]) == "Test text. 日本語。中文。"
    demo.describe(receipt)
    output = capsys.readouterr().out
    assert "provided_text_replay" in output and "expected_IG=" in output and "signed_ΔH=" in output
    assert "online_world_weight_updates=0" in output
    frozen = demo.replay(directory, "Test text. 日本語。中文。", freeze_world=True)
    assert not frozen["world_adaptation_enabled"]
    assert frozen["metrics"]["world_model_updates"] == 0


def test_missing_or_corrupted_artifacts_are_actionable_and_no_output_is_overwritten(tmp_path):
    with pytest.raises(ValueError, match="run scripts/benchmark_active_context.py"):
        demo.load_study(tmp_path)
    directory = study(tmp_path)
    checkpoint = directory / "warm_started_model.json"
    checkpoint.write_text(checkpoint.read_text() + " ")
    with pytest.raises(ValueError, match="checkpoint integrity"):
        demo.load_study(directory)
    study(tmp_path)
    target = tmp_path / "existing.json"
    target.write_text("preserve")
    with pytest.raises(SystemExit):
        demo.main(["--study", str(directory), "--text", "Test.", "--output", str(target)])
    assert target.read_text() == "preserve"
