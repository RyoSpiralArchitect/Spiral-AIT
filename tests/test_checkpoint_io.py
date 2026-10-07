import json
from pathlib import Path

import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated import checkpoint


def test_checkpoint_roundtrip_creates_parent_directories(tmp_path):
    destination = tmp_path / "nested" / "model.json"
    payload = {"text": "日本語 🌀", "weights": [[1.0, 2.0], [3.0, 4.0]]}
    checkpoint.save_checkpoint(destination, payload)
    assert checkpoint.load_checkpoint(destination) == payload
    assert list(destination.parent.iterdir()) == [destination]


def test_unserialisable_payload_preserves_previous_checkpoint(tmp_path):
    destination = tmp_path / "model.json"
    checkpoint.save_checkpoint(destination, {"version": "old"})
    previous = destination.read_bytes()
    with pytest.raises(TypeError):
        checkpoint.save_checkpoint(destination, {"invalid": object()})
    assert destination.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_replace_sees_a_complete_sibling_file(tmp_path, monkeypatch):
    destination = tmp_path / "model.json"
    checkpoint.save_checkpoint(destination, {"version": "old"})
    original_replace = checkpoint.os.replace
    replacements = []

    def inspect_replace(source, target):
        assert Path(source).parent == destination.parent
        assert Path(target) == destination
        assert json.loads(Path(source).read_text()) == {"version": "new"}
        assert checkpoint.load_checkpoint(destination) == {"version": "old"}
        replacements.append(source)
        return original_replace(source, target)

    monkeypatch.setattr(checkpoint.os, "replace", inspect_replace)
    checkpoint.save_checkpoint(destination, {"version": "new"})
    assert len(replacements) == 1
    assert checkpoint.load_checkpoint(destination) == {"version": "new"}
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("failure_point", ["fsync", "replace"])
def test_failed_write_preserves_old_checkpoint_and_removes_temporary(
    tmp_path, monkeypatch, failure_point,
):
    destination = tmp_path / "model.json"
    checkpoint.save_checkpoint(destination, {"version": "old"})
    previous = destination.read_bytes()

    def fail(*args):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(checkpoint.os, failure_point, fail)
    with pytest.raises(OSError, match="simulated storage failure"):
        checkpoint.save_checkpoint(destination, {"version": "new"})
    assert destination.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [destination]
