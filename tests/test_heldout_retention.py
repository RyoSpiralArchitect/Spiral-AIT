from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path

import pytest

from scripts import benchmark_retention_guard as runner
from scripts import benchmark_slavic_learning as old
from spiralreality_AIT_onepass_aifcore_integrated.integrated.heldout_corpus import HeldoutSentence, load_heldout_sources, parse_heldout_conllu, select_unseen
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def conllu(text="Пам’ять", *, document=True, identifier="s1"):
    return (("# newdoc id = source-A\n" if document else "")
            + f"# sent_id = {identifier}\n# text = {text}\n1\t{text}\t_\t_\t_\t_\t0\troot\t_\t_\n\n")


def test_fresh_parser_preserves_unicode_and_document_absence():
    text = "Пам’ятьʼа\u0301\u2028"
    rows, rejected = parse_heldout_conllu(conllu(text), "uk", "UD_Ukrainian-IU")
    assert rows[0].text == "".join(rows[0].segments) == text and not rejected
    missing, _ = parse_heldout_conllu(conllu(document=False), "zh", "UD_Chinese-GSDSimp")
    assert missing[0].document_id is None
    with pytest.raises(ValueError, match="Unique"):
        parse_heldout_conllu(conllu() + conllu(document=False), "uk", "UD_Ukrainian-IU")


def test_fresh_overlap_guard_precedes_caps_and_never_repairs_originals():
    def row(identifier, text):
        return HeldoutSentence("uk", "UD_Ukrainian-IU", identifier, "doc", text, (text,))
    examples = [row("1", "OLD"), row("2", "new"), row("3", " NEW "), row("4", "Пам’ять")]
    chosen, receipt = select_unseen(examples, [" old "], cap=1, max_chars=100, seed=1)
    assert [r.text for r in chosen] == ["Пам’ять"]
    assert {r["reason"] for r in receipt["exclusions"]} == {"overlap_with_any_prior_source", "ambiguous_normalized_supervision"}
    with pytest.raises(ValueError, match="Insufficient"):
        select_unseen(examples, ["old"], cap=2, max_chars=100, seed=1)
    assert select_unseen(list(reversed(examples)), ["old"], cap=1, max_chars=100, seed=1) == (chosen, receipt)


def manifest(tmp_path):
    source = {"language": "uk", "source_id": "UD_Ukrainian-IU", "revision": "a" * 40, "files": {}}
    for filename, content in (("uk_iu-ud-test.conllu", conllu()), ("README.md", "Attribution"), ("LICENSE.txt", "CC BY-NC-SA 4.0")):
        data = content.encode()
        source["files"][filename] = {"url": f"https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-IU/{'a' * 40}/{filename}",
                                    "bytes": len(data), "sha256": old.hashlib.sha256(data).hexdigest()}
        destination = tmp_path / "cache" / "uk" / ("a" * 40) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "sources": [source]}))
    return path


def test_heldout_loader_is_offline_and_detects_byte_tampering(tmp_path):
    path = manifest(tmp_path)
    rows, receipt = load_heldout_sources(path, tmp_path / "cache")
    assert len(rows) == 1 and receipt["sources"][0]["source_documents"] == 1
    target = next((tmp_path / "cache").rglob("*.conllu"))
    target.write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        load_heldout_sources(path, tmp_path / "cache")
    target.unlink()
    with pytest.raises(FileNotFoundError, match="explicitly"):
        load_heldout_sources(path, tmp_path / "cache")


def test_training_worker_and_completed_receipt_validate_real_continuation(tmp_path, monkeypatch):
    for key, value in old.THREADS.items():
        monkeypatch.setenv(key, value)
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19)
    state = model.state_dict()
    row = {"language": "uk", "text": "аб вг", "segments": ["аб", " ", "вг"]}
    payload = {"slavic_train": [row], "rehearsal": [{**row, "language": "zh"}]}
    source, inputs = tmp_path / "source.json", tmp_path / "input.json"
    old.immutable_json(source, state); old.immutable_json(inputs, payload)
    cfg = json.loads((runner.ROOT / "configs/slavic-learning-pilot.json").read_text())
    settings = old.training_settings(state, cfg); settings["epochs"] = 1
    task = {"contract_sha256": "frozen", "seed": 19, "source_sha256": runner.fingerprints(), "runtime": runner.runtime(),
            "initial_checkpoint": str(source), "initial_checkpoint_sha256": old.file_hash(source),
            "initial_parameter_sha256": old.digest(old.parameter_state(old.load_model(state, 19))),
            "input": str(inputs), "input_sha256": old.digest(payload), "training": settings,
            "output_checkpoint": str(tmp_path / "trained.json"), "receipt": str(tmp_path / "receipt.json")}
    task_path = tmp_path / "task.json"; old.immutable_json(task_path, task)
    runner.train_worker(task_path)
    receipt = runner.read(task["receipt"])
    runner.validate_training(receipt, task, payload)
    assert receipt["processed_examples"] == 2
    assert receipt["initial_component_sha256"]["phase"] == receipt["final_component_sha256"]["phase"]
    receipt["final_parameter_sha256"] = "corrupt"
    with pytest.raises(ValueError, match="mismatch"):
        runner.validate_training(receipt, task, payload)
