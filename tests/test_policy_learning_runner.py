from argparse import Namespace
import json
from pathlib import Path
import subprocess

import pytest

from scripts import benchmark_policy_learning as runner
from scripts.context_training_worker import run as train_worker
from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import sha256


def test_development_selection_ignores_test_scores_and_prefers_small_rate_on_ties():
    candidates = [{"run_id": "slow", "encoder_lr": 0.001, "development": {"macro_f1": 0.8}, "test_f1": 0.0},
                  {"run_id": "fast", "encoder_lr": 0.05, "development": {"macro_f1": 0.8}, "test_f1": 1.0}]
    choice = runner.choose_on_development(candidates, 0.81)
    assert choice["run_id"] == "slow"
    assert not choice["development_prefers_context"]
    candidates[1]["development"]["macro_f1"] = 0.82
    assert runner.choose_on_development(candidates, 0.81)["development_prefers_context"]


def test_completed_checkpoint_and_worker_source_mismatches_fail_closed(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "training").mkdir()
    checkpoint = tmp_path / "checkpoints/run.json"
    checkpoint.write_text("original")
    receipt = {"run_id": "run", "contract_sha256": "contract", "checkpoint_sha256": sha256(checkpoint.read_bytes())}
    (tmp_path / "training/run.json").write_text(json.dumps(receipt))
    assert runner.completed_training(tmp_path, "run", "contract") == receipt
    checkpoint.write_text("modified")
    with pytest.raises(ValueError, match="checkpoint integrity"):
        runner.completed_training(tmp_path, "run", "contract")
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"implementation_root": str(tmp_path), "implementation_files": {"checkpoints/run.json": "wrong"}}))
    with pytest.raises(ValueError, match="Implementation differs"):
        train_worker(task)


def test_offline_study_runs_all_arms_selects_then_resumes_without_retraining(tmp_path, monkeypatch):
    sources = []
    for language in ("en", "ja", "zh"):
        revision = "a" * 40
        directory = tmp_path / "cache" / language / revision
        directory.mkdir(parents=True)
        blocks = []
        for index in range(16):
            forms = [language, "-", str(index)]
            lines = [f"# newdoc id = doc{index}", f"# sent_id = s{index}",
                     f"# parallel_id = pud/s{index}", "# text = " + "".join(forms)]
            lines += ["\t".join([str(i + 1), form] + ["_"] * 8) for i, form in enumerate(forms)]
            blocks.append("\n".join(lines))
        files = {f"{language}_pud-ud-test.conllu": "\n\n".join(blocks),
                 "README.md": "Own synthetic test fixture", "LICENSE.txt": "Own synthetic test fixture"}
        metadata = {}
        for name, content in files.items():
            data = content.encode()
            (directory / name).write_bytes(data)
            metadata[name] = {"bytes": len(data), "sha256": sha256(data), "url": "unused-offline"}
        sources.append({"language": language, "revision": revision, "files": metadata})
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"sources": sources}))
    config = json.loads((runner.ROOT / "configs/policy-learning-comparison.json").read_text())
    config.update(reference_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runner.ROOT, text=True).strip(),
                  fold_count=4, training_document_caps=[2], seeds=[19], encoder_learning_rates=[0.01])
    config["model"].update(latent_dim=4)
    config["training"].update(epochs=1, hidden_dim=4, emb_dim=4, context_hidden_dim=4, lexical_buckets=16)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "output"
    args = Namespace(config=config_path, sources=manifest, cache_dir=tmp_path / "cache", output=output,
                     publish=tmp_path / "published", workers=1, download=False, prepare_only=True)
    runner.run(args)
    assert not (output / "training").exists()
    args.prepare_only = False
    runner.run(args)
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["training_runs"]) == 3
    assert len(summary["development_selections"]) == 1
    assert (output / "development_selection.json").exists()
    for language in ("en", "ja", "zh"):
        assert set(runner.ARMS) <= summary["budgets"]["2"][language].keys()
        for arm in runner.ARMS:
            assert summary["budgets"]["2"][language][arm]["unique_test_sentences"] == 4
    # A matching resume must use the completed workers, and reproduce summaries.
    original_run = runner.subprocess.run
    def refuse_training(command, *args, **kwargs):
        if any(str(part).endswith("context_training_worker.py") for part in command):
            pytest.fail("unexpected training worker")
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(runner.subprocess, "run", refuse_training)
    args.publish = None
    runner.run(args)
    assert json.loads((output / "summary.json").read_text()) == summary
