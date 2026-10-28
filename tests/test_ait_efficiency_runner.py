from argparse import Namespace
import copy
import json
import os
from pathlib import Path
import platform
import shutil

import numpy as np
import pytest

from scripts import benchmark_ait_efficiency as runner
from spiralreality_AIT_onepass_aifcore_integrated.integrated.acquisition import ContextAcquisitionModel
from spiralreality_AIT_onepass_aifcore_integrated.integrated.active_context import ActiveContextConfig
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


def config():
    result = json.loads((runner.ROOT / "configs/ait-efficiency-pilot.json").read_text())
    result["resource_probe_script"] = "scripts/benchmark_ait_efficiency.py"
    return result


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "system": platform.system(),
            "thread_environment": runner.THREADS, "rss_floor_bytes": 1000, "process_peak_rss_bytes": 2000}


def rows():
    result = []
    for language in runner.LANGUAGES:
        text = language + "-abcdefghijklmno"
        result.append({"language": language, "parallel_id": "p1", "document_group": "g1", "text": text,
                       "segments": [text], "characters": len(text), "source_sha256": runner.hashlib.sha256(text.encode()).hexdigest(),
                       "labels_sha256": runner.digest([text])})
    return result


def identity():
    return [{key: value for key, value in row.items() if key not in ("text", "segments")} for row in rows()]


def receipt(job, contract="fixture-contract", source=None):
    result = []
    for row in identity():
        timing = job["measurement"] == "timing"
        quality = {"tp": 0, "fp": 0, "fn": 0, "f1": 1.0, "positions": row["characters"] - 1}
        result.append({**row, "prediction_sha256": runner.digest([row["language"], job["mode"]]),
                       "behavior_sha256": runner.digest([job["mode"], "candidate plans", "metrics"]),
                       "state_sha256": runner.digest([job["mode"], "phase/replay/weights"]),
                       "boundary": quality, "nonwhitespace": quality.copy(), "episode_elapsed_ms": 12.0 if timing else None,
                       "python_peak_bytes": None if timing else 200, "python_retained_bytes": None if timing else 100})
    return {"contract_sha256": contract, "job": job, "source_sha256": source or {"module.py": "f" * 64},
            "results": result, "results_sha256": runner.digest(result), "runtime": runtime()}


def test_jobs_are_frozen_complete_and_memory_is_not_timed():
    settings = config()
    jobs = runner.jobs(settings)
    assert len(jobs) == 16 and jobs == runner.jobs(settings)
    assert len({runner.job_name(job) for job in jobs}) == 16
    assert sum(job["measurement"] == "timing" for job in jobs) == 12
    for job in jobs:
        value = receipt(job)
        runner.validate_receipt(value, job, identity(), "fixture-contract", {"module.py": "f" * 64}, runtime())
        if job["measurement"] == "memory":
            value["results"][0]["episode_elapsed_ms"] = 50.0
            value["results_sha256"] = runner.digest(value["results"])
            with pytest.raises(ValueError, match="cannot supply speed"):
                runner.validate_receipt(value, job, identity(), "fixture-contract", {"module.py": "f" * 64}, runtime())


@pytest.mark.parametrize("system,value,expected", [("Darwin", 12345, 12345), ("Linux", 12345, 12345 * 1024)])
def test_rss_units_are_platform_explicit(system, value, expected):
    assert runner.rss_bytes(value, system) == expected
    with pytest.raises(ValueError, match="only on Darwin and Linux"):
        runner.rss_bytes(value, "unknown")


def test_json_keys_roundtrip_and_immutable_receipts_do_not_change(tmp_path):
    value = {"histogram": {2: 1, 10: 2}, "signed_entropy": -0.25}
    path = tmp_path / "receipt.json"
    before = runner.digest(value)
    runner.immutable_json(path, value)
    assert runner.digest(json.loads(path.read_text())) == before
    runner.immutable_json(path, value)
    with pytest.raises(ValueError, match="Immutable"):
        runner.immutable_json(path, {"changed": True})
    with pytest.raises(ValueError):
        runner.immutable_json(tmp_path / "invalid.json", {"value": float("nan")})
    assert not (tmp_path / "invalid.json").exists()


def test_strict_equality_includes_candidates_metrics_and_final_continuation_state():
    values = [receipt(job) for job in runner.jobs(config())]
    runner.equality_gate(values)
    for field in ("prediction_sha256", "behavior_sha256", "state_sha256", "boundary", "nonwhitespace"):
        changed = copy.deepcopy(values)
        optimized = next(item for item in changed if item["job"]["variant"] == "optimized")
        optimized["results"][0][field] = "changed"
        with pytest.raises(ValueError, match="equality gate failed"):
            runner.equality_gate(changed)


def test_rehashed_incomplete_rows_wrong_quality_and_changed_runtime_reject():
    job = runner.jobs(config())[0]
    value = receipt(job)
    for mutate in (
        lambda item: item["results"].pop(),
        lambda item: item["results"][0]["boundary"].update(f1=0.2),
        lambda item: item["runtime"].update(numpy="different-version"),
    ):
        changed = copy.deepcopy(value)
        mutate(changed)
        changed["results_sha256"] = runner.digest(changed["results"])
        with pytest.raises(ValueError):
            runner.validate_receipt(changed, job, identity(), "fixture-contract", {"module.py": "f" * 64}, runtime())


def test_all_repeats_reported_without_picking_best_and_memory_separate():
    values = [receipt(job) for job in runner.jobs(config())]
    for item in values:
        if item["job"]["measurement"] == "timing":
            for row in item["results"]:
                row["episode_elapsed_ms"] = [30, 10, 20][item["job"]["repeat"]]
    result = runner.summarize(values, config())
    language = result["fixed_channel"]["variants"]["baseline"]["languages"]["en"]
    assert language["total_seconds_by_repeat"] == [0.03, 0.01, 0.02]
    assert language["median_total_seconds"] == 0.02
    assert language["median_python_peak_bytes"] == 200
    assert len(result["fixed_channel"]["variants"]["baseline"]["untraced_timing_workers"]) == 3


def fake_study(tmp_path, monkeypatch):
    settings = config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(settings))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    source = {"module.py": "f" * 64}
    monkeypatch.setattr(runner, "export_baseline", lambda *args: source)
    monkeypatch.setattr(runner, "source_hashes", lambda *args: source)
    monkeypatch.setattr(runner, "prepare_inputs", lambda *args: ({"rows": rows(), "state": "fixture"}, {"rows": identity()}))
    calls = []
    def execute(request_path, path, source_root, baseline_directory, job):
        calls.append(job)
        request = json.loads(Path(request_path).read_text())
        runner.immutable_json(path, receipt(job, request["contract_sha256"], source))
    monkeypatch.setattr(runner, "execute_worker", execute)
    args = Namespace(config=config_path, baseline=baseline, source_study="fixture", sources="fixture", cache_dir="fixture",
                     output=tmp_path / "result", prepare_only=True, publish=None)
    return args, calls


def test_full_prepare_run_publish_and_resume_cannot_silently_recompute(tmp_path, monkeypatch):
    args, calls = fake_study(tmp_path, monkeypatch)
    runner.run(args)
    assert not calls and not (args.output / "worker_input.json").exists()
    frozen = (args.output / "plan.json").read_bytes()
    args.prepare_only, args.publish = False, tmp_path / "public"
    runner.run(args)
    assert len(calls) == 16
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["exact_behavior_gate"] == "passed" and summary["episodes"] == 48
    assert (args.publish / "plan.json").read_bytes() == frozen
    assert len((args.publish / "paired_scores.jsonl").read_text().splitlines()) == 7
    args.publish = None
    monkeypatch.setattr(runner, "execute_worker", lambda *args: pytest.fail("Resumed study recomputed a worker"))
    runner.run(args)
    path = next((args.output / "workers").glob("*.json"))
    value = json.loads(path.read_text())
    value["results"].pop()
    value["results_sha256"] = runner.digest(value["results"])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Incomplete worker"):
        runner.run(args)


def test_fresh_worker_imports_each_requested_source_and_preserves_exact_behavior(tmp_path):
    # Separate immutable source copies make this test independent of concurrent
    # edits to the main checkout and detect accidental editable-install imports.
    copies = {}
    for variant in ("baseline", "optimized"):
        root = tmp_path / variant
        origin = runner.ROOT if variant == "baseline" else copies["baseline"]
        shutil.copytree(origin / runner.PACKAGE, root / runner.PACKAGE, ignore=shutil.ignore_patterns("__pycache__"))
        copies[variant] = root
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19)
    settings = config()
    request = {"contract_sha256": "isolated-worker-fixture", "config": settings, "runtime": runtime(),
               "source_sha256": {key: runner.source_hashes(path) for key, path in copies.items()},
               "inputs": {"state": model.state_dict(), "channel": ContextAcquisitionModel().export_state(), "rows": rows(),
                          "model_seed": 19, "active": {"max_window_chars": 24, "context_chars": 8, "trace_limit": 256}}}
    request["input_sha256"] = runner.digest(request["inputs"])
    input_path = tmp_path / "inputs.json"
    runner.immutable_json(input_path, request)
    outputs = []
    for variant, measurement in (("baseline", "timing"), ("optimized", "memory")):
        job = {"variant": variant, "mode": "bayesian_regime", "measurement": measurement, "repeat": 0}
        path = tmp_path / (variant + ".json")
        runner.execute_worker(input_path, path, copies[variant], copies[variant], job)
        value = json.loads(path.read_text())
        runner.validate_receipt(value, job, identity(), request["contract_sha256"], request["source_sha256"][variant], runtime())
        outputs.append(value)
    for a, b in zip(outputs[0]["results"], outputs[1]["results"]):
        assert a["behavior_sha256"] == b["behavior_sha256"]
        assert a["state_sha256"] == b["state_sha256"]
        assert a["prediction_sha256"] == b["prediction_sha256"]
    assert all(row["episode_elapsed_ms"] is None and row["python_peak_bytes"] > 0 for row in outputs[1]["results"])
