#!/usr/bin/env python3
"""Isolated training worker for a pinned implementation and local input file."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def run(task_path):
    task = json.loads(Path(task_path).read_text())
    for relative, expected in task["implementation_files"].items():
        if hashlib.sha256((Path(task["implementation_root"]) / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Implementation differs from the frozen task")
    # Import exactly one implementation per process, including for the historical
    # optimizer. The parent verifies its immutable source snapshot before launch.
    sys.path.insert(0, task["implementation_root"])
    import numpy as np
    from spiralreality_AIT_onepass_aifcore_integrated.integrated.checkpoint import save_checkpoint
    from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig

    payload = Path(task["training_input"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != task["training_input_sha256"]:
        raise ValueError("Training input differs from the prepared task")
    examples = json.loads(payload)
    model = OnePassAIT(seed=task["seed"], **task["model"])
    initial = {key: value.copy() for key, value in model.encoder.trainable_parameters().items()}
    cfg = StudentTrainingConfig(**task["training"])
    summary = model.train_student([row["text"] for row in examples], [row["segments"] for row in examples], cfg)
    change = sum(float(np.sum((parameter - initial[key]) ** 2))
                 for key, parameter in model.encoder.trainable_parameters().items()) ** 0.5
    if (cfg.train_context_encoder and change <= 0) or (not cfg.train_context_encoder and change != 0):
        raise ValueError("Encoder intervention does not match the task")
    checkpoint = Path(task["checkpoint"])
    save_checkpoint(checkpoint, model.state_dict())
    probes = []
    for text in ("Hello world.", "日本語です。", "中文测试。"):
        result = model.segment_text(text, include_confidence=True)
        probes.append([result["tokens"], result["boundary_probabilities"]])
    receipt = {"contract_sha256": task["contract_sha256"], "run_id": task["run_id"],
               "training_input_sha256": task["training_input_sha256"], "seed": task["seed"],
               "training_config": task["training"], "model_config": task["model"],
               "implementation": task["implementation"], "encoder_parameter_change_l2": change,
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "inference_probe_sha256": hashlib.sha256(json.dumps(probes, sort_keys=True, ensure_ascii=False,
                                                                      separators=(",", ":")).encode()).hexdigest(),
               "train_sequences": summary["train_sequences"], "epochs_completed": summary["epochs_completed"],
               "train_seconds": summary["train_seconds"], "history": summary["history"]}
    save_checkpoint(task["receipt"], receipt)
    print(json.dumps({"run_id": task["run_id"], "stage": "trained", "encoder_change_l2": change}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    run(parser.parse_args().task)
