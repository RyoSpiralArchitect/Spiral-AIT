#!/usr/bin/env python3
"""Synthetic resource probe for observation bookkeeping, not task efficacy."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc

PACKAGE = "spiralreality_AIT_onepass_aifcore_integrated"
WEIGHTS = ("W1", "b1", "W_mu", "b_mu", "W_sigma", "b_sigma")
PROTOCOL = {"schema_version": 1, "kind": "synthetic_observation_transaction_resource_probe",
            "seed": 5042, "latent_dim": 16, "replay_rows": [150, 2048],
            "arrivals_per_pass": 64, "untraced_repeats": 5, "traced_passes": 1,
            "train": False, "policy": "SeekEvidence",
            "scope": "Deterministic provided numeric fixtures; no text, gold labels, encoder forward, or fitting. Setup and hashes are outside timing. Five fresh-fixture untraced passes measure only delivered observation calls. A separate fresh-fixture pass measures peak Python-traced allocations; traced elapsed time is never used for speed. This is bookkeeping evidence, not end-to-end speed, process RSS, real-task accuracy, or learning efficacy."}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def json_digest(value):
    tree = json.loads(json.dumps(value, allow_nan=False))
    return digest(json.dumps(tree, sort_keys=True, separators=(",", ":")).encode())


def load_package(root):
    """Select a source tree before importing any package implementation."""
    root = Path(root).resolve()
    package = root / PACKAGE
    if not (package / "__init__.py").is_file():
        raise ValueError("package-root must contain the requested package source")
    if any(name == PACKAGE or name.startswith(PACKAGE + ".") for name in sys.modules):
        raise ValueError("Run the probe in a fresh process before importing AIT")
    sys.path.insert(0, str(root))
    implementation = importlib.import_module(PACKAGE + ".integrated.onepass_ait")
    if Path(implementation.__file__).resolve() != (package / "integrated/onepass_ait.py").resolve():
        raise ValueError("Imported implementation does not match selected package-root")
    import numpy as np
    return root, implementation.OnePassAIT, np


def fingerprints(root):
    return {str(path.relative_to(root)): digest(path.read_bytes())
            for path in sorted((root / PACKAGE).rglob("*.py"))}


def weight_signatures(model):
    return {name: json_digest(getattr(model.dynamics, name).tolist()) for name in WEIGHTS}


def run(package_root, *, label):
    root, model_type, np = load_package(package_root)
    code = fingerprints(root)
    seed = PROTOCOL["seed"]
    count = PROTOCOL["arrivals_per_pass"]
    base = np.linspace(-0.2, 0.2, PROTOCOL["latent_dim"])
    context = np.full(PROTOCOL["latent_dim"], 0.1)
    arrivals = [(base + index * 0.001, context, base + index * 0.001 + 0.02)
                for index in range(count)]

    def fixture(rows):
        model = model_type(latent_dim=16, seed=seed, encoder_layers=1, encoder_heads=2,
                           encoder_backend="numpy")
        policy = model.policy_vecs[PROTOCOL["policy"]]
        policy = policy / (np.linalg.norm(policy) + 1e-8)
        # Explicit fixture construction is excluded from arrival timing. Replay
        # is prepopulated without fitting, and phase histories start empty.
        for index in range(rows):
            before, ctx, after = arrivals[index % count]
            model.dynamics.record(before, policy, ctx, after)
        model.last_observation_id = rows
        return model

    def deliver(model):
        for before, ctx, after in arrivals:
            receipt = model.observe_transition(before, PROTOCOL["policy"], ctx, after,
                        observation_id=model.last_observation_id + 1,
                        source="provided_synthetic_numeric_resource_probe", train=False)
            if not receipt["recorded"] or receipt["training"]["trained"]:
                raise ValueError("Synthetic resource probe changed its no-fitting contract")

    rows = []
    if tracemalloc.is_tracing():
        raise ValueError("Untraced timing requires tracemalloc to be disabled")
    for size in PROTOCOL["replay_rows"]:
        samples, signatures, final_hashes = [], [], []
        initial_hash = initial_weights = None
        for _ in range(PROTOCOL["untraced_repeats"]):
            model = fixture(size)
            before_weights = weight_signatures(model)
            before_state = json_digest(model.state_dict())
            if initial_hash is not None and (before_state != initial_hash or before_weights != initial_weights):
                raise ValueError("Fresh fixture state is not deterministic")
            initial_hash, initial_weights = before_state, before_weights
            started = time.perf_counter_ns()
            deliver(model)
            samples.append((time.perf_counter_ns() - started) / count / 1e6)
            signatures.append(weight_signatures(model))
            final_hashes.append(json_digest(model.state_dict()))
        model = fixture(size)
        tracemalloc.start()
        try:
            deliver(model)
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        signatures.append(weight_signatures(model))
        final_hashes.append(json_digest(model.state_dict()))
        if any(value != initial_weights for value in signatures) or len(set(final_hashes)) != 1:
            raise ValueError("Weight freeze or repeat/final-state equivalence failed")
        rows.append({"replay_rows": size, "untraced_ms_per_arrival": samples,
                     "median_untraced_ms_per_arrival": statistics.median(samples),
                     "traced_current_bytes": current, "traced_peak_bytes": peak,
                     "initial_state_sha256": initial_hash, "final_state_sha256": final_hashes[0],
                     "frozen_weight_sha256": initial_weights,
                     "recorded_observations_after": model.dynamics.recorded_observations,
                     "retained_replay_rows_after": len(model.dynamics.buffer)})
    if code != fingerprints(root):
        raise ValueError("Selected implementation changed during resource probe")
    return {"protocol": PROTOCOL, "label": label, "package_import_verified": True,
            "source_sha256": code, "probe_sha256": digest(Path(__file__).read_bytes()),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "thread_environment": {
                            name: os.environ.get(name) for name in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
            "results": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", default="unspecified")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite a resource-probe receipt")
        receipt = run(args.package_root, label=args.label)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"complete": True, "kind": PROTOCOL["kind"], "label": args.label,
                          "results": [{key: row[key] for key in ("replay_rows", "median_untraced_ms_per_arrival", "traced_peak_bytes")} for row in receipt["results"]]}))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
