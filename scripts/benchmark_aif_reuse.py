"""Compare shared legacy-EFE features with the former five-pass orchestration.

Both paths use the same current weights and CRF. This isolates inference reuse;
it does not measure the new posterior-risk selector, whole-revision performance,
or segmentation quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import TRAIN_TEXTS
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import (
    OnePassAIT, SegmentationAIFConfig, StudentTrainingConfig,
)


def run(repeats: int = 9) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    files = ["boundary.py", "onepass_ait.py", "encoder.py", "encoder_backends.py", "phase.py"]
    package = ROOT / "spiralreality_AIT_onepass_aifcore_integrated" / "integrated"
    def fingerprints():
        return {**{name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in files},
                "benchmark_aif_reuse.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    before = fingerprints()
    cfg = SegmentationAIFConfig(selection_mode="legacy_efe")
    model = OnePassAIT(latent_dim=32, seed=5042, encoder_backend="numpy")
    if model.encoder_backend_name() != "spectral-numpy:cpu":
        raise RuntimeError("This matched CPU benchmark requires the NumPy encoder")
    model.train_student(TRAIN_TEXTS[:3], cfg=StudentTrainingConfig(
        epochs=3, validation_split=0.0, phase_lr=0.3,
    ))
    reference = OnePassAIT(latent_dim=32, seed=5042, encoder_backend="numpy")
    reference.load_state_dict(model.state_dict())
    original_probs = reference.student.boundary_probs_with_logit_bias

    def recompute_probs(text, logit_bias=0.0, *, logits=None):
        return original_probs(text, logit_bias)

    reference.student.boundary_probs_with_logit_bias = recompute_probs

    def recomputed(text):
        # Four independent forward passes for policy marginals, then a fifth
        # for the selected Viterbi path, matching the old orchestration.
        selection = reference._select_policy_from_logits(text, (), cfg)
        bias = cfg.logit_bias_by_policy[selection["chosen_policy"]]
        result = reference.student.decode_with_logit_bias(text, bias)
        return {**result, "chosen_policy": selection["chosen_policy"], "aif": selection}

    def shared(text):
        return model.segment_text(text, use_aif=True, aif_cfg=cfg, return_metadata=True)

    counts = {"recomputed": 0, "shared": 0}
    for name, instance in (("recomputed", reference), ("shared", model)):
        forward = instance.encoder.forward
        def counted(*args, _name=name, _forward=forward, **kwargs):
            counts[_name] += 1
            return _forward(*args, **kwargs)
        instance.encoder.forward = counted

    texts = [
        "Streaming inference should retain complete words while comparing policies.",
        "新しい文章でも、分割と推論の結果を確認する。",
        "Evidence arrives gradually; revise the provisional interpretation carefully.",
    ]
    rows = []
    for text in texts:
        counts.update(recomputed=0, shared=0)
        expected, actual = recomputed(text), shared(text)
        if expected != actual:
            raise RuntimeError("AIF output changed")
        passes = dict(counts)
        if passes != {"recomputed": 5, "shared": 1}:
            raise RuntimeError(f"Unexpected encoder pass counts: {passes}")
        timings = {"recomputed": [], "shared": []}
        funcs = {"recomputed": recomputed, "shared": shared}
        for index in range(repeats):
            order = ("recomputed", "shared") if index % 2 == 0 else ("shared", "recomputed")
            for name in order:
                start = time.perf_counter()
                result = funcs[name](text)
                timings[name].append((time.perf_counter() - start) * 1000)
                if "".join(result["tokens"]) != text:
                    raise RuntimeError("Segmentation failed to reconstruct input")
        medians = {name: statistics.median(values) for name, values in timings.items()}
        rows.append({"text": text, "chars": len(text), "encoder_passes": passes,
                     "latency_ms": timings, "median_ms": medians,
                     "median_ratio": medians["recomputed"] / medians["shared"],
                     "identical_tokens_policy_and_scores": True})
    if before != fingerprints():
        raise RuntimeError("Source files changed during measurement")
    return {
        "method": "matched_current_weights_recomputed_vs_shared_AIF_features",
        "selection_mode": cfg.selection_mode,
        "scope": "Local CPU inference only; no quality or whole-revision speed claim.",
        "seed": 5042, "repeats": repeats, "warmup_per_path_per_text": 1,
        "order": "alternating", "python": platform.python_version(),
        "numpy": np.__version__, "machine": platform.machine(),
        "encoder_backend": model.encoder_backend_name(),
        "source_sha256": before,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, default=Path("reports/aif_reuse.json"))
    args = parser.parse_args()
    report = run(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: row[k] for k in ("chars", "encoder_passes", "median_ms", "median_ratio")}
                      for row in report["rows"]], indent=2))
