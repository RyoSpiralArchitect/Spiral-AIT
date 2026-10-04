"""Fixed, paired held-out ablation of the optional lexical CRF residual.

The feature design and three seeds are fixed before test scoring. This is a
small synthetic-corpus experiment, not a parameter search or production claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from spiralreality_AIT_onepass_aifcore_integrated.integrated.benchmark import (
    load_benchmark_split, segmentation_f1, training_report,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import (
    OnePassAIT, StudentTrainingConfig,
)


def run(seeds=(5042, 101, 2026)):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least one distinct seed")
    package = ROOT / "spiralreality_AIT_onepass_aifcore_integrated" / "integrated"
    files = ["boundary.py", "onepass_ait.py", "encoder.py", "encoder_backends.py", "phase.py", "benchmark.py"]
    def fingerprints():
        return {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in files}
    before = fingerprints()
    results = []
    for seed in seeds:
        split = load_benchmark_split(seed=seed, max_samples=None)
        conditions = {}
        for buckets in (0, 4096):
            cfg = StudentTrainingConfig(
                lr=0.05, epochs=8, batch_size=2, validation_split=0.25,
                patience=3, hidden_dim=20, emb_dim=14, window=2,
                phase_lr=0.3, cache_sequences=False, shuffle_train=False,
                lexical_buckets=buckets, lexical_lr=0.05,
            )
            model = OnePassAIT(latent_dim=32, seed=seed)
            summary = model.train_student(split.train_texts, split.train_segments, cfg=cfg)
            rows = []
            for text, gold, language in zip(split.test_texts, split.test_segments, split.test_languages):
                predicted = model.segment_text(text)
                rows.append({
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "language": language, "text": text, "gold": list(gold),
                    "predicted": predicted, "f1": segmentation_f1(text, gold, predicted),
                })
            conditions[str(buckets)] = {
                "config": asdict(cfg), "macro_f1": statistics.mean(row["f1"] for row in rows),
                "per_language": {
                    lang: {"samples": sum(row["language"] == lang for row in rows),
                           "f1": statistics.mean(row["f1"] for row in rows if row["language"] == lang)}
                    for lang in sorted(set(split.test_languages))
                },
                "training": training_report(split, summary, cfg.validation_split), "rows": rows,
            }
        for partition in ("train", "validation"):
            if conditions["0"]["training"][partition] != conditions["4096"]["training"][partition]:
                raise RuntimeError(f"Unmatched {partition} partitions")
        results.append({"seed": seed, "split": split.receipt, "conditions": conditions,
                        "lexical_macro_f1_delta": conditions["4096"]["macro_f1"] - conditions["0"]["macro_f1"]})
    if before != fingerprints():
        raise RuntimeError("Source files changed during measurement")
    return {
        "method": "paired_lexical_residual_0_vs_4096_buckets",
        "scope": "33 synthetic examples; test folds across seeds overlap; no independent external test set.",
        "averaging": "macro_f1 is the mean of per-text boundary F1; mean_macro_f1 averages those means across seeds, not languages.",
        "protocol": "Fixed 7 templates, 4096 buckets, lexical_lr=.05; no held-out hyperparameter tuning. Early stopping uses internal validation only.",
        "python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
        "source_sha256": before, "seeds": list(seeds), "results": results,
        "mean_macro_f1": {str(b): statistics.mean(row["conditions"][str(b)]["macro_f1"] for row in results)
                          for b in (0, 4096)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[5042, 101, 2026])
    parser.add_argument("--output", type=Path, default=Path("reports/lexical_ablation.json"))
    args = parser.parse_args()
    report = run(args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mean_macro_f1": report["mean_macro_f1"], "paired_seeds": [
        {"seed": row["seed"], "delta": row["lexical_macro_f1_delta"],
         "f1": {key: condition["macro_f1"] for key, condition in row["conditions"].items()}}
        for row in report["results"]]}, indent=2))
