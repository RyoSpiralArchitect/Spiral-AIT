#!/usr/bin/env python3
"""Run reproducible latency, F1, and robustness evaluations for SpiralReality AIT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spiralreality_AIT_onepass_aifcore_integrated.integrated.benchmark import (
    load_benchmark_split,
    segmentation_f1,
    training_report,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import (
    GateDiagnostics,
    OnePassAIT,
    StudentTrainingConfig,
)


def _segment_lengths(segments: Sequence[str]) -> List[int]:
    return [len(tok) for tok in segments]


def _segments_from_lengths(text: str, lengths: Sequence[int]) -> List[str]:
    segments: List[str] = []
    start = 0
    for length in lengths:
        end = start + length
        segments.append(text[start:end])
        start = end
    if start < len(text):
        segments.append(text[start:])
    return segments


def _perturb_text(text: str, noise_level: float, rng: random.Random) -> str:
    """Return a perturbed version of *text* without changing its length."""

    def flip_case(ch: str) -> str:
        flipped = ch.swapcase()
        return flipped if len(flipped) == 1 else ch

    glyphs = [
        "~",
        "?",
        "…",
        "○",
        "◇",
        "▪",
    ]

    out_chars: List[str] = []
    for ch in text:
        if ch.strip() == "":
            out_chars.append(ch)
            continue
        if rng.random() > noise_level:
            out_chars.append(ch)
            continue
        if rng.random() < 0.5:
            out_chars.append(flip_case(ch))
            continue
        replacement = rng.choice(glyphs)
        if len(replacement) != 1:
            replacement = replacement[0]
        out_chars.append(replacement)
    return "".join(out_chars)


def _collect_gate_diagnostics(ait: OnePassAIT, text: str) -> Dict[str, float]:
    ait.encode(text)
    diagnostics: GateDiagnostics = ait.gate_diagnostics()
    attention = diagnostics.attention_strength if diagnostics.attention_strength else []
    attn_mean = float(sum(attention) / len(attention)) if attention else 0.0
    attn_std = (
        float(math.sqrt(sum((v - attn_mean) ** 2 for v in attention) / len(attention)))
        if attention
        else 0.0
    )
    return {
        "mask_energy": float(diagnostics.mask_energy),
        "attention_mean": attn_mean,
        "attention_std": attn_std,
    }


def run_evaluation(
    output_dir: Path,
    latency_runs: int,
    robustness_trials: int,
    robustness_noise: float,
    seed: int,
    max_samples: int | None = None,
    test_fraction: float = 0.25,
    languages: Sequence[str] | None = None,
) -> Dict[str, object]:
    if latency_runs < 1 or robustness_trials < 1:
        raise ValueError("latency_runs and robustness_trials must be positive")
    if not 0.0 <= robustness_noise <= 1.0:
        raise ValueError("robustness_noise must be between zero and one")
    split = load_benchmark_split(
        languages=languages, max_samples=max_samples, test_fraction=test_fraction, seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ait = OnePassAIT(latent_dim=32, seed=seed)
    cfg = StudentTrainingConfig(
        lr=0.05,
        epochs=16,
        batch_size=2,
        validation_split=0.25 if len(split.train_texts) > 1 else 0.0,
        patience=4,
        hidden_dim=20,
        emb_dim=14,
        window=2,
        phase_lr=0.4,
        cache_sequences=True,
        shuffle_train=True,
    )

    texts, segments = split.test_texts, split.test_segments
    summary = ait.train_student(split.train_texts, split.train_segments, cfg=cfg)
    train_info = training_report(split, summary, cfg.validation_split)

    per_sample: List[Dict[str, object]] = []
    lengths_cache = [_segment_lengths(seg) for seg in segments]
    f1_scores: List[float] = []

    for text, gold_segments, language in zip(texts, segments, split.test_languages):
        predicted_result = ait.student.decode(text)
        predicted = (
            predicted_result["tokens"]
            if isinstance(predicted_result, dict)
            else predicted_result
        )
        f1 = segmentation_f1(text, gold_segments, predicted)
        per_sample.append(
            {
                "text": text,
                "language": language,
                "f1": f1,
                "gold_segments": gold_segments,
                "predicted_segments": predicted,
            }
        )
        f1_scores.append(f1)

    latency_prompt = " ".join(texts[:2])
    latency_samples: List[float] = []
    for _ in range(latency_runs):
        ait._encode_cache.pop(latency_prompt, None)
        start = time.perf_counter()
        ait.encode(latency_prompt)
        latency_samples.append(time.perf_counter() - start)
    latency = statistics.mean(latency_samples)

    rng = random.Random(seed + 42)
    robustness_records: List[Dict[str, object]] = []
    mean_robustness_per_text: List[Tuple[str, float]] = []
    training_texts = set(split.train_texts)
    excluded_training_collisions = 0

    for text, gold_segments, lengths in zip(texts, segments, lengths_cache):
        per_text_scores: List[float] = []
        for trial in range(robustness_trials):
            noisy_text = _perturb_text(text, robustness_noise, rng)
            if noisy_text in training_texts:
                excluded_training_collisions += 1
                continue
            projected_gold = _segments_from_lengths(noisy_text, lengths)
            predicted_result = ait.student.decode(noisy_text)
            predicted = (
                predicted_result["tokens"]
                if isinstance(predicted_result, dict)
                else predicted_result
            )
            score = segmentation_f1(noisy_text, projected_gold, predicted)
            robustness_records.append(
                {
                    "text": text,
                    "trial": trial,
                    "noisy_text": noisy_text,
                    "robustness_f1": score,
                }
            )
            per_text_scores.append(score)
        if not per_text_scores:
            raise ValueError("all perturbation trials for a test text collided with training examples")
        mean_robustness_per_text.append((text, float(statistics.mean(per_text_scores))))

    overall_robustness = [rec["robustness_f1"] for rec in robustness_records]

    gate_info = _collect_gate_diagnostics(ait, texts[0])

    results: Dict[str, object] = {
        "schema_version": 2,
        "evaluation_partition": "held_out_test",
        "split": split.receipt,
        "protocol": {
            "training": "Only train_pool is passed to training; internal validation is drawn from this pool.",
            "internal_validation_fraction": cfg.validation_split,
            "segmentation_and_robustness": "Held-out test texts only; teacher labels define gold, not predictions.",
            "latency": "Encode cache cleared before every measured encode call.",
            "scope": "Small synthetic corpus; no external-corpus generalization claim.",
        },
        "seed": seed,
        "latency_prompt": latency_prompt,
        "latency_runs": latency_runs,
        "latency_seconds": latency,
        "latency_samples_seconds": latency_samples,
        "segmentation": {
            "per_sample": per_sample,
            "mean": float(statistics.mean(f1_scores)) if f1_scores else 0.0,
            "stdev": float(statistics.pstdev(f1_scores)) if len(f1_scores) > 1 else 0.0,
        },
        "robustness": {
            "excluded_training_collisions": excluded_training_collisions,
            "noise_level": robustness_noise,
            "trials": robustness_trials,
            "records": robustness_records,
            "per_text_mean": [
                {"text": text, "mean_f1": score} for text, score in mean_robustness_per_text
            ],
            "mean": float(statistics.mean(overall_robustness)) if overall_robustness else 0.0,
            "stdev": float(statistics.pstdev(overall_robustness))
            if len(overall_robustness) > 1
            else 0.0,
        },
        "gate_diagnostics": gate_info,
        "train_summary": train_info["summary"],
        "training": train_info,
    }

    json_path = output_dir / "evaluation_metrics.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with (output_dir / "segmentation_f1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "f1"])
        writer.writerows((entry["text"], f'{entry["f1"]:.6f}') for entry in per_sample)

    with (output_dir / "robustness.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "trial", "robustness_f1"])
        writer.writerows((entry["text"], entry["trial"], f'{entry["robustness_f1"]:.6f}') for entry in robustness_records)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation"),
        help="Directory where evaluation artefacts will be stored.",
    )
    parser.add_argument("--latency-runs", type=int, default=8, help="Number of encode passes for latency.")
    parser.add_argument(
        "--robustness-trials",
        type=int,
        default=5,
        help="Number of perturbation trials per sample when estimating robustness.",
    )
    parser.add_argument(
        "--robustness-noise",
        type=float,
        default=0.12,
        help="Probability of perturbing a character when measuring robustness.",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Random seed used for evaluation runs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap unique samples across languages before splitting.")
    parser.add_argument("--test-fraction", type=float, default=0.25, help="Held-out fraction within each selected language.")
    parser.add_argument("--languages", nargs="+", default=None, help="Curated multilingual language codes to include alongside the reflective corpus.")
    args = parser.parse_args()

    results = run_evaluation(
        output_dir=args.output,
        latency_runs=args.latency_runs,
        robustness_trials=args.robustness_trials,
        robustness_noise=args.robustness_noise,
        seed=args.seed,
        max_samples=args.max_samples,
        test_fraction=args.test_fraction,
        languages=args.languages,
    )
    print(json.dumps({k: v for k, v in results.items() if k not in {"train_summary", "segmentation", "robustness"}}, indent=2))


if __name__ == "__main__":
    main()
