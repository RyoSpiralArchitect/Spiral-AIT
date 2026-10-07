"""Boundary, marginal-calibration, and document-paired evaluation helpers."""
from __future__ import annotations

import numpy as np


def boundary_targets(text, segments):
    if any(not isinstance(part, str) or not part for part in segments) or "".join(segments) != text:
        raise ValueError("Segments must reconstruct the input exactly")
    targets = np.zeros(max(0, len(text) - 1), dtype=float)
    offset = 0
    for part in segments[:-1]:
        offset += len(part)
        targets[offset - 1] = 1
    return targets


def boundary_counts(text, gold, predicted, *, exclude_whitespace=False):
    labels = boundary_targets(text, gold).astype(bool)
    decisions = boundary_targets(text, predicted).astype(bool)
    if exclude_whitespace:
        mask = np.array([not text[i].isspace() and not text[i + 1].isspace() for i in range(len(text) - 1)], dtype=bool)
        labels, decisions = labels[mask], decisions[mask]
    tp, fp, fn = (int(np.sum(labels & decisions)), int(np.sum(~labels & decisions)), int(np.sum(labels & ~decisions)))
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / denominator if denominator else 1.0,
            "positions": len(labels)}


def temperature_scale(probabilities, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if temperature == 1:
        return values.copy()
    values = np.clip(values, 1e-12, 1 - 1e-12)
    logits = (np.log(values) - np.log1p(-values)) / temperature
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def probability_sums(probabilities, labels, *, bins=10):
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    values = np.asarray(probabilities, dtype=float)
    targets = np.asarray(labels, dtype=float)
    temperature_scale(values, 1)  # Validate independently of clipping for NLL.
    if values.ndim != 1 or values.shape != targets.shape or np.any((targets != 0) & (targets != 1)):
        raise ValueError("one binary target is required per boundary probability")
    clipped = np.clip(values, 1e-12, 1 - 1e-12)
    indices = np.minimum((values * bins).astype(int), bins - 1)
    return {"positions": len(values), "squared_error_sum": float(np.sum((values - targets) ** 2)),
            "negative_log_likelihood_sum": float(-np.sum(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped))),
            "bin_counts": np.bincount(indices, minlength=bins).tolist(),
            "bin_probability_sums": np.bincount(indices, weights=values, minlength=bins).tolist(),
            "bin_label_sums": np.bincount(indices, weights=targets, minlength=bins).tolist()}


def aggregate_probability_sums(rows):
    if not rows:
        return {"positions": 0, "brier": None, "nll": None, "ece": None}
    count = sum(row["positions"] for row in rows)
    probability_totals = np.sum([row["bin_probability_sums"] for row in rows], axis=0)
    label_totals = np.sum([row["bin_label_sums"] for row in rows], axis=0)
    return {"positions": count,
            "brier": sum(row["squared_error_sum"] for row in rows) / count if count else None,
            "nll": sum(row["negative_log_likelihood_sum"] for row in rows) / count if count else None,
            "ece": float(np.abs(probability_totals - label_totals).sum()) / count if count else None,
            "bin_counts": np.sum([row["bin_counts"] for row in rows], axis=0).tolist()}


def fit_temperature(probabilities, labels, candidates):
    if not len(labels):
        raise ValueError("Calibration requires a nonempty development partition")
    scores = [(probability_sums(temperature_scale(probabilities, value), labels)["negative_log_likelihood_sum"], value)
              for value in candidates]
    if not scores:
        raise ValueError("Provide at least one predetermined temperature candidate")
    # Prefer identity on an exact loss tie.
    _, best = min(scores, key=lambda pair: (pair[0], abs(pair[1] - 1)))
    return float(best)


def paired_document_interval(rows, *, seed=42, samples=1000):
    """Resample paired documents, conditional on the already fitted fold models."""
    by_document = {}
    for group, difference in rows:
        by_document.setdefault(group, []).append(difference)
    if not by_document:
        raise ValueError("Paired document results are required")
    keys = sorted(by_document)
    sums = np.array([sum(by_document[key]) for key in keys])
    counts = np.array([len(by_document[key]) for key in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(samples, len(keys)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"delta_mean_f1": float(sums.sum() / counts.sum()),
            "document_bootstrap_95_percent": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "document_groups": len(keys), "resamples": samples,
            "scope": "conditional document uncertainty; overlapping CV training sets and initialization uncertainty are not captured"}
