"""Offline interpolation of compatible Python segmentation checkpoints.

The returned checkpoint has one ordinary head and encoder: inference does not
keep two models or route examples using language labels. This is an opt-in
weight-space intervention, not an output-preservation or retention guarantee.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numbers

import numpy as np

from .onepass_ait import OnePassAIT


_HEAD_ARRAYS = ("embeddings", "W_window", "b_window", "W_out", "gate_w", "transitions",
                "lexical_weights", "ctx_W1", "ctx_b1", "ctx_w")
_HEAD_SCALARS = ("b_out", "gate_b", "ctx_b")


def _load_python(state):
    if state.get("student", {}).get("fitted_backend") != "python":
        raise ValueError("Checkpoint blending requires explicit Python weight ownership")
    encoder = state["encoder"]
    if state["student"].get("_encoder") != encoder:
        raise ValueError("Checkpoint contains inconsistent encoder copies")
    model = OnePassAIT(latent_dim=state["latent_dim"], encoder_layers=encoder["n_layers"],
                       encoder_heads=encoder["n_heads"], encoder_backend="numpy")
    model.load_state_dict(deepcopy(state))
    return model


def blend_checkpoints(reference, adapted, *, adaptation_weight):
    """Return ``(1-weight)*reference + weight*adapted`` segmentation weights.

    Both states must have the same architecture, precision, lexical layout and
    phase. Every head/encoder tensor is checked before interpolation; inputs
    remain unchanged. Other state (including AIF dynamics) is copied from the
    adapted checkpoint, not averaged. Endpoints return exact deep copies after
    compatibility validation. This does not restore training RNG state.
    """
    if (isinstance(adaptation_weight, bool) or not isinstance(adaptation_weight, numbers.Real)
            or not math.isfinite(adaptation_weight) or not 0 <= adaptation_weight <= 1):
        raise ValueError("adaptation_weight must be finite and within [0, 1]")
    first, second = _load_python(reference), _load_python(adapted)
    for name in ("architecture", "use_encoder_context", "context_hidden_dim"):
        if reference["student"].get(name) != adapted["student"].get(name):
            raise ValueError("Checkpoint blending requires matching " + name)
    for name in ("version", "buckets"):
        if reference["student"]["lexical"][name] != adapted["student"]["lexical"][name]:
            raise ValueError("Checkpoint blending requires matching lexical layout")
    if (reference["phase"] != adapted["phase"]
            or reference["student"]["_phase"] != adapted["student"]["_phase"]):
        raise ValueError("Checkpoint blending requires unchanged phase state")
    if any(reference["encoder"][key] != adapted["encoder"][key] for key in ("d_model", "n_layers", "n_heads")):
        raise ValueError("Checkpoint blending requires matching encoder architecture")

    left = {"head." + key: getattr(first.student, key) for key in _HEAD_ARRAYS}
    right = {"head." + key: getattr(second.student, key) for key in _HEAD_ARRAYS}
    left.update({"encoder." + key: value for key, value in first.encoder.trainable_parameters().items()})
    right.update({"encoder." + key: value for key, value in second.encoder.trainable_parameters().items()})
    if left.keys() != right.keys():
        raise ValueError("Checkpoint encoder parameter names differ")
    for name, value in left.items():
        other = right[name]
        if value is None and other is None:
            continue
        if (not isinstance(value, np.ndarray) or not isinstance(other, np.ndarray)
                or value.shape != other.shape or value.dtype != other.dtype
                or not np.all(np.isfinite(value)) or not np.all(np.isfinite(other))):
            raise ValueError("Incompatible or nonfinite checkpoint tensor: " + name)
    for name in _HEAD_SCALARS:
        if any(not math.isfinite(getattr(model.student, name)) for model in (first, second)):
            raise ValueError("Nonfinite checkpoint scalar: " + name)
    if adaptation_weight in (0, 1):
        return deepcopy(reference if adaptation_weight == 0 else adapted)
    weight = float(adaptation_weight)
    for name, value in left.items():
        if value is not None:
            mixed = (1 - weight) * value + weight * right[name]
            if not np.all(np.isfinite(mixed)):
                raise ValueError("Nonfinite blended tensor: " + name)
            right[name][...] = mixed
    for name in _HEAD_SCALARS:
        setattr(second.student, name, (1 - weight) * getattr(first.student, name)
                + weight * getattr(second.student, name))
    return second.state_dict()


__all__ = ["blend_checkpoints"]
