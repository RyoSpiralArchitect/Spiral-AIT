from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import operator
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from .boundary_cpp import CompiledStudentHandle, compiled_backend_devices, load_compiled_student
from .boundary_julia import JuliaStudentHandle, julia_backend_devices, load_julia_student
from .np_compat import np
from .phase import PhaseBasisLearner, PhaseBasisState
from .utils import is_cjk, is_kana, is_latin, is_punct, is_space, seeded_vector, sigmoid

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .encoder import SpectralTransformerAdapter


logger = logging.getLogger(__name__)


_CHAR_CLASSES = ["space", "latin", "cjk", "kana", "punct", "digit", "other"]
_LEXICAL_VERSION = "signed-char-ngram-v1"
_LEXICAL_TEMPLATES = ((-1,), (0,), (1,), (2,), (-1, 0), (0, 1), (1, 2))


logger = logging.getLogger(__name__)


def _log_backend_exception(stage: str, backend: str, exc: Exception) -> None:
    payload = {
        "event": "boundary_backend_failure",
        "stage": stage,
        "backend": backend,
        "error": exc.__class__.__name__,
        "message": str(exc),
    }
    logger.exception(json.dumps(payload, ensure_ascii=False))


def _log_backend_selection(stage: str, backend: str, fallbacks: Sequence[str]) -> None:
    payload = {
        "event": "boundary_backend_selection",
        "stage": stage,
        "backend": backend,
        "fallback_chain": list(fallbacks),
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


def _char_category(ch: str) -> int:
    if is_space(ch):
        return 0
    if is_latin(ch):
        return 1
    if is_cjk(ch):
        return 2
    if is_kana(ch):
        return 3
    if is_punct(ch):
        return 4
    if ch.isdigit():
        return 5
    return 6


@dataclass
class BoundarySequence:
    text: str
    categories: np.ndarray
    labels: np.ndarray
    curvature: np.ndarray
    phases: np.ndarray
    lexical_features: Optional[Tuple[Tuple[Tuple[int, int], ...], ...]] = None
    lexical_buckets: int = 0


@dataclass
class StudentTrainingConfig:
    lr: float = 0.05
    epochs: int = 80
    batch_size: int = 2
    reg: float = 1e-4
    crf_lr: float = 0.02
    phase_lr: float = 1.0
    encoder_lr: float = 2e-3
    hidden_dim: int = 24
    emb_dim: int = 16
    window: int = 2
    validation_split: float = 0.15
    patience: int = 6
    dtype: str = "float32"
    max_grad_norm: Optional[float] = 10.0
    cache_sequences: bool = True
    shuffle_train: bool = True
    device_preference: str = "auto"
    use_encoder_context: bool = True
    context_lr: float = 0.02
    context_hidden_dim: int = 32
    lexical_buckets: int = 0
    lexical_lr: float = 0.05


class BoundaryStudent:
    """Trainable boundary detector with a shallow NN + CRF head."""

    def __init__(self, phase: PhaseBasisLearner, seed: int = 0):
        self.phase = phase
        self.rng = random.Random(seed)
        self.hidden_dim = 24
        self.emb_dim = 16
        self.window = 2
        self.window_dim = self.emb_dim * (self.window * 2)
        self.dtype = np.float32 if hasattr(np, "float32") else float
        self.max_grad_norm: Optional[float] = 10.0
        self.lexical_buckets = 0
        self._fitted_backend: Optional[str] = None
        self._init_parameters()
        self.encoder_adapter: Optional["SpectralTransformerAdapter"] = None
        self.use_encoder_context: bool = True
        self.context_hidden_dim: int = 32
        self.ctx_W1: Optional[np.ndarray] = None
        self.ctx_b1: Optional[np.ndarray] = None
        self.ctx_w: Optional[np.ndarray] = None
        self.ctx_b: float = 0.0
        self.history: List[Dict[str, float]] = []
        self.best_state: Optional[Dict[str, object]] = None
        self._last_backend_used: Optional[str] = None
        self._last_backend_fallbacks: List[str] = []
        self.julia_backend: Optional[JuliaStudentHandle] = load_julia_student()
        if self.julia_backend is not None:
            try:
                self.julia_backend.attach_phase(self.phase)
            except Exception as exc:
                logger.warning(
                    "Disabling Julia boundary backend after attach_phase failure",
                    exc_info=True,
                )
                self.julia_backend = None
        self.compiled_backend: Optional[CompiledStudentHandle] = load_compiled_student()
        if self.compiled_backend is not None:
            try:
                self.compiled_backend.attach_phase(self.phase)
            except Exception as exc:
                logger.warning(
                    "Disabling compiled boundary backend after attach_phase failure",
                    exc_info=True,
                )
                self.compiled_backend = None

    def configure(self, cfg: StudentTrainingConfig) -> None:
        lexical_buckets = self._validate_lexical_config(cfg)
        self.history = []
        self.best_state = None
        self.hidden_dim = cfg.hidden_dim
        self.emb_dim = cfg.emb_dim
        self.window = cfg.window
        self.window_dim = self.emb_dim * (self.window * 2)
        if cfg.dtype == "float64" and hasattr(np, "float64"):
            self.dtype = np.float64  # type: ignore[assignment]
        else:
            self.dtype = np.float32 if hasattr(np, "float32") else float
        self.max_grad_norm = cfg.max_grad_norm
        self.lexical_buckets = lexical_buckets
        self._init_parameters()
        self._fitted_backend = "python"
        self.use_encoder_context = bool(getattr(cfg, "use_encoder_context", True))
        self.context_hidden_dim = int(getattr(cfg, "context_hidden_dim", self.context_hidden_dim))
        if self.context_hidden_dim <= 0:
            self.context_hidden_dim = 1
        self.ctx_W1 = self.ctx_b1 = self.ctx_w = None
        self.ctx_b = 0.0
        self._ensure_context_parameters()
        # Native heads do not yet implement the lexical emission residual.
        if self.lexical_buckets:
            return
        self._select_backend_device(cfg.device_preference)
        if self.julia_backend is not None:
            try:
                self.julia_backend.configure(cfg.__dict__)
            except Exception as exc:
                logger.warning(
                    "Julia boundary backend configure failed; falling back",
                    exc_info=True,
                )
                self.julia_backend = None
        if self.compiled_backend is not None:
            try:
                self.compiled_backend.configure(cfg.__dict__)
            except Exception as exc:
                logger.warning(
                    "Compiled boundary backend configure failed; falling back",
                    exc_info=True,
                )
                self.compiled_backend = None

    @staticmethod
    def _validate_lexical_config(cfg: StudentTrainingConfig) -> int:
        try:
            buckets = operator.index(cfg.lexical_buckets)
        except TypeError as exc:
            raise ValueError("lexical_buckets must be a nonnegative integer") from exc
        if buckets < 0:
            raise ValueError("lexical_buckets must be a nonnegative integer")
        if not math.isfinite(cfg.lexical_lr) or cfg.lexical_lr <= 0:
            raise ValueError("lexical_lr must be finite and positive")
        return buckets

    def _require_python_head(self) -> None:
        if self._fitted_backend in ("compiled", "julia"):
            raise ValueError(
                f"The fitted {self._fitted_backend} boundary head has no synchronized "
                "Python weights; confidence, AIF and Python-logit evaluation require "
                "training or loading a Python boundary head"
            )

    def _require_fitted_backend(self) -> None:
        if self._fitted_backend in ("compiled", "julia"):
            if getattr(self, f"{self._fitted_backend}_backend") is None:
                raise RuntimeError(f"The fitted {self._fitted_backend} boundary backend is unavailable")

    def bind_encoder(self, encoder: "SpectralTransformerAdapter") -> None:
        self.encoder_adapter = encoder
        self._ensure_context_parameters()
        if self.julia_backend is not None:
            try:
                self.julia_backend.attach_encoder(encoder)
            except Exception as exc:
                logger.warning(
                    "Julia boundary backend encoder bind failed; disabling backend",
                    exc_info=True,
                )
                self.julia_backend = None
        if self.compiled_backend is not None:
            try:
                self.compiled_backend.attach_encoder(encoder)
            except Exception as exc:
                logger.warning(
                    "Compiled boundary backend encoder bind failed; disabling backend",
                    exc_info=True,
                )
                self.compiled_backend = None

    def _backend_metadata(
        self, backend: str, fallbacks: Optional[List[str]] = None, stage: str = "selection"
    ) -> Dict[str, object]:
        meta_fallbacks = list(fallbacks) if fallbacks else []
        self._last_backend_used = backend
        self._last_backend_fallbacks = meta_fallbacks
        _log_backend_selection(stage, backend, meta_fallbacks)
        return {"backend_used": backend, "backend_fallbacks": meta_fallbacks}

    def _init_parameters(self) -> None:
        def rand_vec(size: int, scale: float = 0.1):
            values = [self.rng.uniform(-scale, scale) for _ in range(size)]
            return np.array(values, dtype=self.dtype)

        num_classes = len(_CHAR_CLASSES)
        self.embeddings = np.stack([rand_vec(self.emb_dim, 0.2) for _ in range(num_classes)], axis=0)
        self.W_window = np.stack([rand_vec(self.window_dim, 0.1) for _ in range(self.hidden_dim)], axis=0)
        self.b_window = np.zeros(self.hidden_dim, dtype=self.dtype)
        self.W_out = rand_vec(self.hidden_dim, 0.1)
        self.b_out = 0.0
        self.gate_w = rand_vec(3, 0.05)
        self.gate_b = 0.0
        self.transitions = np.zeros((2, 2), dtype=self.dtype)
        self.lexical_weights = np.zeros(self.lexical_buckets, dtype=self.dtype)

    def _context_ready(self) -> bool:
        if not self.use_encoder_context:
            return False
        if self.encoder_adapter is None:
            return False
        if self.ctx_W1 is None or self.ctx_b1 is None or self.ctx_w is None:
            return False
        dim = self._encoder_model_dim()
        if dim is None:
            return False
        hidden = int(self.ctx_w.shape[0])
        if self.ctx_W1.shape != (hidden, int(dim)):
            return False
        if int(getattr(self.ctx_b1, "shape", (0,))[0]) != hidden:
            return False
        return True

    def _encoder_model_dim(self) -> Optional[int]:
        if self.encoder_adapter is None:
            return None
        d_model = getattr(self.encoder_adapter, "d_model", None)
        if d_model is not None:
            try:
                return int(d_model)
            except Exception:
                return None
        if hasattr(self.encoder_adapter, "export_state"):
            try:
                state = self.encoder_adapter.export_state()
            except Exception:
                state = None
            if isinstance(state, dict) and "d_model" in state:
                try:
                    return int(state["d_model"])
                except Exception:
                    return None
        if self.ctx_W1 is not None:
            try:
                return int(self.ctx_W1.shape[1])
            except Exception:
                return None
        return None

    def _ensure_context_parameters(self) -> None:
        if not self.use_encoder_context:
            self.ctx_W1 = None
            self.ctx_b1 = None
            self.ctx_w = None
            self.ctx_b = 0.0
            return
        if self.encoder_adapter is None:
            return
        dim = self._encoder_model_dim()
        if dim is None:
            return
        dim = int(dim)
        if dim <= 0:
            return
        hidden = max(1, int(self.context_hidden_dim))
        if self.ctx_W1 is None or self.ctx_W1.shape != (hidden, dim):
            base = seeded_vector(
                f"BoundaryStudent.ctx_W1::{hidden}::{dim}",
                dim=hidden * dim,
                dtype=self.dtype,
            )
            self.ctx_W1 = np.array(base, dtype=self.dtype, copy=True).reshape(hidden, dim) * 0.05
            self.ctx_b1 = np.zeros(hidden, dtype=self.dtype)
            base_out = seeded_vector(
                f"BoundaryStudent.ctx_w::{hidden}::{dim}",
                dim=hidden,
                dtype=self.dtype,
            )
            self.ctx_w = np.array(base_out, dtype=self.dtype, copy=True) * 0.05
            self.ctx_b = 0.0
        else:
            if self.ctx_b1 is None or int(getattr(self.ctx_b1, "shape", (0,))[0]) != hidden:
                self.ctx_b1 = np.zeros(hidden, dtype=self.dtype)
            if self.ctx_w is None or int(self.ctx_w.shape[0]) != hidden:
                base_out = seeded_vector(
                    f"BoundaryStudent.ctx_w::{hidden}::{dim}",
                    dim=hidden,
                    dtype=self.dtype,
                )
                self.ctx_w = np.array(base_out, dtype=self.dtype, copy=True) * 0.05
                self.ctx_b = 0.0

        if self.ctx_W1 is not None and hasattr(self.ctx_W1, "dtype") and self.ctx_W1.dtype != self.dtype:
            self.ctx_W1 = np.array(self.ctx_W1, dtype=self.dtype, copy=True)
        if self.ctx_b1 is not None and hasattr(self.ctx_b1, "dtype") and self.ctx_b1.dtype != self.dtype:
            self.ctx_b1 = np.array(self.ctx_b1, dtype=self.dtype, copy=True)
        if self.ctx_w is not None and hasattr(self.ctx_w, "dtype") and self.ctx_w.dtype != self.dtype:
            self.ctx_w = np.array(self.ctx_w, dtype=self.dtype, copy=True)

    def _encode_context(self, seq: BoundarySequence) -> Optional[np.ndarray]:
        if not self._context_ready():
            return None
        assert self.encoder_adapter is not None
        d_model = self._encoder_model_dim()
        if d_model is None:
            return None
        d_model = int(d_model)
        if d_model <= 0:
            return None
        text = seq.text
        if not text:
            return None
        X = np.stack(
            [seeded_vector(f"char::{ch}", dim=d_model, dtype=self.dtype) for ch in text],
            axis=0,
        ).astype(float, copy=False)
        gate_pos = np.array([sigmoid(float(v)) for v in seq.curvature], dtype=float)
        if gate_pos.shape[0] != X.shape[0]:
            gate_pos = np.resize(gate_pos, (X.shape[0],))
        gate_mask = np.minimum.outer(gate_pos, gate_pos).astype(float, copy=False)
        H = self.encoder_adapter.forward(X, gate_pos, gate_mask=gate_mask)
        return np.asarray(H, dtype=float)

    # ------------------------------------------------------------------
    # Dataset construction helpers
    # ------------------------------------------------------------------
    def build_sequences(self, texts: Sequence[str], segments: Sequence[Sequence[str]]) -> List[BoundarySequence]:
        self._validate_training_data(texts, segments)
        return [self._build_sequence(text, seg) for text, seg in zip(texts, segments)]

    @staticmethod
    def _validate_training_data(texts: Sequence[str], segments: Sequence[Sequence[str]]) -> None:
        if len(texts) != len(segments):
            raise ValueError("texts and segments must have the same length")
        for index, (text, parts) in enumerate(zip(texts, segments)):
            if not isinstance(text, str):
                raise ValueError(f"text {index} must be a string")
            if isinstance(parts, str) or any(not isinstance(part, str) for part in parts):
                raise ValueError(f"segments {index} must be a sequence of strings")
            if "".join(parts) != text or (text and any(not part for part in parts)):
                raise ValueError(f"segments {index} must partition the original text exactly")

    def _build_sequence(self, text: str, seg: Sequence[str]) -> BoundarySequence:
        categories = np.array([_char_category(ch) for ch in text], dtype=int)
        labels = np.array(self._segments_to_boundaries(text, seg), dtype=int)
        seq = BoundarySequence(
            text=text,
            categories=categories,
            labels=labels,
            curvature=np.zeros(len(text), dtype=self.dtype),
            phases=np.zeros((len(text), 3), dtype=self.dtype),
            lexical_features=self._lexical_features(text) if self.lexical_buckets else None,
            lexical_buckets=self.lexical_buckets,
        )
        self._refresh_phase_features(seq)
        return seq

    def _lexical_features(self, text: str) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        """Seven fixed positional templates with deterministic signed hashing.

        Offsets are relative to the character left of each boundary. Unicode
        code points and explicit BOS/EOS markers keep keys unambiguous. Hash
        collisions add their signed counts in both forward and backward paths.
        The table bounds parameter memory independently of the vocabulary.
        """
        if not self.lexical_buckets:
            return ()
        features = []
        for boundary in range(max(0, len(text) - 1)):
            counts: Dict[int, int] = {}
            for template in _LEXICAL_TEMPLATES:
                values = []
                for offset in template:
                    index = boundary + offset
                    values.append("BOS" if index < 0 else "EOS" if index >= len(text) else str(ord(text[index])))
                key = f"{','.join(map(str, template))}:{','.join(values)}".encode("ascii")
                hashed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
                bucket = (hashed & ((1 << 63) - 1)) % self.lexical_buckets
                sign = -1 if hashed >> 63 else 1
                counts[bucket] = counts.get(bucket, 0) + sign
            features.append(tuple((bucket, count) for bucket, count in sorted(counts.items()) if count))
        return tuple(features)

    def _refresh_phase_features(self, seq: BoundarySequence) -> None:
        """Only text, categories, labels and lexical hashes are immutable."""
        if not seq.text:
            return
        curvature_raw = self.phase.curvature(seq.text)
        seq.curvature = np.array(
            curvature_raw.to_list() if hasattr(curvature_raw, "to_list") else list(curvature_raw),
            dtype=self.dtype,
        )
        seq.phases = np.array([self.phase.phase_triplet(ch) for ch in seq.text], dtype=self.dtype)

    def _segments_to_boundaries(self, text: str, seg: Sequence[str]) -> List[int]:
        cuts = set()
        idx = 0
        for tok in seg:
            idx += len(tok)
            if idx < len(text):
                cuts.add(idx)
        return [1 if (i + 1) in cuts else 0 for i in range(len(text) - 1)]

    def _labels_to_int(self, labels: Sequence[int]) -> List[int]:
        if hasattr(labels, "tolist"):
            raw = labels.tolist()
        else:
            raw = list(labels)
        return [int(round(float(v))) for v in raw]

    # ------------------------------------------------------------------
    # Forward utilities
    # ------------------------------------------------------------------
    def _window_indices(self, idx: int, length: int) -> List[int]:
        indices: List[int] = []
        for offset in range(-self.window + 1, 1):
            pos = idx + offset
            if pos < 0:
                indices.append(-1)
            else:
                indices.append(min(pos, length - 1))
        for offset in range(1, self.window + 1):
            pos = idx + offset
            if pos >= length:
                indices.append(length - 1)
            else:
                indices.append(pos)
        return indices

    def _window_vector(self, embeddings: np.ndarray, indices: List[int]) -> np.ndarray:
        if len(indices) == 0:
            return np.zeros(0, dtype=self.dtype)
        if hasattr(embeddings, "shape"):
            window = np.zeros((len(indices), self.emb_dim), dtype=self.dtype)
            for pos, idx in enumerate(indices):
                if 0 <= idx < embeddings.shape[0]:
                    window[pos] = embeddings[int(idx)]
            return np.reshape(window, (len(indices) * self.emb_dim,))
        vec: List[float] = []
        for idx in indices:
            if idx < 0 or idx >= len(embeddings):
                vec.extend([0.0] * self.emb_dim)
            else:
                vec.extend(embeddings[idx])
        return np.array(vec, dtype=self.dtype)

    def _gate_features(self, seq: BoundarySequence, idx: int) -> np.ndarray:
        curv = seq.curvature
        phases = seq.phases
        left_phase = float(phases[idx][0]) if idx < len(phases) else 0.0
        right_phase = float(phases[idx + 1][1]) if idx + 1 < len(phases) else float(phases[-1][1])
        curv_left = float(curv[idx]) if idx < len(curv) else 0.0
        curv_right = float(curv[idx + 1]) if idx + 1 < len(curv) else float(curv[-1])
        phase_feature = math.sin(left_phase - right_phase)
        curv_feature = math.tanh(0.5 * (curv_left + curv_right))
        return np.array([phase_feature, curv_feature, 1.0], dtype=self.dtype)

    def _linear_forward(self, window_vec: np.ndarray) -> np.ndarray:
        vec = window_vec if hasattr(window_vec, "shape") else np.array(window_vec, dtype=self.dtype)
        return self.W_window @ vec + self.b_window

    def _forward_sequence(self, seq: BoundarySequence) -> Tuple[List[float], List[Dict[str, object]]]:
        # The basis changes during training and can be restored by early stopping.
        # Refresh cached train and validation features using the same basis as inference.
        self._refresh_phase_features(seq)
        if self.lexical_buckets and (
            seq.lexical_features is None or seq.lexical_buckets != self.lexical_buckets
        ):
            seq.lexical_features = self._lexical_features(seq.text)
            seq.lexical_buckets = self.lexical_buckets
        length = len(seq.categories)
        embeddings = np.zeros((length, self.emb_dim), dtype=self.dtype)
        for i, cat in enumerate(seq.categories):
            embeddings[i] = self.embeddings[int(cat)]
        ctx_H = self._encode_context(seq)
        caches: List[Dict[str, object]] = []
        logits: List[float] = []
        for idx in range(len(seq.labels)):
            indices = self._window_indices(idx, len(embeddings))
            window_vec = self._window_vector(embeddings, indices)
            pre = self._linear_forward(window_vec)
            hidden = np.tanh(pre) if hasattr(np, "tanh") else np.array([math.tanh(float(v)) for v in pre], dtype=self.dtype)
            gate_feats = self._gate_features(seq, idx)
            core = float(np.dot(self.W_out, hidden)) + self.b_out
            gate_score = float(np.dot(self.gate_w, gate_feats)) + self.gate_b
            ctx_score = 0.0
            ctx_delta = None
            ctx_hidden = None
            if (
                ctx_H is not None
                and self.ctx_W1 is not None
                and self.ctx_b1 is not None
                and self.ctx_w is not None
                and idx + 1 < ctx_H.shape[0]
            ):
                delta = ctx_H[idx + 1] - ctx_H[idx]
                ctx_delta = np.asarray(delta, dtype=self.dtype)
                ctx_pre = self.ctx_W1 @ ctx_delta + self.ctx_b1
                ctx_hidden = np.tanh(ctx_pre)
                ctx_score = float(np.dot(self.ctx_w, ctx_hidden)) + float(self.ctx_b)
            lexical_features = seq.lexical_features[idx] if self.lexical_buckets else ()
            lexical_score = sum(float(self.lexical_weights[bucket]) * count for bucket, count in lexical_features)
            logits.append(core + gate_score + ctx_score + lexical_score)
            caches.append(
                {
                    "indices": indices,
                    "window": window_vec,
                    "pre": pre,
                    "hidden": hidden,
                    "gate_feats": gate_feats,
                    "ctx_delta": ctx_delta,
                    "ctx_hidden": ctx_hidden,
                    "lexical_features": lexical_features,
                }
            )
        return logits, caches

    # ------------------------------------------------------------------
    # CRF helpers
    # ------------------------------------------------------------------
    def _logsumexp(self, values: Iterable[float]) -> float:
        vals = list(values)
        m = max(vals)
        if m == float("-inf"):
            return m
        return m + math.log(sum(math.exp(v - m) for v in vals))

    def _crf_loss(self, logits: List[float], labels: List[int]) -> Tuple[float, List[float], List[List[float]], List[List[float]]]:
        length = len(logits)
        if length == 0:
            return 0.0, [], [[0.0, 0.0], [0.0, 0.0]], []
        # Avoid float32 accumulation across long sequences, even when model
        # parameters are float32. Dynamic-programming scores use Python floats.
        trans = [[float(value) for value in row] for row in self.transitions]
        logits = [float(value) for value in logits]
        emit = [[0.0 for _ in range(length)], logits]
        alpha = [[0.0, 0.0] for _ in range(length)]
        alpha[0][0] = emit[0][0]
        alpha[0][1] = emit[1][0]
        for i in range(1, length):
            for state in (0, 1):
                scores = [alpha[i - 1][prev] + trans[prev][state] for prev in (0, 1)]
                alpha[i][state] = emit[state][i] + self._logsumexp(scores)
        log_z = self._logsumexp(alpha[-1])

        score = emit[labels[0]][0]
        for i in range(1, length):
            score += emit[labels[i]][i] + trans[labels[i - 1]][labels[i]]
        nll = log_z - score

        beta = [[0.0, 0.0] for _ in range(length)]
        beta[-1][0] = 0.0
        beta[-1][1] = 0.0
        for i in range(length - 2, -1, -1):
            for state in (0, 1):
                scores = [
                    trans[state][next_state]
                    + emit[next_state][i + 1]
                    + beta[i + 1][next_state]
                    for next_state in (0, 1)
                ]
                beta[i][state] = self._logsumexp(scores)

        grad_logits = [0.0 for _ in range(length)]
        marginals: List[List[float]] = []
        for i in range(length):
            gamma0 = alpha[i][0] + beta[i][0]
            gamma1 = alpha[i][1] + beta[i][1]
            # Local two-state normalization keeps probabilities in [0, 1]
            # without relying on cancellation against the global partition.
            difference = gamma1 - gamma0
            if difference >= 0:
                p1 = 1.0 / (1.0 + math.exp(-difference))
            else:
                exponent = math.exp(difference)
                p1 = exponent / (1.0 + exponent)
            p0 = 1.0 - p1
            grad_logits[i] = p1 - labels[i]
            marginals.append([p0, p1])

        grad_trans = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(1, length):
            pair_scores = [
                alpha[i - 1][prev] + trans[prev][state] + emit[state][i] + beta[i][state]
                for prev in (0, 1) for state in (0, 1)
            ]
            maximum = max(pair_scores)
            pair_weights = [math.exp(score - maximum) for score in pair_scores]
            pair_total = sum(pair_weights)
            for prev in (0, 1):
                for state in (0, 1):
                    grad_trans[prev][state] += pair_weights[2 * prev + state] / pair_total
        for i in range(1, length):
            grad_trans[labels[i - 1]][labels[i]] -= 1.0
        grad_logits_arr = np.array(grad_logits, dtype=self.dtype)
        grad_trans_arr = np.array(grad_trans, dtype=self.dtype)
        marginals_arr = [np.array(m, dtype=self.dtype) for m in marginals]
        return nll, grad_logits_arr, grad_trans_arr, marginals_arr

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _split_indices(self, total: int, validation_split: float) -> Tuple[List[int], List[int]]:
        if total <= 1 or validation_split <= 0.0:
            return list(range(total)), []
        indices = list(range(total))
        self.rng.shuffle(indices)
        train_count = int(total * (1.0 - validation_split))
        if train_count <= 0:
            train_count = 1
        if train_count >= total:
            train_count = total - 1
        train_idx = indices[:train_count]
        val_idx = indices[train_count:]
        return train_idx, val_idx

    def train(
        self,
        texts: Sequence[str],
        segments: Sequence[Sequence[str]],
        cfg: Optional[StudentTrainingConfig] = None,
    ) -> Dict[str, object]:
        cfg = cfg or StudentTrainingConfig()
        lexical_buckets = self._validate_lexical_config(cfg)
        texts_list = list(texts)
        segments_list = [list(seg) if not isinstance(seg, str) else seg for seg in segments]
        self._validate_training_data(texts_list, segments_list)
        if cfg.batch_size <= 0 or cfg.epochs < 0:
            raise ValueError("batch_size must be positive and epochs must be nonnegative")
        if cfg.hidden_dim <= 0 or cfg.emb_dim <= 0 or cfg.window <= 0:
            raise ValueError("hidden_dim, emb_dim and window must be positive")
        if not 0.0 <= cfg.validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")
        # Validate before configuring either the Python or a native backend.
        texts, segments = texts_list, segments_list
        if not lexical_buckets:
            # A successful native retrain must not retain an earlier residual.
            self.lexical_buckets = 0
            self.lexical_weights = np.zeros(0, dtype=self.dtype)
        self.best_state = None
        self.history = []
        fallbacks: List[str] = []
        use_context = bool(getattr(cfg, "use_encoder_context", False)) and self.encoder_adapter is not None
        if not use_context and not lexical_buckets and self.julia_backend is not None:
            cfg_dict = dict(cfg.__dict__)
            try:
                self._select_backend_device(cfg.device_preference)
                summary = self.julia_backend.train(texts, segments, cfg_dict)
                self._fitted_backend = "julia"
                self.use_encoder_context = False
                if isinstance(summary, dict):
                    self.history = list(summary.get("history", []))
                    backend_id = f"julia:{self.julia_backend.device}"
                    meta = self._backend_metadata(backend_id, fallbacks)
                    summary.setdefault("backend", backend_id)
                    summary.setdefault("available_devices", self.backend_inventory())
                    summary.setdefault("backend_used", meta["backend_used"])
                    if meta["backend_fallbacks"]:
                        summary.setdefault("backend_fallbacks", meta["backend_fallbacks"])
                return summary
            except Exception:
                backend_id = f"julia:{getattr(self.julia_backend, 'device', 'unknown')}"
                fallbacks.append(backend_id)
                logger.warning("Julia backend training failed, falling back to alternative implementation.", exc_info=True)
                self.julia_backend = None
        if not use_context and not lexical_buckets and self.compiled_backend is not None:
            cfg_dict = dict(cfg.__dict__)
            try:
                self._select_backend_device(cfg.device_preference)
                summary = self.compiled_backend.train(texts, segments, cfg_dict)
                self._fitted_backend = "compiled"
                self.use_encoder_context = False
                if isinstance(summary, dict):
                    self.history = list(summary.get("history", []))
                    backend_id = f"compiled:{self.compiled_backend.device}"
                    meta = self._backend_metadata(backend_id, fallbacks)
                    summary.setdefault("backend", backend_id)
                    summary.setdefault("available_devices", self.backend_inventory())
                    summary.setdefault("backend_used", meta["backend_used"])
                    if meta["backend_fallbacks"]:
                        summary.setdefault("backend_fallbacks", meta["backend_fallbacks"])
                return summary
            except Exception:
                backend_id = f"compiled:{getattr(self.compiled_backend, 'device', 'unknown')}"
                fallbacks.append(backend_id)
                # Fallback to pure NumPy implementation if compiled backend fails.
                logger.warning("Compiled backend training failed, reverting to pure Python implementation.", exc_info=True)
                self.compiled_backend = None
        self.configure(cfg)
        dataset_size = len(texts_list)
        if dataset_size == 0:
            self.history = []
            backend = "numpy"
            meta = self._backend_metadata(backend, fallbacks)
            summary: Dict[str, object] = {
                "train_sequences": 0,
                "val_sequences": 0,
                "train_tokens": 0,
                "train_seconds": 0.0,
                "tokens_per_second": 0.0,
                "history": [],
                "backend": backend,
                "backend_used": backend,
                "dtype": getattr(self.dtype, "name", getattr(self.dtype, "__name__", str(self.dtype))),
                "cache_sequences": bool(cfg.cache_sequences),
                "shuffle_train": bool(cfg.shuffle_train),
                "cached_sequences": 0,
                "lexical_buckets": self.lexical_buckets,
                "lexical_lr": cfg.lexical_lr,
                "available_devices": self.backend_inventory(),
                "backend_used": meta["backend_used"],
            }
            if meta["backend_fallbacks"]:
                summary["backend_fallbacks"] = meta["backend_fallbacks"]
            return summary

        train_idx, val_idx = self._split_indices(dataset_size, cfg.validation_split)
        sequence_cache: Dict[int, BoundarySequence] = {}

        def get_sequence(idx: int) -> BoundarySequence:
            if cfg.cache_sequences and idx in sequence_cache:
                return sequence_cache[idx]
            seq = self._build_sequence(texts_list[idx], segments_list[idx])
            if cfg.cache_sequences:
                sequence_cache[idx] = seq
            return seq

        val_seqs = [get_sequence(idx) for idx in val_idx]
        token_cache: Dict[int, int] = {}

        def seq_token_count(idx: int) -> int:
            if idx in token_cache:
                return token_cache[idx]
            seq = get_sequence(idx)
            tokens = int(len(seq.labels)) + 1
            token_cache[idx] = tokens
            return tokens

        train_tokens = sum(seq_token_count(idx) for idx in train_idx)
        best_val = float("inf")
        best_epoch: Optional[int] = None
        patience = 0
        history: List[Dict[str, float]] = []
        start_time = time.perf_counter()
        for epoch in range(cfg.epochs):
            if cfg.shuffle_train:
                self.rng.shuffle(train_idx)
            accum = self._zero_grad()
            batch_count = 0
            total_loss = 0.0
            for idx in train_idx:
                seq = get_sequence(idx)
                loss, grads, marginals = self._sequence_gradients(seq, cfg)
                total_loss += loss
                self._accumulate(accum, grads)
                batch_count += 1
                if batch_count % cfg.batch_size == 0:
                    self._apply_gradients(accum, cfg, cfg.batch_size)
                    accum = self._zero_grad()
            if batch_count % cfg.batch_size != 0:
                self._apply_gradients(accum, cfg, batch_count % cfg.batch_size)
            train_count = max(1, len(train_idx))
            metrics = {"epoch": float(epoch + 1), "train_loss": float(total_loss / train_count)}
            if val_seqs:
                val_loss, val_f1 = self.evaluate(val_seqs)
                metrics.update({"val_loss": val_loss, "val_f1": val_f1})
                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    best_epoch = epoch + 1
                    patience = 0
                    self.best_state = self._capture_state()
                else:
                    patience += 1
                    if patience >= cfg.patience:
                        history.append(metrics)
                        break
            history.append(metrics)
        if self.best_state is not None:
            self._restore_state(self.best_state)
        self.history = history
        elapsed = max(1e-9, time.perf_counter() - start_time)
        summary: Dict[str, object] = {
            "train_sequences": len(train_idx),
            "train_indices": sorted(train_idx),
            "validation_indices": sorted(val_idx),
            "history": history,
            "train_tokens": train_tokens,
            "train_seconds": elapsed,
            "tokens_per_second": train_tokens * len(history) / elapsed if train_tokens else 0.0,
            "processed_train_tokens": train_tokens * len(history),
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "backend": "numpy",
            "dtype": getattr(self.dtype, "name", getattr(self.dtype, "__name__", str(self.dtype))),
            "cache_sequences": bool(cfg.cache_sequences),
            "shuffle_train": bool(cfg.shuffle_train),
            "cached_sequences": len(sequence_cache) if cfg.cache_sequences else 0,
            "available_devices": self.backend_inventory(),
            "device_preference": cfg.device_preference,
            "lexical_buckets": self.lexical_buckets,
            "lexical_lr": cfg.lexical_lr,
        }
        meta = self._backend_metadata(str(summary["backend"]), fallbacks)
        summary.setdefault("backend_used", meta["backend_used"])
        if meta["backend_fallbacks"] and "backend_fallbacks" not in summary:
            summary["backend_fallbacks"] = meta["backend_fallbacks"]
        if val_seqs:
            summary["val_sequences"] = len(val_seqs)
            summary["val_tokens"] = sum(int(len(seq.labels)) + 1 for seq in val_seqs)
            # Report the model actually returned, which may predate the final epoch.
            val_loss, val_f1 = self.evaluate(val_seqs)
            summary["val_loss"] = val_loss
            summary["val_f1"] = val_f1
        backend_used = summary.get("backend")
        if isinstance(backend_used, str):
            self._last_backend_used = backend_used
        summary.setdefault("backend_used", self._last_backend_used)
        return summary

    def _zero_grad(self) -> Dict[str, object]:
        num_classes = len(self.embeddings)
        grads: Dict[str, object] = {
            "embeddings": np.zeros((num_classes, self.emb_dim), dtype=self.dtype),
            "W_window": np.zeros((self.hidden_dim, self.window_dim), dtype=self.dtype),
            "b_window": np.zeros(self.hidden_dim, dtype=self.dtype),
            "W_out": np.zeros(self.hidden_dim, dtype=self.dtype),
            "b_out": 0.0,
            "gate_w": np.zeros(len(self.gate_w), dtype=self.dtype),
            "gate_b": 0.0,
            "transitions": np.zeros((2, 2), dtype=self.dtype),
        }
        if self.lexical_buckets:
            grads["lexical_weights"] = {}
        if self.ctx_W1 is not None:
            grads["ctx_W1"] = np.zeros_like(self.ctx_W1)
        if self.ctx_b1 is not None:
            grads["ctx_b1"] = np.zeros_like(self.ctx_b1)
        if self.ctx_w is not None:
            grads["ctx_w"] = np.zeros_like(self.ctx_w)
            grads["ctx_b"] = 0.0
        return grads

    def _sequence_gradients(
        self, seq: BoundarySequence, cfg: StudentTrainingConfig
    ) -> Tuple[float, Dict[str, object], List[np.ndarray]]:
        logits, caches = self._forward_sequence(seq)
        label_list = self._labels_to_int(seq.labels)
        loss, grad_logits, grad_trans, marginals = self._crf_loss(logits, label_list)
        grads = self._zero_grad()
        embed_grads = np.zeros((len(seq.categories), self.emb_dim), dtype=self.dtype)
        for i, cache in enumerate(caches):
            grad_logit = float(grad_logits[i])
            hidden = cache["hidden"]
            pre = cache["pre"]
            window_vec = cache["window"]
            indices = cache["indices"]
            gate_feats = cache["gate_feats"]
            ctx_delta = cache.get("ctx_delta")
            ctx_hidden = cache.get("ctx_hidden")
            if self.lexical_buckets:
                lexical_grads = grads["lexical_weights"]
                for bucket, count in cache["lexical_features"]:
                    lexical_grads[bucket] = lexical_grads.get(bucket, 0.0) + grad_logit * count

            grads["gate_w"] += grad_logit * gate_feats
            grads["gate_b"] += grad_logit
            self.phase.apply_error(seq.text, i, grad_logit, scale=cfg.phase_lr)

            grads["W_out"] += grad_logit * hidden
            grads["b_out"] += grad_logit

            if ctx_delta is not None and ctx_hidden is not None and "ctx_w" in grads:
                ctx_hidden_arr = np.asarray(ctx_hidden, dtype=self.dtype)
                ctx_delta_arr = np.asarray(ctx_delta, dtype=self.dtype)
                grads["ctx_w"] += grad_logit * ctx_hidden_arr
                grads["ctx_b"] += grad_logit
                if "ctx_W1" in grads and "ctx_b1" in grads and self.ctx_w is not None:
                    ctx_hidden_sq = ctx_hidden_arr * ctx_hidden_arr
                    grad_ctx_pre = (grad_logit * self.ctx_w) * (np.ones_like(ctx_hidden_sq) - ctx_hidden_sq)
                    grads["ctx_b1"] += grad_ctx_pre
                    grads["ctx_W1"] += np.outer(grad_ctx_pre, ctx_delta_arr)

            grad_hidden = grad_logit * self.W_out
            hidden_sq = hidden * hidden if hasattr(hidden, "__mul__") else np.array([float(h) ** 2 for h in hidden])
            grad_pre = grad_hidden * (1.0 - hidden_sq)
            grads["b_window"] += grad_pre
            grads["W_window"] += np.outer(grad_pre, window_vec)
            grad_window = self.W_window.T @ grad_pre
            grad_window_matrix = np.reshape(grad_window, (len(indices), self.emb_dim))
            for pos, char_idx in enumerate(indices):
                if 0 <= char_idx < embed_grads.shape[0]:
                    embed_grads[int(char_idx)] += grad_window_matrix[pos]

        for pos, cat in enumerate(seq.categories):
            grads["embeddings"][int(cat)] += embed_grads[pos]

        grads["transitions"] += grad_trans

        if self.encoder_adapter is not None:
            gate_targets = self._char_gate_targets(label_list, marginals)
            base_gate = [sigmoid(float(c)) for c in (seq.curvature.tolist() if hasattr(seq.curvature, "tolist") else seq.curvature)]
            self.encoder_adapter.tune_from_boundary(base_gate, gate_targets, lr=cfg.encoder_lr)

        loss += 0.5 * cfg.reg * self._l2_norm()
        return float(loss), grads, marginals

    def _char_gate_targets(self, labels: List[int], marginals: List[List[float]]) -> List[float]:
        length = len(labels) + 1
        targets = [0.0 for _ in range(length)]
        preds = [m[1] for m in marginals]
        for i, label in enumerate(labels):
            if label >= 0.5:
                targets[i] = max(targets[i], 1.0)
                targets[i + 1] = max(targets[i + 1], 1.0)
            else:
                val = preds[i]
                targets[i] = max(targets[i], val)
                targets[i + 1] = max(targets[i + 1], val)
        return targets

    def _l2_norm(self) -> float:
        total = 0.0
        total += float(np.sum(self.W_window * self.W_window))
        total += float(np.sum(self.W_out * self.W_out))
        total += float(np.sum(self.gate_w * self.gate_w))
        total += float(np.sum(self.embeddings * self.embeddings))
        if self.lexical_buckets:
            total += float(np.sum(self.lexical_weights * self.lexical_weights))
        if self.ctx_W1 is not None:
            total += float(np.sum(self.ctx_W1 * self.ctx_W1))
        if self.ctx_w is not None:
            total += float(np.sum(self.ctx_w * self.ctx_w))
        return total

    def _accumulate(self, accum: Dict[str, object], grads: Dict[str, object]) -> None:
        accum["embeddings"] += grads["embeddings"]
        accum["W_window"] += grads["W_window"]
        accum["b_window"] += grads["b_window"]
        accum["W_out"] += grads["W_out"]
        accum["b_out"] += grads["b_out"]
        accum["gate_w"] += grads["gate_w"]
        accum["gate_b"] += grads["gate_b"]
        accum["transitions"] += grads["transitions"]
        if "lexical_weights" in accum:
            for bucket, value in grads["lexical_weights"].items():
                accum["lexical_weights"][bucket] = accum["lexical_weights"].get(bucket, 0.0) + value
        if "ctx_W1" in accum and "ctx_W1" in grads:
            accum["ctx_W1"] += grads["ctx_W1"]
        if "ctx_b1" in accum and "ctx_b1" in grads:
            accum["ctx_b1"] += grads["ctx_b1"]
        if "ctx_w" in accum and "ctx_w" in grads:
            accum["ctx_w"] += grads["ctx_w"]
            accum["ctx_b"] += grads["ctx_b"]

    def _apply_gradients(self, grads: Dict[str, object], cfg: StudentTrainingConfig, batch_size: int) -> None:
        base_scale = cfg.lr / max(1, batch_size)
        grad_scale = 1.0
        if cfg.max_grad_norm:
            norm = self._grad_norm(grads)
            if norm > cfg.max_grad_norm:
                grad_scale = cfg.max_grad_norm / (norm + 1e-9)
        scale = base_scale * grad_scale
        crf_scale = cfg.crf_lr / max(1, batch_size) * grad_scale

        self.embeddings -= scale * (grads["embeddings"] + cfg.reg * self.embeddings)
        self.W_window -= scale * (grads["W_window"] + cfg.reg * self.W_window)
        self.b_window -= scale * grads["b_window"]
        self.W_out -= scale * (grads["W_out"] + cfg.reg * self.W_out)
        self.b_out -= scale * grads["b_out"]
        self.gate_w -= scale * (grads["gate_w"] + cfg.reg * self.gate_w)
        self.gate_b -= scale * grads["gate_b"]
        self.transitions -= crf_scale * (grads["transitions"] + cfg.reg * self.transitions)
        if self.lexical_buckets:
            lexical_scale = cfg.lexical_lr / max(1, batch_size) * grad_scale
            self.lexical_weights *= 1.0 - lexical_scale * cfg.reg
            for bucket, value in grads["lexical_weights"].items():
                self.lexical_weights[bucket] -= lexical_scale * value
        if any(key in grads for key in ("ctx_W1", "ctx_b1", "ctx_w")):
            ctx_scale = cfg.context_lr / max(1, batch_size) * grad_scale
            if self.ctx_W1 is not None and "ctx_W1" in grads:
                self.ctx_W1 -= ctx_scale * (grads["ctx_W1"] + cfg.reg * self.ctx_W1)
            if self.ctx_b1 is not None and "ctx_b1" in grads:
                self.ctx_b1 -= ctx_scale * grads["ctx_b1"]
            if self.ctx_w is not None and "ctx_w" in grads:
                self.ctx_w -= ctx_scale * (grads["ctx_w"] + cfg.reg * self.ctx_w)
                self.ctx_b -= ctx_scale * grads.get("ctx_b", 0.0)

    def _grad_norm(self, grads: Dict[str, object]) -> float:
        total = 0.0
        total += float(np.sum(grads["embeddings"] * grads["embeddings"]))
        total += float(np.sum(grads["W_window"] * grads["W_window"]))
        total += float(np.sum(grads["b_window"] * grads["b_window"]))
        total += float(np.sum(grads["W_out"] * grads["W_out"]))
        total += float(np.sum(grads["gate_w"] * grads["gate_w"]))
        total += float(np.sum(grads["transitions"] * grads["transitions"]))
        total += float(grads["b_out"] ** 2)
        total += float(grads["gate_b"] ** 2)
        total += sum(float(value) ** 2 for value in grads.get("lexical_weights", {}).values())
        if "ctx_W1" in grads:
            total += float(np.sum(grads["ctx_W1"] * grads["ctx_W1"]))
        if "ctx_b1" in grads:
            total += float(np.sum(grads["ctx_b1"] * grads["ctx_b1"]))
        if "ctx_w" in grads:
            total += float(np.sum(grads["ctx_w"] * grads["ctx_w"]))
            total += float(grads.get("ctx_b", 0.0) ** 2)
        return math.sqrt(total)

    def _capture_state(self) -> Dict[str, object]:
        state: Dict[str, object] = {
            "fitted_backend": self._fitted_backend or "python",
            "architecture": {
                "hidden_dim": self.hidden_dim,
                "emb_dim": self.emb_dim,
                "window": self.window,
                "dtype": getattr(self.dtype, "__name__", str(self.dtype)),
            },
            "_phase": {
                "dim": self.phase.dim,
                "lr": self.phase.lr,
                "basis": self.phase.export_state().basis,
            },
            "embeddings": self.embeddings.tolist() if hasattr(self.embeddings, "tolist") else [row[:] for row in self.embeddings],
            "W_window": self.W_window.tolist() if hasattr(self.W_window, "tolist") else [row[:] for row in self.W_window],
            "b_window": self.b_window.tolist() if hasattr(self.b_window, "tolist") else self.b_window[:],
            "W_out": self.W_out.tolist() if hasattr(self.W_out, "tolist") else self.W_out[:],
            "b_out": self.b_out,
            "gate_w": self.gate_w.tolist() if hasattr(self.gate_w, "tolist") else self.gate_w[:],
            "gate_b": self.gate_b,
            "transitions": self.transitions.tolist() if hasattr(self.transitions, "tolist") else [row[:] for row in self.transitions],
            "lexical": {
                "version": _LEXICAL_VERSION,
                "buckets": self.lexical_buckets,
                "weights": self.lexical_weights.tolist(),
            },
        }
        if self.encoder_adapter is not None and hasattr(self.encoder_adapter, "export_state"):
            state["_encoder"] = copy.deepcopy(self.encoder_adapter.export_state())
        state["use_encoder_context"] = bool(self.use_encoder_context)
        state["context_hidden_dim"] = int(self.context_hidden_dim)
        if self.ctx_W1 is not None:
            state["ctx_W1"] = (
                self.ctx_W1.tolist()
                if hasattr(self.ctx_W1, "tolist")
                else [row[:] for row in self.ctx_W1]
            )
        if self.ctx_b1 is not None:
            state["ctx_b1"] = self.ctx_b1.tolist() if hasattr(self.ctx_b1, "tolist") else list(self.ctx_b1)
        if self.ctx_w is not None:
            state["ctx_w"] = self.ctx_w.tolist() if hasattr(self.ctx_w, "tolist") else list(self.ctx_w)
            state["ctx_b"] = float(self.ctx_b)
        return state

    def _restore_state(self, state: Dict[str, object]) -> None:
        self._fitted_backend = state.get("fitted_backend", "python")
        architecture = state.get("architecture", {})
        if architecture.get("dtype") == "float64" and hasattr(np, "float64"):
            self.dtype = np.float64
        elif architecture.get("dtype") == "float32" and hasattr(np, "float32"):
            self.dtype = np.float32
        lexical = state.get("lexical")
        if lexical is None:
            self.lexical_buckets = 0
            self.lexical_weights = np.zeros(0, dtype=self.dtype)
        else:
            if lexical.get("version") != _LEXICAL_VERSION:
                raise ValueError("unsupported lexical feature version in checkpoint")
            buckets = lexical["buckets"]
            if not isinstance(buckets, int) or buckets < 0:
                raise ValueError("invalid lexical bucket count in checkpoint")
            weights = np.array(lexical["weights"], dtype=self.dtype)
            if weights.shape != (buckets,) or not np.all(np.isfinite(weights)):
                raise ValueError("invalid lexical weights in checkpoint")
            self.lexical_buckets, self.lexical_weights = buckets, weights
        self.embeddings = np.array(state["embeddings"], dtype=self.dtype)
        self.W_window = np.array(state["W_window"], dtype=self.dtype)
        # Older checkpoints have no architecture block; infer it from the weights.
        self.emb_dim = int(self.embeddings.shape[1])
        self.hidden_dim = int(self.W_window.shape[0])
        self.window_dim = int(self.W_window.shape[1])
        if self.window_dim % (2 * self.emb_dim):
            raise ValueError("checkpoint window weights do not match embedding dimensions")
        self.window = self.window_dim // (2 * self.emb_dim)
        if "_phase" in state:
            phase_state = state["_phase"]
            self.phase.dim = int(phase_state["dim"])
            self.phase.lr = float(phase_state["lr"])
            self.phase.load_state(PhaseBasisState(basis=copy.deepcopy(phase_state["basis"])))
        if "_encoder" in state and self.encoder_adapter is not None:
            self.encoder_adapter.load_state(copy.deepcopy(state["_encoder"]))
        self.b_window = np.array(state["b_window"], dtype=self.dtype)
        self.W_out = np.array(state["W_out"], dtype=self.dtype)
        self.b_out = float(state["b_out"])
        self.gate_w = np.array(state["gate_w"], dtype=self.dtype)
        self.gate_b = float(state["gate_b"])
        self.transitions = np.array(state["transitions"], dtype=self.dtype)
        self.use_encoder_context = bool(state.get("use_encoder_context", self.use_encoder_context))
        if "context_hidden_dim" in state:
            try:
                self.context_hidden_dim = max(1, int(state.get("context_hidden_dim", self.context_hidden_dim)))
            except Exception:
                self.context_hidden_dim = max(1, int(self.context_hidden_dim))
        if "ctx_W1" in state:
            self.ctx_W1 = np.array(state["ctx_W1"], dtype=self.dtype)
        else:
            self.ctx_W1 = None
        if "ctx_b1" in state:
            self.ctx_b1 = np.array(state["ctx_b1"], dtype=self.dtype)
        else:
            self.ctx_b1 = None
        if "ctx_w" in state:
            self.ctx_w = np.array(state["ctx_w"], dtype=self.dtype)
            self.ctx_b = float(state.get("ctx_b", 0.0))
            if self.ctx_W1 is None and self.ctx_w.ndim == 1 and self.ctx_w.shape[0] >= 1:
                # Backwards-compat: older checkpoints stored a linear ctx_w over encoder deltas.
                dim = int(self.ctx_w.shape[0])
                scale = 0.25
                self.context_hidden_dim = dim
                self.ctx_W1 = (np.eye(dim, dtype=self.dtype) * scale).astype(self.dtype, copy=False)
                self.ctx_b1 = np.zeros(dim, dtype=self.dtype)
                self.ctx_w = (self.ctx_w / scale).astype(self.dtype, copy=False)
        else:
            self.ctx_W1 = None
            self.ctx_b1 = None
            self.ctx_w = None
            self.ctx_b = 0.0

    def export_state(self) -> Dict[str, object]:
        self._require_fitted_backend()
        state = self._capture_state()
        if self._fitted_backend == "julia":
            try:
                state["_julia"] = {
                    "backend": self.julia_backend.backend,
                    "device": self.julia_backend.device,
                    "state": self.julia_backend.export_state(),
                }
            except Exception as exc:
                raise RuntimeError("Failed to export fitted Julia boundary backend state") from exc
        if self._fitted_backend == "compiled":
            try:
                state["_compiled"] = {
                    "device": self.compiled_backend.device,
                    "state": self.compiled_backend.export_state(),
                }
            except Exception as exc:
                raise RuntimeError("Failed to export fitted compiled boundary backend state") from exc
        return state

    def validate_state_ownership(self, state: Dict[str, object]) -> str:
        """Resolve fitted ownership and availability without changing live state."""
        compiled_state = state.get("_compiled") if isinstance(state, dict) else None
        julia_state = state.get("_julia") if isinstance(state, dict) else None
        fitted = state.get("fitted_backend")
        if fitted is None:
            candidates = [name for name, value in (("compiled", compiled_state), ("julia", julia_state)) if value is not None]
            lexical = state.get("lexical", {})
            python_features = state.get("use_encoder_context", False) or lexical.get("buckets", 0)
            # Old native training returned before updating use_encoder_context,
            # and old exports also included unrelated installed native heads.
            # Neither a True flag nor a native payload proves ownership when
            # both are present. Refuse to silently select unsynchronized weights.
            if len(candidates) > 1 or (candidates and python_features):
                raise ValueError(
                    "Legacy checkpoint has ambiguous fitted backend ownership; "
                    "re-export with explicit fitted_backend provenance"
                )
            fitted = candidates[0] if candidates else "python"
        if fitted not in ("python", "compiled", "julia"):
            raise ValueError("Unsupported fitted boundary backend in checkpoint")
        native_state = compiled_state if fitted == "compiled" else julia_state
        native_handle = getattr(self, f"{fitted}_backend", None)
        if fitted != "python":
            if native_state is None or native_handle is None:
                raise ValueError(f"Checkpoint requires its fitted {fitted} backend and serialized weights")
            if state.get("use_encoder_context", False) or state.get("lexical", {}).get("buckets", 0):
                raise ValueError("Native checkpoint cannot use Python-only context or lexical emissions")
        return fitted

    def load_state(self, state: Dict[str, object]) -> None:
        fitted = self.validate_state_ownership(state)
        native_state = state.get(f"_{fitted}")
        native_handle = getattr(self, f"{fitted}_backend", None)
        base = dict(state) if isinstance(state, dict) else state
        if isinstance(base, dict) and "_compiled" in base:
            base = dict(base)
            base.pop("_compiled", None)
        if isinstance(base, dict) and "_julia" in base:
            base = dict(base)
            base.pop("_julia", None)
        self._restore_state(base)
        self._fitted_backend = fitted
        self._ensure_context_parameters()
        # Availability is independent of fitted ownership. Keep installed
        # handles for later checkpoint loads; _fitted_backend guards inference.
        if fitted != "python":
            try:
                native_handle.load_state(native_state.get("state", {}))
            except Exception as exc:
                setattr(self, f"{fitted}_backend", None)
                raise ValueError(f"Failed to restore fitted {fitted} boundary weights") from exc

    # ------------------------------------------------------------------
    # Evaluation and inference
    # ------------------------------------------------------------------
    def evaluate(self, sequences: Sequence[BoundarySequence]) -> Tuple[float, float]:
        self._require_python_head()
        total_loss = 0.0
        total_tp = total_fp = total_fn = 0
        for seq in sequences:
            logits, _ = self._forward_sequence(seq)
            labels = self._labels_to_int(seq.labels)
            loss, _, _, marginals = self._crf_loss(logits, labels)
            total_loss += loss
            preds = self._viterbi(logits)
            tp, fp, fn = self._boundary_confusion(preds, labels)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return float(total_loss / max(1, len(sequences))), float(f1)

    def _boundary_confusion(self, preds: List[int], labels: List[int]) -> Tuple[int, int, int]:
        tp = fp = fn = 0
        for p, y in zip(preds, labels):
            if p == 1 and y == 1:
                tp += 1
            elif p == 1 and y == 0:
                fp += 1
            elif p == 0 and y == 1:
                fn += 1
        return tp, fp, fn

    def _viterbi(self, logits: List[float]) -> List[int]:
        length = len(logits)
        if length == 0:
            return []
        trans = self.transitions
        emit = [[0.0 for _ in range(length)], [logits[i] for i in range(length)]]
        dp = [[float("-inf"), float("-inf")] for _ in range(length)]
        back: List[List[int]] = [[0, 0] for _ in range(length)]
        dp[0][0] = emit[0][0]
        dp[0][1] = emit[1][0]
        for i in range(1, length):
            for state in (0, 1):
                best_score = float("-inf")
                best_prev = 0
                for prev in (0, 1):
                    trans_val = trans[prev][state]
                    score = dp[i - 1][prev] + (float(trans_val) if hasattr(trans_val, "__float") else trans_val)
                    if score > best_score:
                        best_score = score
                        best_prev = prev
                dp[i][state] = best_score + emit[state][i]
                back[i][state] = best_prev
        last_state = 1 if dp[-1][1] > dp[-1][0] else 0
        out = [0 for _ in range(length)]
        out[-1] = last_state
        for i in range(length - 1, 0, -1):
            out[i - 1] = back[i][out[i]]
        return out

    def boundary_probs(self, text: str) -> np.ndarray:
        self._require_fitted_backend()
        fallbacks: List[str] = []
        if self._fitted_backend == "python" or (
            self._fitted_backend is None and (self.lexical_buckets or self._context_ready())
        ):
            if len(text) <= 1:
                self._backend_metadata("python", fallbacks, stage="boundary_probs")
                return np.zeros(0, dtype=float)
            seq = self.build_sequences([text], [[text]])[0]
            logits, _ = self._forward_sequence(seq)
            labels = self._labels_to_int(seq.labels)
            _, _, _, marginals = self._crf_loss(logits, labels)
            probs = [m[1] for m in marginals]
            self._backend_metadata("python", fallbacks, stage="boundary_probs")
            return np.array(probs, dtype=float)
        if self.julia_backend is not None and self._fitted_backend in (None, "julia"):
            try:
                result = self.julia_backend.boundary_probs(text)
                backend_id = f"julia:{self.julia_backend.device}"
                self._backend_metadata(backend_id, fallbacks, stage="boundary_probs")
                return result
            except Exception as exc:
                if self._fitted_backend == "julia":
                    raise RuntimeError("Fitted Julia boundary probability computation failed") from exc
                backend_id = f"julia:{getattr(self.julia_backend, 'device', 'unknown')}"
                fallbacks.append(backend_id)
                _log_backend_exception("boundary_probs", backend_id, exc)
                self.julia_backend = None
        if self.compiled_backend is not None and self._fitted_backend in (None, "compiled"):
            try:
                result = self.compiled_backend.boundary_probs(text)
                backend_id = f"compiled:{self.compiled_backend.device}"
                self._backend_metadata(backend_id, fallbacks, stage="boundary_probs")
                return result
            except Exception as exc:
                if self._fitted_backend == "compiled":
                    raise RuntimeError("Fitted compiled boundary probability computation failed") from exc
                backend_id = f"compiled:{getattr(self.compiled_backend, 'device', 'unknown')}"
                fallbacks.append(backend_id)
                _log_backend_exception("boundary_probs", backend_id, exc)
                self.compiled_backend = None
        if len(text) <= 1:
            self._backend_metadata("python", fallbacks, stage="boundary_probs")
            return np.zeros(0, dtype=float)
        seq = self.build_sequences([text], [[text]])[0]
        logits, _ = self._forward_sequence(seq)
        labels = self._labels_to_int(seq.labels)
        _, _, _, marginals = self._crf_loss(logits, labels)
        probs = [m[1] for m in marginals]
        self._backend_metadata("python", fallbacks, stage="boundary_probs")
        return np.array(probs, dtype=float)

    def _python_logits(self, text: str) -> List[float]:
        self._require_python_head()
        if len(text) <= 1:
            return []
        seq = self.build_sequences([text], [[text]])[0]
        logits, _ = self._forward_sequence(seq)
        return [float(v) for v in logits]

    def _resolve_logits(self, text: str, logits: Optional[Sequence[float]]) -> List[float]:
        self._require_python_head()
        if logits is None:
            return self._python_logits(text)
        if len(logits) != max(0, len(text) - 1):
            raise ValueError("logits must contain one score per character boundary")
        return [float(value) for value in logits]

    def boundary_probs_with_logit_bias(
        self, text: str, logit_bias: float = 0.0, *, logits: Optional[Sequence[float]] = None
    ) -> np.ndarray:
        """Return CRF marginals after applying a constant logit bias.

        This intentionally uses the pure-Python implementation (even if Julia/C++
        backends are available) so the caller can explore policy-conditioned
        shifts without requiring native backends to re-implement the knobs.
        """

        logits = self._resolve_logits(text, logits)
        if not logits:
            return np.zeros(0, dtype=float)
        bias = float(logit_bias)
        logits = [val + bias for val in logits]
        labels = [0 for _ in logits]
        _, _, _, marginals = self._crf_loss(logits, labels)
        probs = [float(m[1]) for m in marginals]
        self._backend_metadata("python", [], stage="boundary_probs_with_logit_bias")
        return np.array(probs, dtype=float)

    def decode(self, text: str) -> Dict[str, object]:
        self._require_fitted_backend()
        fallbacks: List[str] = []
        if not text:
            return {"tokens": [], **self._backend_metadata("python", fallbacks, stage="decode")}
        if self._fitted_backend == "python" or (
            self._fitted_backend is None and (self.lexical_buckets or self._context_ready())
        ):
            seq = self.build_sequences([text], [[text]])[0]
            logits, _ = self._forward_sequence(seq)
            labels = self._viterbi(logits)
            tokens: List[str] = []
            start = 0
            for i, label in enumerate(labels):
                if label == 1:
                    tokens.append(text[start : i + 1])
                    start = i + 1
            tokens.append(text[start:])
            meta = self._backend_metadata("python", fallbacks, stage="decode")
            return {"tokens": tokens, **meta}
        if self.julia_backend is not None and self._fitted_backend in (None, "julia"):
            try:
                tokens = list(self.julia_backend.decode(text))
                backend_id = f"julia:{self.julia_backend.device}"
                meta = self._backend_metadata(backend_id, fallbacks, stage="decode")
                return {"tokens": tokens, **meta}
            except Exception as exc:
                if self._fitted_backend == "julia":
                    raise RuntimeError("Fitted Julia boundary decoding failed") from exc
                backend_id = f"julia:{getattr(self.julia_backend, 'device', 'unknown')}"
                fallbacks.append(backend_id)
                logger.exception(
                    "Julia backend decode failed; trying alternative backend.",
                    extra={"backend": backend_id, "event": "boundary_backend_failure"},
                )
                self.julia_backend = None
        if self.compiled_backend is not None and self._fitted_backend in (None, "compiled"):
            try:
                tokens = list(self.compiled_backend.decode(text))
                backend_id = f"compiled:{self.compiled_backend.device}"
                meta = self._backend_metadata(backend_id, fallbacks, stage="decode")
                return {"tokens": tokens, **meta}
            except Exception as exc:
                if self._fitted_backend == "compiled":
                    raise RuntimeError("Fitted compiled boundary decoding failed") from exc
                backend_id = f"compiled:{getattr(self.compiled_backend, 'device', 'unknown')}"
                fallbacks.append(backend_id)
                logger.exception(
                    "Compiled backend decode failed; reverting to Python implementation.",
                    extra={"backend": backend_id, "event": "boundary_backend_failure"},
                )
                self.compiled_backend = None
        seq = self.build_sequences([text], [[text]])[0]
        logits, _ = self._forward_sequence(seq)
        labels = self._viterbi(logits)
        tokens: List[str] = []
        start = 0
        for i, label in enumerate(labels):
            if label == 1:
                tokens.append(text[start : i + 1])
                start = i + 1
        tokens.append(text[start:])
        meta = self._backend_metadata("python", fallbacks)
        return {"tokens": tokens, **meta}

    def decode_with_logit_bias(
        self, text: str, logit_bias: float = 0.0, *, logits: Optional[Sequence[float]] = None
    ) -> Dict[str, object]:
        """Decode a segmentation with a constant logit bias applied.

        Like :meth:`boundary_probs_with_logit_bias`, this uses the pure-Python CRF
        path to keep behaviour consistent across environments regardless of
        whether optional native backends are installed.
        """

        logits = self._resolve_logits(text, logits)
        if not text:
            return {
                "tokens": [], "logit_bias": float(logit_bias),
                **self._backend_metadata("python", [], stage="decode_with_logit_bias"),
            }
        bias = float(logit_bias)
        logits = [val + bias for val in logits]
        labels = self._viterbi(logits)
        tokens: List[str] = []
        start = 0
        for i, label in enumerate(labels):
            if label == 1:
                tokens.append(text[start : i + 1])
                start = i + 1
        tokens.append(text[start:])
        meta = self._backend_metadata("python", [], stage="decode_with_logit_bias")
        meta["logit_bias"] = bias
        return {"tokens": tokens, **meta}

    def _select_backend_device(self, preference: Optional[str]) -> None:
        if preference is None:
            return
        pref = preference.lower()
        handles: List[Tuple[str, Optional[Any]]] = [
            ("julia", self.julia_backend),
            ("compiled", self.compiled_backend),
        ]
        for _, handle in handles:
            if handle is None:
                continue
            target: Optional[str]
            if pref == "auto":
                target = handle.preferred_device()
            else:
                target = pref
            if target:
                try:
                    if handle.to_device(target):
                        continue
                except Exception as exc:
                    backend_name = getattr(handle, "name", handle.__class__.__name__)
                    logger.warning(
                        "Failed to switch %s backend to device '%s'",
                        backend_name,
                        target,
                        exc_info=True,
                    )
                    continue

    def backend_inventory(self) -> Dict[str, List[str]]:
        inventory: Dict[str, List[str]] = {"python": ["cpu"]}
        if self.compiled_backend is not None:
            try:
                inventory["compiled"] = list(self.compiled_backend.available_devices())
            except Exception as exc:
                logger.warning(
                    "Compiled boundary backend device inventory unavailable; using current device",
                    exc_info=True,
                )
                inventory["compiled"] = [self.compiled_backend.device]
        else:
            devices = list(compiled_backend_devices())
            if devices:
                inventory["compiled"] = devices
        if self.julia_backend is not None:
            try:
                inventory["julia"] = list(self.julia_backend.available_devices())
            except Exception as exc:
                logger.warning(
                    "Julia boundary backend device inventory unavailable; using current device",
                    exc_info=True,
                )
                inventory["julia"] = [self.julia_backend.device]
        else:
            devices = list(julia_backend_devices())
            if devices:
                inventory["julia"] = devices
        if self.encoder_adapter is not None and hasattr(self.encoder_adapter, "device_inventory"):
            try:
                inventory["encoder"] = list(self.encoder_adapter.device_inventory())
            except Exception as exc:
                logger.warning(
                    "Encoder adapter device inventory unavailable", exc_info=True
                )
        return inventory

    def backend_metadata(self) -> Dict[str, object]:
        return {
            "backend_used": self._last_backend_used,
            "backend_fallbacks": list(self._last_backend_fallbacks),
        }
