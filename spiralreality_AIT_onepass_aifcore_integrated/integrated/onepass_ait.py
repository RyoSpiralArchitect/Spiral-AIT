from __future__ import annotations

import math
import operator
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .np_compat import np
from .aif_core import EFEConfig, EFEEngine, GaussianBelief, QuadraticPreference
from .boundary import BoundaryStudent, StudentTrainingConfig
from .dynamics import LatentDynamicsModel
from .encoder import SpectralTransformerAdapter, ToyTransformerAdapter
from .encoder_backends import ExternalEncoderHandle, load_external_adapter
from .multilingual import (
    build_multilingual_corpus,
    language_histogram,
    language_statistics,
)
from .phase import PhaseBasisLearner
from .streaming_inference import ChunkedStreamingSegmenter
from .utils import is_punct, is_space, seeded_vector, sigmoid, unit


@dataclass
class GateDiagnostics:
    gate_trace: List[float]
    attention_strength: List[float]
    mask_energy: float = 0.0


@dataclass
class SegmentationAIFConfig:
    """Configuration for Active-Inference-style policy selection over segmentation."""

    prior_sigma: float = 0.35
    target_boundary_rate: float = 0.18
    target_punct_boundary_rate: float = 0.85
    target_entropy: float = 0.15

    w_boundary_rate: float = 1.0
    w_punct_boundary_rate: float = 2.0
    w_entropy: float = 1.0

    efe_epistemic_w0: float = 1.0
    efe_epistemic_w1: float = 0.6

    obs_sigma_by_policy: Dict[str, float] = field(
        default_factory=lambda: {
            "ProbeMotivation": 0.12,
            "ProbeReliability": 0.10,
            "SeekEvidence": 0.08,
            "DecideNow": 0.20,
        }
    )
    logit_bias_by_policy: Dict[str, float] = field(
        default_factory=lambda: {
            "ProbeMotivation": 0.0,
            "ProbeReliability": -0.35,
            "SeekEvidence": -0.10,
            "DecideNow": 0.35,
        }
    )


class OnePassAIT:
    def __init__(
        self,
        latent_dim: int = 64,
        seed: int = 4242,
        *,
        encoder_layers: int = 6,
        encoder_heads: int = 4,
        encode_cache_max_entries: int = 16,
        encode_cache_max_bytes: int = 32 * 1024 * 1024,
    ):
        for name, limit in (("encode_cache_max_entries", encode_cache_max_entries),
                            ("encode_cache_max_bytes", encode_cache_max_bytes)):
            try:
                validated = operator.index(limit)
            except TypeError as exc:
                raise ValueError(f"{name} must be a non-negative integer") from exc
            if isinstance(limit, bool) or validated < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            setattr(self, f"_{name}", validated)
        self.latent_dim = latent_dim
        self.rng = np.random.default_rng(seed)
        self.policies = ["ProbeMotivation", "ProbeReliability", "SeekEvidence", "DecideNow"]
        self.policy_vecs = {p: seeded_vector(p, latent_dim) for p in self.policies}
        self.goal_vec = unit(self.rng.normal(size=latent_dim))
        self.phase = PhaseBasisLearner(dim=latent_dim)
        self.student = BoundaryStudent(self.phase, seed=seed)
        self.encoder_handle: Optional[ExternalEncoderHandle] = load_external_adapter(
            latent_dim, int(encoder_layers), seed
        )
        if self.encoder_handle is not None:
            self.encoder = self.encoder_handle.impl
        else:
            self.encoder = SpectralTransformerAdapter(
                d_model=latent_dim,
                n_layers=int(encoder_layers),
                n_heads=int(encoder_heads),
                seed=seed,
            )
        self.student.bind_encoder(self.encoder)
        self.dynamics = LatentDynamicsModel(latent_dim, latent_dim, seed=seed)
        self.beta_ewma = 0.2
        self.gate_a0, self.gate_a1 = 1.0, 1.3
        self.mu = np.zeros(latent_dim)
        self.Sigma = np.eye(latent_dim)
        self.R3_mix = 0.0
        self.R2_time = 0.0
        self._phi_hist: Dict[str, List[float]] = {"AB": [], "BC": [], "CA": []}
        self.last_gate_trace: List[float] = []
        self.last_attention: List[np.ndarray] = []
        self.last_gate_mask: Optional[np.ndarray] = None
        self._encode_cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()
        self._last_segment_metadata: Dict[str, object] = {}

    def train_student(
        self,
        texts: Optional[Sequence[str]] = None,
        segments: Optional[Sequence[Sequence[str]]] = None,
        cfg: Optional[StudentTrainingConfig] = None,
        languages: Optional[Sequence[str]] = None,
        include_reflective: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        """Train the boundary student on the provided or assembled corpus."""

        dataset_texts: List[str] = []
        dataset_segments: List[List[str]] = []
        dataset_tags: List[str] = []

        if texts is not None:
            dataset_texts = list(texts)
            if segments is not None:
                dataset_segments = [list(seg) for seg in segments]
            else:
                from .corpus import teacher_segments

                dataset_segments = teacher_segments(dataset_texts)
            dataset_tags.extend(["custom"] * len(dataset_texts))
        elif segments is not None:
            raise ValueError("segments were provided without matching texts")

        if languages is not None:
            ml_seed = seed if seed is not None else self._next_seed()
            ml_texts, ml_segments, ml_tags = build_multilingual_corpus(
                languages=languages,
                include_reflective=include_reflective,
                shuffle=shuffle,
                seed=ml_seed,
            )
            if dataset_texts:
                dataset_texts.extend(ml_texts)
                dataset_segments.extend(ml_segments)
                dataset_tags.extend(ml_tags)
            else:
                dataset_texts = ml_texts
                dataset_segments = ml_segments
                dataset_tags = ml_tags

        if not dataset_texts and texts is None and languages is None:
            from .corpus import TRAIN_TEXTS, teacher_segments

            dataset_texts = list(TRAIN_TEXTS)
            dataset_segments = teacher_segments(dataset_texts)
            dataset_tags = ["reflective"] * len(dataset_texts)

        if not dataset_texts:
            raise ValueError("Training requires at least one text")
        summary = self.student.train(dataset_texts, dataset_segments, cfg=cfg)
        self._encode_cache.clear()
        if isinstance(summary, dict):
            summary = dict(summary)
            summary.setdefault("dataset_size", len(dataset_texts))
            summary.setdefault("dataset_tags", dataset_tags)
            summary.setdefault("dataset_languages", language_histogram(dataset_tags))
            if dataset_texts:
                summary.setdefault(
                    "dataset_language_stats",
                    language_statistics(dataset_texts, dataset_segments, dataset_tags),
                )
            summary.setdefault("dataset_texts", dataset_texts)
            summary.setdefault("dataset_segments", dataset_segments)
            summary.setdefault("encoder_backend", self.encoder_backend_name())
            summary.setdefault(
                "backend_used",
                summary.get("backend", self.student.backend_metadata().get("backend_used")),
            )
            if hasattr(self.encoder, "device_inventory"):
                try:
                    summary.setdefault(
                        "encoder_devices",
                        list(getattr(self.encoder, "device_inventory")()),
                    )
                except Exception:
                    pass
            summary.setdefault("available_devices", self.student.backend_inventory())
        else:
            summary = {
                "result": summary,
                "dataset_size": len(dataset_texts),
                "dataset_tags": dataset_tags,
                "dataset_languages": language_histogram(dataset_tags),
                "dataset_language_stats": language_statistics(
                    dataset_texts, dataset_segments, dataset_tags
                )
                if dataset_texts
                else {},
                "dataset_texts": dataset_texts,
                "dataset_segments": dataset_segments,
                "encoder_backend": self.encoder_backend_name(),
                "encoder_devices": list(getattr(self.encoder, "device_inventory")())
                if hasattr(self.encoder, "device_inventory")
                else [getattr(self.encoder, "device", "cpu")],
                "available_devices": self.student.backend_inventory(),
                "backend_used": self.student.backend_metadata().get("backend_used"),
            }
        return summary

    def encoder_backend_name(self) -> str:
        if self.encoder_handle is not None:
            return f"{self.encoder_handle.backend}:{self.encoder_handle.device}"
        backend = getattr(self.encoder, "backend", "numpy")
        device = getattr(self.encoder, "device", "cpu")
        return f"{backend}:{device}"

    def _next_seed(self) -> int:
        if hasattr(self.rng, "integers"):
            return int(self.rng.integers(2**31 - 1))
        if hasattr(self.rng, "randint"):
            return int(self.rng.randint(0, 2**31 - 1))
        if hasattr(self.rng, "random"):
            return int(self.rng.random() * (2**31 - 1))
        return 4242

    def _char_embs(self, text: str) -> np.ndarray:
        embs = [seeded_vector(f"char::{c}", self.latent_dim) for c in text]
        return np.stack(embs, axis=0)

    def encode(self, text: str) -> Dict[str, np.ndarray]:
        if not text:
            self.last_gate_trace = []
            self.last_attention = []
            self.last_gate_mask = None
            result = {
                "H": np.zeros((0, self.latent_dim)),
                "r2_local": np.zeros(0),
                "ps": np.zeros(0),
                "gate_pos": np.zeros(0),
                "phase_local": np.zeros((0, 3)),
                "gate_mask": np.zeros((0, 0)),
            }
            result.update(self.student.backend_metadata())
            return result
        if text in self._encode_cache:
            cached = self._encode_cache[text]
            self._encode_cache.move_to_end(text)
            result = {
                key: value.copy() for key, value in cached.items()
                if not key.startswith("_attention_")
            }
            self.last_gate_trace = result["gate_pos"].tolist()
            self.last_gate_mask = result["gate_mask"].copy()
            self.last_attention = [
                value.copy() for key, value in cached.items()
                if key.startswith("_attention_")
            ]
            result.update(self.student.backend_metadata())
            return result
        chars = list(text)
        X = self._char_embs(text)
        ps = self.student.boundary_probs(text)
        curvature = self.phase.curvature(text)
        curv_list = curvature.to_list() if hasattr(curvature, "to_list") else list(curvature)
        char_signal = [0.0 for _ in chars]
        for i in range(len(chars) - 1):
            val = float(ps[i]) if i < len(ps) else 0.0
            char_signal[i] = max(char_signal[i], val)
            char_signal[i + 1] = max(char_signal[i + 1], val)
        gate_from_curv = [sigmoid(v) for v in curv_list]
        gate_pos = np.array([
            0.55 * char_signal[i] + 0.45 * gate_from_curv[i] for i in range(len(chars))
        ], dtype=float)
        phase_local = self.phase.local_features(text)
        phase_bias = self._phase_positional(phase_local)
        X_phase = X + phase_bias
        if len(gate_pos):
            gate_list = gate_pos.tolist() if hasattr(gate_pos, "tolist") else list(gate_pos)
            gate_mask = np.array(
                [[min(gi, gj) for gj in gate_list] for gi in gate_list], dtype=float
            )
        else:
            gate_mask = np.zeros((0, 0))
        H = self.encoder.forward(X_phase, gate_pos, gate_mask=gate_mask)
        self.last_gate_trace = gate_pos.tolist()
        self.last_attention = [np.array(attn).copy() for attn in self.encoder.last_attn]
        self.last_gate_mask = gate_mask.copy()
        result = {
            "H": H,
            "r2_local": curvature,
            "ps": ps,
            "gate_pos": gate_pos,
            "phase_local": phase_local,
            "gate_mask": gate_mask,
        }
        self._cache_encoding(text, result)
        result.update(self.student.backend_metadata())
        return result

    def _cache_encoding(self, text: str, result: Dict[str, np.ndarray]) -> None:
        if not self._encode_cache_max_entries or not self._encode_cache_max_bytes:
            return
        arrays = dict(result)
        arrays.update({f"_attention_{index}": attn for index, attn in enumerate(self.last_attention)})
        entry_bytes = sum(int(array.nbytes) for array in arrays.values())
        # Decide before copying large attention matrices. Oversized results
        # remain usable, but do not displace reusable smaller cache entries.
        if entry_bytes > self._encode_cache_max_bytes:
            return
        retained_bytes = sum(int(array.nbytes) for entry in self._encode_cache.values() for array in entry.values())
        while self._encode_cache and (
            len(self._encode_cache) >= self._encode_cache_max_entries
            or retained_bytes + entry_bytes > self._encode_cache_max_bytes
        ):
            _, evicted = self._encode_cache.popitem(last=False)
            retained_bytes -= sum(int(array.nbytes) for array in evicted.values())
        self._encode_cache[text] = {key: array.copy() for key, array in arrays.items()}

    def _phase_positional(self, phase_local: np.ndarray) -> np.ndarray:
        if hasattr(phase_local, "tolist"):
            phase_rows = phase_local.tolist()
        elif hasattr(phase_local, "to_list"):
            phase_rows = phase_local.to_list()
        else:
            phase_rows = list(phase_local)
        if not phase_rows:
            return np.zeros((0, self.latent_dim), dtype=float)
        seq_len = len(phase_rows)
        feat_dim = len(phase_rows[0]) if phase_rows and hasattr(phase_rows[0], "__len__") else 0
        proj_list = [[0.0 for _ in range(self.latent_dim)] for _ in range(seq_len)]
        freqs = [0.75 + 0.5 * idx for idx in range(max(1, feat_dim))]
        for i in range(seq_len):
            offset = 0
            row = phase_rows[i]
            if hasattr(row, "__iter__") and not isinstance(row, (int, float)):
                iterable_row = list(row)
            else:
                iterable_row = [float(row)]
            for feat_idx, val in enumerate(iterable_row):
                freq = freqs[feat_idx]
                if offset < self.latent_dim:
                    proj_list[i][offset] = math.sin(freq * val)
                if offset + 1 < self.latent_dim:
                    proj_list[i][offset + 1] = math.cos(freq * val)
                offset += 2
                if offset >= self.latent_dim:
                    break
        proj_arr = np.array(proj_list, dtype=float)
        return proj_arr * 0.1

    def policy_local_gates(self, H: np.ndarray, r2_local: np.ndarray, policy: str) -> np.ndarray:
        pv = unit(self.policy_vecs[policy])
        rows = H.to_list() if hasattr(H, "to_list") else H.tolist()
        aligns = np.array([float(np.dot(unit(row), pv)) for row in rows])
        r2_vals = r2_local.to_list() if hasattr(r2_local, "to_list") else list(r2_local)
        alpha_loc = self.gate_a0 + self.gate_a1 * np.clip(r2_vals, -5.0, 5.0)
        g = 1.0 / (1.0 + np.exp(-alpha_loc * aligns))
        return g

    def predict_next(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        policy: str,
        ctx_vec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        policy_vec = unit(self.policy_vecs[policy])
        teacher_delta = (
            0.22 * policy_vec
            + 0.12 * self.goal_vec * (1.0 / (1.0 + np.linalg.norm(mu)))
            + 0.15 * unit(ctx_vec)
        )
        teacher_mu = mu + teacher_delta
        teacher_Sigma = 0.9 * Sigma + 0.05 * np.eye(self.latent_dim)
        self.dynamics.record(mu, policy_vec, ctx_vec, teacher_mu)
        if len(self.dynamics.buffer) >= 32:
            self.dynamics.train()
        if self.dynamics.is_ready():
            return self.dynamics.predict(mu, policy_vec, ctx_vec)
        return teacher_mu, teacher_Sigma

    def update_global_phases(self, mu: np.ndarray) -> None:
        for key in ("AB", "BC", "CA"):
            plane = self.phase.basis[key]
            x = float(np.dot(plane[0], mu))
            y = float(np.dot(plane[1], mu))
            p = math.atan2(y, x)
            hist = self._phi_hist[key]
            if not hist:
                hist.append(p)
            else:
                last = hist[-1]
                d = p - last
                d -= 2 * math.pi * round(d / (2 * math.pi))
                hist.append(last + d)
            self._phi_hist[key] = hist[-6:]
        dv = {}
        for key in ("AB", "BC", "CA"):
            hist = self._phi_hist[key]
            if len(hist) < 2:
                dv[key] = 0.0
            else:
                dv[key] = hist[-1] - hist[-2]
        chi = math.copysign(1.0, dv.get("AB", 1.0)) * math.copysign(1.0, dv.get("BC", 1.0)) * math.copysign(1.0, dv.get("CA", 1.0))
        chi = 1.0 if chi > 0 else -1.0
        r = math.sqrt(sum(v * v for v in dv.values()) / max(1, len(dv))) * max(0.0, chi)
        self.R3_mix = (1 - self.beta_ewma) * self.R3_mix + self.beta_ewma * min(5.0, r)
        acc = 0.0
        for key in ("AB", "BC", "CA"):
            hist = self._phi_hist[key]
            if len(hist) >= 3:
                acc += abs(hist[-1] - 2 * hist[-2] + hist[-3])
        curv = acc / 3.0 if acc else 0.0
        self.R2_time = (1 - self.beta_ewma) * self.R2_time + self.beta_ewma * min(5.0, curv)

    def _boundary_entropy_mean(self, probs: np.ndarray) -> float:
        if probs is None:
            return 0.0
        arr = np.asarray(probs, dtype=float).reshape(-1)
        if arr.size == 0:
            return 0.0
        p = np.clip(arr, 1e-6, 1.0 - 1e-6)
        ent = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
        return float(ent.mean()) if ent.size else 0.0

    def _segmentation_stats(self, text: str, probs: np.ndarray) -> np.ndarray:
        """Return a compact observation vector for AIF policy selection."""

        arr = np.asarray(probs, dtype=float).reshape(-1)
        if arr.size == 0 or len(text) <= 1:
            return np.array([0.0, 0.0, 0.0], dtype=float)

        boundary_rate = float(arr.mean())

        punct_mask = np.zeros(arr.shape[0], dtype=bool)
        for i in range(arr.shape[0]):
            left = text[i]
            right = text[i + 1]
            if is_space(left) or is_space(right) or is_punct(left) or is_punct(right):
                punct_mask[i] = True

        punct_boundary_rate = (
            float(arr[punct_mask].mean()) if punct_mask.any() else 0.0
        )
        entropy = self._boundary_entropy_mean(arr)
        return np.array([boundary_rate, punct_boundary_rate, entropy], dtype=float)

    def _r3_mix_from_boundary_probs(self, probs: np.ndarray) -> float:
        ent = self._boundary_entropy_mean(probs)
        ent_norm = 0.0 if ent <= 0.0 else min(1.0, ent / math.log(2.0))
        # Map mean boundary entropy to a small exploration signal.
        return float((ent_norm - 0.5) * 6.0)

    def select_policy_aif(self, text: str, cfg: Optional[SegmentationAIFConfig] = None) -> Dict[str, object]:
        """Select a segmentation policy by minimizing expected free energy (EFE)."""

        logits = self.student._python_logits(text)
        return self._select_policy_from_logits(text, logits, cfg)

    def _select_policy_from_logits(
        self, text: str, logits: Sequence[float], cfg: Optional[SegmentationAIFConfig]
    ) -> Dict[str, object]:
        """Reuse one feature/encoder pass for all policy-conditioned CRF paths."""

        cfg = cfg or SegmentationAIFConfig()
        if not text or len(text) <= 1:
            return {"chosen_policy": "DecideNow", "candidates": [], "r3_mix": 0.0}

        base_probs = self.student.boundary_probs_with_logit_bias(text, 0.0, logits=logits)
        r3_mix = self._r3_mix_from_boundary_probs(base_probs)
        stats_dim = 3
        prior = GaussianBelief(mu=np.zeros(stats_dim), Sigma=(cfg.prior_sigma**2) * np.eye(stats_dim))

        target = np.array(
            [cfg.target_boundary_rate, cfg.target_punct_boundary_rate, cfg.target_entropy],
            dtype=float,
        )
        W = np.diag([cfg.w_boundary_rate, cfg.w_punct_boundary_rate, cfg.w_entropy]).astype(float)
        pref = QuadraticPreference(o_star=target, W=W)
        efe_cfg = EFEConfig(
            epistemic_weight_base=cfg.efe_epistemic_w0,
            epistemic_weight_gain=cfg.efe_epistemic_w1,
        )
        efe = EFEEngine(pref, cfg=efe_cfg)

        candidates: List[Dict[str, object]] = []
        for policy in self.policies:
            logit_bias = float(cfg.logit_bias_by_policy.get(policy, 0.0))
            obs_sigma = float(cfg.obs_sigma_by_policy.get(policy, 0.12))
            probs = base_probs if abs(logit_bias) < 1e-12 else self.student.boundary_probs_with_logit_bias(text, logit_bias, logits=logits)
            stats = self._segmentation_stats(text, probs)
            o_Sigma = (obs_sigma**2) * np.eye(stats_dim)
            post = prior.merge_with_observation(stats, o_Sigma)
            term = efe.decompose_step(prior, stats, o_Sigma, post, r3_mix=r3_mix)
            candidates.append(
                {
                    "policy": policy,
                    "logit_bias": logit_bias,
                    "obs_sigma": obs_sigma,
                    "stats": stats.tolist(),
                    **term,
                }
            )

        chosen = min(candidates, key=lambda row: float(row["total"])) if candidates else None
        return {
            "chosen_policy": chosen["policy"] if chosen is not None else None,
            "r3_mix": float(r3_mix),
            "baseline_stats": self._segmentation_stats(text, base_probs).tolist(),
            "preference": {
                "target": target.tolist(),
                "weights": [cfg.w_boundary_rate, cfg.w_punct_boundary_rate, cfg.w_entropy],
            },
            "candidates": candidates,
        }

    def segment_text(
        self,
        text: str,
        return_metadata: bool = False,
        *,
        use_aif: bool = False,
        aif_cfg: Optional[SegmentationAIFConfig] = None,
        include_confidence: bool = False,
    ):
        """Segment text, optionally returning spans and CRF marginal confidence.

        ``include_confidence`` returns a metadata dictionary even when
        ``return_metadata`` is false. Marginals describe model uncertainty;
        they are not externally calibrated correctness probabilities. This
        option uses the Python CRF so scores and decoded tokens share weights.
        """
        logits = self.student._python_logits(text) if use_aif or include_confidence else None
        extra: Dict[str, object] = {}
        logit_bias = 0.0
        if use_aif:
            assert logits is not None
            selection = self._select_policy_from_logits(text, logits, aif_cfg)
            chosen_policy = selection.get("chosen_policy") or "ProbeMotivation"
            cfg = aif_cfg or SegmentationAIFConfig()
            logit_bias = float(cfg.logit_bias_by_policy.get(str(chosen_policy), 0.0))
            extra = {"chosen_policy": chosen_policy, "aif": selection}

        if logits is not None:
            result = self.student.decode_with_logit_bias(text, logit_bias=logit_bias, logits=logits)
        else:
            result = self.student.decode(text)

        if isinstance(result, dict) and "tokens" in result:
            metadata = {k: v for k, v in result.items() if k != "tokens"}
            metadata.update(extra)
            if include_confidence:
                assert logits is not None
                probs = self.student.boundary_probs_with_logit_bias(text, logit_bias, logits=logits)
                metadata["boundary_probabilities"] = probs.tolist()
                metadata["boundary_entropy"] = self._boundary_entropy_mean(probs)
                spans = []
                start = 0
                for token in result["tokens"]:
                    end = start + len(token)
                    spans.append({
                        "start": start, "end": end,
                        "boundary_probability": float(probs[end - 1]) if end < len(text) else None,
                    })
                    start = end
                metadata["spans"] = spans
            self._last_segment_metadata = metadata
            if return_metadata or include_confidence:
                return {"tokens": result["tokens"], **metadata}
            return result["tokens"]
        self._last_segment_metadata = extra
        return result

    def streaming_segmenter(
        self,
        *,
        max_window_chars: int = 512,
        lookahead_chars: int = 64,
        context_chars: int = 128,
        hard_split: bool = True,
        use_aif: bool = False,
        aif_cfg: Optional[SegmentationAIFConfig] = None,
        min_boundary_confidence: float = 0.0,
    ) -> ChunkedStreamingSegmenter:
        def segmenter(window: str):
            return self.segment_text(
                window, use_aif=use_aif, aif_cfg=aif_cfg,
                include_confidence=min_boundary_confidence > 0.0,
            )

        return ChunkedStreamingSegmenter(
            segmenter,
            max_window_chars=max_window_chars,
            lookahead_chars=lookahead_chars,
            context_chars=context_chars,
            hard_split=hard_split,
            min_boundary_confidence=min_boundary_confidence,
        )

    def gate_diagnostics(self) -> GateDiagnostics:
        strengths = []
        for attn in self.last_attention:
            if hasattr(attn, "tolist"):
                matrix = attn.tolist()
            elif hasattr(attn, "to_list"):
                matrix = attn.to_list()
            else:
                matrix = list(attn)
            strengths.append(float(max(sum(row) for row in matrix)))
        mask_energy = 0.0
        if self.last_gate_mask is not None:
            if hasattr(self.last_gate_mask, "tolist"):
                matrix = self.last_gate_mask.tolist()
            elif hasattr(self.last_gate_mask, "to_list"):
                matrix = self.last_gate_mask.to_list()
            else:
                matrix = self.last_gate_mask
            if matrix and hasattr(matrix, "__iter__") and not isinstance(matrix[0], (int, float)):
                flat = [float(val) for row in matrix for val in row]
            else:
                flat = [float(val) for val in matrix]
            if flat:
                mask_energy = sum(flat) / len(flat)
        return GateDiagnostics(
            gate_trace=self.last_gate_trace[:],
            attention_strength=strengths,
            mask_energy=mask_energy,
        )

    def state_dict(self) -> Dict[str, object]:
        encoder_state = (
            self.encoder_handle.export_state()
            if self.encoder_handle is not None
            else self.encoder.export_state()
        )
        return {
            "latent_dim": self.latent_dim,
            "goal_vec": self.goal_vec.tolist(),
            "policy_vecs": {k: v.tolist() for k, v in self.policy_vecs.items()},
            "phase": self.phase.export_state().basis,
            "student": self.student.export_state(),
            "encoder": encoder_state,
            "dynamics": self.dynamics.export_state(),
            "phi_hist": self._phi_hist,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state.get("latent_dim") != self.latent_dim:
            raise ValueError("Latent dimension mismatch in checkpoint")
        # The phase is shared with the student. Reject ambiguous ownership or
        # unavailable fitted backends before changing any outer model state.
        self.student.validate_state_ownership(state["student"])
        self.goal_vec = np.array(state["goal_vec"], dtype=float)
        self.policy_vecs = {k: np.array(v, dtype=float) for k, v in state["policy_vecs"].items()}
        phase_state = self.phase.export_state()
        phase_state.basis = state["phase"]
        self.phase.load_state(phase_state)
        self.student.load_state(state["student"])
        if self.encoder_handle is not None:
            self.encoder_handle.load_state(state["encoder"])
        else:
            self.encoder.load_state(state["encoder"])
        self.dynamics.load_state(state["dynamics"])
        self._phi_hist = {k: list(state.get("phi_hist", {}).get(k, [])) for k in ("AB", "BC", "CA")}
        self._encode_cache.clear()
        self._last_segment_metadata = {}
        self.last_gate_trace = []
        self.last_attention = []
        self.last_gate_mask = None
