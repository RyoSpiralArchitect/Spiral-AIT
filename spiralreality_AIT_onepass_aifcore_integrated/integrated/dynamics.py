from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import List, Tuple

from .np_compat import np


@dataclass
class DynamicsConfig:
    hidden_dim: int = 128
    lr: float = 0.01
    epochs: int = 8
    batch_size: int = 16
    max_grad_norm: float | None = 10.0
    min_observations: int = 32
    train_interval: int = 16
    max_buffer: int = 2048


class LatentDynamicsModel:
    """Residual model fitted to explicitly recorded transition observations.

    Prediction is pure. New models optimize mean diagonal-Gaussian NLL with
    positive softplus process variance. Legacy variance links remain available
    for checkpoint inference, but legacy training is not real-observation evidence.
    """

    def __init__(self, latent_dim: int, context_dim: int, seed: int = 0, cfg: DynamicsConfig | None = None):
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.cfg = cfg or DynamicsConfig()
        self._validate_config()
        rng = np.random.default_rng(seed)
        input_dim = latent_dim * 2 + context_dim
        self.W1 = rng.normal(scale=0.2, size=(self.cfg.hidden_dim, input_dim))
        self.b1 = np.zeros(self.cfg.hidden_dim)
        self.W_mu = rng.normal(scale=0.1, size=(latent_dim, self.cfg.hidden_dim))
        self.b_mu = np.zeros(latent_dim)
        self.W_sigma = rng.normal(scale=0.05, size=(latent_dim, self.cfg.hidden_dim))
        self.b_sigma = np.zeros(latent_dim)
        self.buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self.trained_steps = 0
        self.recorded_observations = 0
        self.trained_observations = 0
        self.variance_link = "softplus_gaussian_nll_v2"

    def _validate_config(self):
        for name in ("hidden_dim", "batch_size", "min_observations", "train_interval", "max_buffer"):
            value = getattr(self.cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.cfg.min_observations > self.cfg.max_buffer:
            raise ValueError("min_observations must fit in max_buffer")
        if isinstance(self.cfg.epochs, bool) or not isinstance(self.cfg.epochs, int) or self.cfg.epochs < 0:
            raise ValueError("epochs must be a nonnegative integer")
        if not math.isfinite(self.cfg.lr) or self.cfg.lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if self.cfg.max_grad_norm is not None and (not math.isfinite(self.cfg.max_grad_norm) or self.cfg.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be finite and positive, or None")

    @staticmethod
    def _vector(value, dimension, name):
        result = np.asarray(value, dtype=float)
        if result.shape != (dimension,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector of length {dimension}")
        return result

    def _inputs(self, mu, policy_vec, ctx_vec):
        return np.concatenate((self._vector(mu, self.latent_dim, "mu"),
                               self._vector(policy_vec, self.latent_dim, "policy_vec"),
                               self._vector(ctx_vec, self.context_dim, "ctx_vec")))

    def _forward(self, inputs):
        h = np.tanh(self.W1 @ inputs + self.b1)
        delta = self.W_mu @ h + self.b_mu
        logits = self.W_sigma @ h + self.b_sigma
        if self.variance_link == "legacy_sigmoid_proxy_v1":
            variance = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60))) + 1e-3
        else:
            variance = np.logaddexp(0.0, logits) + 1e-3
        return delta, variance

    def predict(self, mu, policy_vec, ctx_vec, *, prior_covariance=None):
        inputs = self._inputs(mu, policy_vec, ctx_vec)
        delta, variance = self._forward(inputs)
        covariance = np.diag(variance)
        if prior_covariance is not None:
            prior = np.asarray(prior_covariance, dtype=float)
            if prior.shape != (self.latent_dim, self.latent_dim) or not np.all(np.isfinite(prior)):
                raise ValueError("prior_covariance must be a finite square matrix")
            h = np.tanh(self.W1 @ inputs + self.b1)
            jacobian = np.eye(self.latent_dim) + (self.W_mu * (1 - h * h)) @ self.W1[:, :self.latent_dim]
            covariance += jacobian @ prior @ jacobian.T
        return inputs[:self.latent_dim] + delta, (covariance + covariance.T) * 0.5

    def record(self, mu, policy_vec, ctx_vec, target_mu):
        inputs = self._inputs(mu, policy_vec, ctx_vec)
        target = self._vector(target_mu, self.latent_dim, "target_mu")
        self.buffer.append((inputs.copy(), (target - inputs[:self.latent_dim]).copy()))
        self.buffer = self.buffer[-self.cfg.max_buffer:]
        self.recorded_observations += 1

    def should_train(self):
        return (len(self.buffer) >= self.cfg.min_observations
                and (not self.trained_observations
                     or self.recorded_observations - self.trained_observations >= self.cfg.train_interval))

    def _loss_and_gradients(self, batch):
        names = ("W1", "b1", "W_mu", "b_mu", "W_sigma", "b_sigma")
        gradients = {name: np.zeros_like(getattr(self, name)) for name in names}
        loss, count = 0.0, max(1, len(batch))
        for inputs, target_delta in batch:
            h = np.tanh(self.W1 @ inputs + self.b1)
            delta = self.W_mu @ h + self.b_mu
            logits = self.W_sigma @ h + self.b_sigma
            variance = np.logaddexp(0.0, logits) + 1e-3
            diff = delta - target_delta
            loss += 0.5 * float(np.sum(np.log(2 * np.pi * variance) + diff * diff / variance)) / count
            grad_mu = diff / variance / count
            link_derivative = 1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700)))
            grad_sigma = 0.5 * (1.0 / variance - diff * diff / (variance * variance)) * link_derivative / count
            gradients["W_mu"] += np.outer(grad_mu, h)
            gradients["b_mu"] += grad_mu
            gradients["W_sigma"] += np.outer(grad_sigma, h)
            gradients["b_sigma"] += grad_sigma
            grad_h = (self.W_mu.T @ grad_mu + self.W_sigma.T @ grad_sigma) * (1 - h * h)
            gradients["W1"] += np.outer(grad_h, inputs)
            gradients["b1"] += grad_h
        return loss, gradients

    def train(self):
        self._validate_config()
        if len(self.buffer) < self.cfg.min_observations or not self.cfg.epochs or not self.cfg.lr:
            return {"trained": False, "recorded_observations": self.recorded_observations}
        self.variance_link = "softplus_gaussian_nll_v2"
        rng = np.random.default_rng(self.trained_steps + 1)
        before = self._loss_and_gradients(self.buffer)[0]
        for _ in range(self.cfg.epochs):
            order = rng.permutation(len(self.buffer))
            for start in range(0, len(order), self.cfg.batch_size):
                batch = [self.buffer[int(index)] for index in order[start:start + self.cfg.batch_size]]
                _, gradients = self._loss_and_gradients(batch)
                norm = math.sqrt(sum(float(np.sum(value * value)) for value in gradients.values()))
                scale = min(1.0, self.cfg.max_grad_norm / (norm + 1e-12)) if self.cfg.max_grad_norm else 1.0
                for name, gradient in gradients.items():
                    setattr(self, name, getattr(self, name) - self.cfg.lr * scale * gradient)
        self.trained_steps += 1
        self.trained_observations = self.recorded_observations
        return {"trained": True, "loss_before": before, "loss_after": self._loss_and_gradients(self.buffer)[0],
                "recorded_observations": self.recorded_observations}

    def is_ready(self):
        return self.trained_observations >= self.cfg.min_observations and self.variance_link == "softplus_gaussian_nll_v2"

    def export_state(self):
        state = {name: getattr(self, name).tolist() for name in ("W1", "b1", "W_mu", "b_mu", "W_sigma", "b_sigma")}
        state.update(version=2, config=asdict(self.cfg), trained_steps=self.trained_steps,
                     recorded_observations=self.recorded_observations, trained_observations=self.trained_observations,
                     variance_link=self.variance_link,
                     buffer=[[inputs.tolist(), target.tolist()] for inputs, target in self.buffer])
        return state

    def load_state(self, state):
        if state.get("version") not in (None, 2):
            raise ValueError("Unsupported dynamics checkpoint version")
        config = state.get("config")
        if config is None:
            config = asdict(self.cfg)
            legacy_hidden = np.asarray(state["b1"])
            if legacy_hidden.ndim != 1:
                raise ValueError("Invalid legacy dynamics hidden dimension")
            config["hidden_dim"] = len(legacy_hidden)
        cfg = DynamicsConfig(**config)
        old_cfg = self.cfg
        self.cfg = cfg
        try:
            self._validate_config()
            shapes = {"W1": (cfg.hidden_dim, self.latent_dim * 2 + self.context_dim), "b1": (cfg.hidden_dim,),
                      "W_mu": (self.latent_dim, cfg.hidden_dim), "b_mu": (self.latent_dim,),
                      "W_sigma": (self.latent_dim, cfg.hidden_dim), "b_sigma": (self.latent_dim,)}
            values = {name: np.asarray(state[name], dtype=float) for name in shapes}
            if any(value.shape != shapes[name] or not np.all(np.isfinite(value)) for name, value in values.items()):
                raise ValueError("Invalid dynamics weights in checkpoint")
            recorded, trained = state.get("recorded_observations", 0), state.get("trained_observations", 0)
            steps = state.get("trained_steps", 0)
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (recorded, trained, steps)):
                raise ValueError("Dynamics observation counters must be integers")
            if recorded < 0 or steps < 0 or not 0 <= trained <= recorded:
                raise ValueError("Invalid dynamics observation counters")
            link = state.get("variance_link", "legacy_sigmoid_proxy_v1")
            if link not in ("legacy_sigmoid_proxy_v1", "softplus_gaussian_nll_v2"):
                raise ValueError("Unknown dynamics variance link")
            buffer = [(self._vector(pair[0], self.latent_dim * 2 + self.context_dim, "stored inputs").copy(),
                       self._vector(pair[1], self.latent_dim, "stored target").copy()) for pair in state.get("buffer", [])]
            if len(buffer) != min(recorded, cfg.max_buffer):
                raise ValueError("Invalid dynamics replay buffer")
            if link == "softplus_gaussian_nll_v2" and bool(steps) != bool(trained):
                raise ValueError("Dynamics training counters are inconsistent")
        except Exception:
            self.cfg = old_cfg
            raise
        for name, value in values.items():
            setattr(self, name, value.copy())
        self.buffer = buffer
        self.recorded_observations, self.trained_observations = recorded, trained
        self.trained_steps = steps
        self.variance_link = link
