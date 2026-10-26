from __future__ import annotations
import math
from .np_compat import np
from .aif_core import GaussianBelief
from .onepass_ait import OnePassAIT

class AITGWMBridge:
    """Wraps One‑Pass AIT as the dynamics used in AIF planning.
    - ctx_provider: policy‑specific local gates over H to produce a context vector.
    - step_fn: uses AIT.predict_next(mu,Sigma,policy,ctx) and returns (belief_next, o_mu, o_Sigma).
    - r3_provider: reads R3_mix updated by delivered observations.
    - observe_execution: admits delivered targets to real transition learning.
    """
    def __init__(self, ait: OnePassAIT, enc: dict, obs_sigma: float = 0.6):
        if not math.isfinite(obs_sigma) or obs_sigma <= 0:
            raise ValueError("obs_sigma must be finite and positive")
        features = np.asarray(enc["H"], dtype=float)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != ait.latent_dim or not np.all(np.isfinite(features)):
            raise ValueError("Bridge requires a finite nonempty contextual encoding")
        self.ait = ait
        self.enc = enc  # {"H","r2_local","ps","gate_pos"}
        self.obs_sigma = obs_sigma
        self.dim = ait.latent_dim

    def ctx_from_prefix(self, prefix: list[str]) -> np.ndarray:
        if not prefix:
            return np.mean(self.enc["H"], axis=0)
        last = prefix[-1]
        g = self.ait.policy_local_gates(self.enc["H"], self.enc["r2_local"], last)
        H = self.enc["H"]
        weights = g.tolist() if hasattr(g, "tolist") else list(g)
        H_rows = H.tolist() if hasattr(H, "tolist") else list(H)
        if not weights:
            return np.mean(H, axis=0)
        dim = len(H_rows[0])
        accum = [0.0 for _ in range(dim)]
        total = 0.0
        for w, row in zip(weights, H_rows):
            total += w
            for j in range(dim):
                accum[j] += w * row[j]
        if total <= 0.0:
            total = 1.0
        ctx = [val / total for val in accum]
        return np.array(ctx)

    def r3_from_prefix(self, prefix: list[str]) -> float:
        # Only delivered observations advance the phase state. Hypothetical
        # prefix length is not evidence for increased exploration.
        return float(self.ait.R3_mix)

    def step_fn(self, belief: GaussianBelief, action: str, ctx_vec: np.ndarray):
        # Forecast moments only. The execution callback supplies observations.
        mu_next, Sigma_next = self.ait.predict_next(belief.mu, belief.Sigma, action, ctx_vec)
        o_mu = mu_next.copy()
        o_Sigma = (self.obs_sigma**2)*np.eye(self.dim)
        return GaussianBelief(mu_next, Sigma_next), o_mu, o_Sigma

    def observe_execution(self, prior, action, context, observation, posterior):
        return self.ait.observe_transition(prior.mu, action, context, observation.mean,
                                           observation_id=observation.observation_id,
                                           source=observation.source, simulated=observation.simulated)
