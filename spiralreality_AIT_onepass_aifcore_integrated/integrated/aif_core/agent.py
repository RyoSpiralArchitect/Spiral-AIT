from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from ..np_compat import np
from .beliefs import GaussianBelief
from .preferences import QuadraticPreference
from .efe import EFEEngine, EFEConfig
from .world_model import ActionSpace
from .planning import CEMPlanner, SimpleTreePlanner, RolloutResult, checked_belief, checked_observation, forecast_step


@dataclass
class AgentConfig:
    horizon: int = 3
    planner: str = "cem"
    cem_K: int = 48
    cem_elite_frac: float = 0.25
    cem_iters: int = 2
    tree_branch_limit: int = 4
    efe_epistemic_w0: float = 1.0
    efe_epistemic_w1: float = 0.6
    obs_sigma: float = 0.6


@dataclass(frozen=True)
class ExecutionObservation:
    """A delivered observation, never a forecast returned by the planner."""
    observation_id: int
    mean: np.ndarray
    covariance: np.ndarray
    source: str
    simulated: bool = False


class ActiveInferenceAgent:
    def __init__(self, dim: int, action_space: ActionSpace, goal_vec: np.ndarray,
                 step_fn: Callable, ctx_provider: Callable, r3_provider: Callable,
                 cfg: AgentConfig | None = None, *, observation_fn: Callable | None = None,
                 on_observation: Callable | None = None):
        self.dim = dim
        self.belief = GaussianBelief(mu=np.zeros(dim), Sigma=np.eye(dim))
        self.action_space = action_space
        self.pref = QuadraticPreference(o_star=goal_vec, W=np.eye(dim))
        self.cfg = cfg or AgentConfig()
        efe_cfg = EFEConfig(epistemic_weight_base=self.cfg.efe_epistemic_w0,
                            epistemic_weight_gain=self.cfg.efe_epistemic_w1)
        self.efe = EFEEngine(self.pref, cfg=efe_cfg)
        if self.cfg.planner == "cem":
            self.planner = CEMPlanner(self.efe, action_space, horizon=self.cfg.horizon,
                                      K=self.cfg.cem_K, elite_frac=self.cfg.cem_elite_frac,
                                      iters=self.cfg.cem_iters)
        elif self.cfg.planner == "tree":
            self.planner = SimpleTreePlanner(self.efe, action_space, horizon=self.cfg.horizon,
                                             branch_limit=self.cfg.tree_branch_limit)
        else:
            raise ValueError("planner must be 'cem' or 'tree'")
        self.step_fn = step_fn
        self.ctx_provider = ctx_provider
        self.r3_provider = r3_provider
        self.obs_sigma = self.cfg.obs_sigma
        self.observation_fn = observation_fn
        self.on_observation = on_observation
        self.last_observation_id = 0
        self._pending_execution = None

    def plan(self) -> RolloutResult:
        """Evaluate hypothetical futures without executing an observation callback."""
        return self.planner.plan(self.belief, self.step_fn, self.ctx_provider, self.r3_provider)

    def _checked_delivery(self, observation, *, after=None):
        if not isinstance(observation, ExecutionObservation):
            raise ValueError("An explicit ExecutionObservation is required")
        identifier = observation.observation_id
        previous = self.last_observation_id if after is None else after
        if isinstance(identifier, bool) or not isinstance(identifier, (int, np.integer)) or identifier <= previous:
            raise ValueError("observation_id must strictly increase; duplicate or out-of-order delivery")
        if not isinstance(observation.source, str) or not observation.source.strip():
            raise ValueError("An explicit observation source is required")
        if not isinstance(observation.simulated, bool):
            raise ValueError("simulated must be a boolean")
        mean, covariance = checked_observation(observation.mean, observation.covariance, self.dim)
        return ExecutionObservation(int(identifier), mean.copy(), covariance.copy(), observation.source, observation.simulated)

    @property
    def pending_action(self):
        """The execution that must be retried after a delivered-data failure."""
        return self._pending_execution["action"] if self._pending_execution is not None else None

    @property
    def pending_observation(self):
        if self._pending_execution is None:
            return None
        observation = self._pending_execution["observation"]
        return ExecutionObservation(observation.observation_id, observation.mean.copy(), observation.covariance.copy(),
                                     observation.source, observation.simulated)

    @staticmethod
    def _same_belief(left, right):
        return np.array_equal(left.mu, right.mu) and np.array_equal(left.Sigma, right.Sigma)

    @staticmethod
    def _same_observation(left, right):
        return (left.observation_id == right.observation_id and left.source == right.source
                and left.simulated == right.simulated and np.array_equal(left.mean, right.mean)
                and np.array_equal(left.covariance, right.covariance))

    def act_and_update(self, action: str, *, observation: ExecutionObservation | None = None) -> dict:
        """Execute/deliver once and install the observation-conditioned posterior.

        ``observation_fn(prior, action, context, predicted_next)`` is an explicit
        environment callback, invoked only here. The optional ``on_observation``
        hook receives prior/action/context/delivery/posterior for online learning.
        Simulated environments must label their deliveries accordingly.
        A delivered observation is retained if assimilation or its learning hook
        fails. Retrying the same action reuses that receipt and its original
        forecast/context; another action or replacement receipt is rejected.
        """
        if action not in self.action_space.all_actions():
            raise ValueError("Unknown action")
        pending = self._pending_execution
        if pending is not None:
            if action != pending["action"]:
                raise ValueError("A pending execution observation must be processed before another action")
            if self.last_observation_id != pending["prior_observation_id"] or not self._same_belief(self.belief, pending["prior"]):
                raise ValueError("Agent belief changed while an execution observation was pending")
            if observation is not None:
                supplied = self._checked_delivery(observation)
                if not self._same_observation(supplied, pending["observation"]):
                    raise ValueError("A different observation cannot replace the pending execution receipt")
            prior, context, predicted = pending["prior"], pending["context"], pending["predicted"]
            observation = pending["observation"]
        else:
            if observation is None and self.observation_fn is None:
                raise ValueError("Execution requires a delivered observation or an observation_fn; forecasts are not observations")
            if observation is not None:
                observation = self._checked_delivery(observation)
            prior = checked_belief(self.belief, self.dim)
            context = np.asarray(self.ctx_provider([action]), dtype=float).copy()
            if context.ndim != 1 or not np.all(np.isfinite(context)):
                raise ValueError("Context must be a finite vector")
            predicted, _, _ = forecast_step(self.step_fn, prior, action, context)
            if observation is None:
                observation = self._checked_delivery(self.observation_fn(prior.copy(), action, context.copy(), predicted.copy()))
            self._pending_execution = {"action": action, "prior": prior.copy(), "prior_observation_id": self.last_observation_id,
                                       "context": context.copy(), "predicted": predicted.copy(), "observation": observation}
        prior_entropy = prior.entropy()
        posterior = checked_belief(predicted.merge_with_observation(observation.mean, observation.covariance), self.dim)
        innovation = observation.mean - predicted.mu
        observation_predictive_covariance = predicted.Sigma + observation.covariance
        surprise = float(innovation @ np.linalg.solve(observation_predictive_covariance, innovation))
        delta = posterior.mu - predicted.mu
        _, prior_logdet = np.linalg.slogdet(predicted.Sigma + 1e-12 * np.eye(self.dim))
        _, posterior_logdet = np.linalg.slogdet(posterior.Sigma + 1e-12 * np.eye(self.dim))
        inverse = np.linalg.inv(predicted.Sigma + 1e-12 * np.eye(self.dim))
        information_gain = 0.5 * (np.trace(inverse @ posterior.Sigma) + delta @ inverse @ delta
                                  - self.dim + prior_logdet - posterior_logdet)
        result = {"observation_id": observation.observation_id, "source": observation.source,
                  "simulated": observation.simulated, "obs_mu_norm": float(np.linalg.norm(observation.mean)),
                  "prior_H": float(prior_entropy), "predicted_H": float(predicted.entropy()),
                  "post_H": float(posterior.entropy()), "innovation_squared": surprise,
                  "information_gain_nats": max(0.0, float(information_gain))}
        if self.on_observation is not None:
            result["learning"] = self.on_observation(prior.copy(), action, context.copy(), observation, posterior.copy())
        self.belief = posterior
        self.last_observation_id = observation.observation_id
        self._pending_execution = None
        return result

    def state_dict(self):
        def belief_state(belief):
            return {"mean": belief.mu.tolist(), "covariance": belief.Sigma.tolist()}
        pending = None
        if self._pending_execution is not None:
            retained = self._pending_execution
            observation = retained["observation"]
            pending = {"action": retained["action"], "prior": belief_state(retained["prior"]),
                       "prior_observation_id": retained["prior_observation_id"], "context": retained["context"].tolist(),
                       "predicted": belief_state(retained["predicted"]),
                       "observation": {"observation_id": observation.observation_id, "mean": observation.mean.tolist(),
                                       "covariance": observation.covariance.tolist(), "source": observation.source,
                                       "simulated": observation.simulated}}
        return {"version": 2, "belief": belief_state(self.belief),
                "last_observation_id": self.last_observation_id, "pending_execution": pending}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or isinstance(state.get("version"), bool)
                or not isinstance(state.get("version"), int) or state.get("version") not in (1, 2)):
            raise ValueError("Unsupported agent checkpoint version")
        expected = {"version", "belief", "last_observation_id"}
        if state["version"] == 2:
            expected.add("pending_execution")
        if set(state) != expected:
            raise ValueError("Invalid agent checkpoint schema")
        identifier = state["last_observation_id"]
        if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier < 0:
            raise ValueError("Invalid observation ID in agent checkpoint")
        belief = checked_belief(GaussianBelief(state["belief"]["mean"], state["belief"]["covariance"]), self.dim)
        pending = state.get("pending_execution")
        if pending is not None:
            required = {"action", "prior", "prior_observation_id", "context", "predicted", "observation"}
            if not isinstance(pending, dict) or set(pending) != required or pending["action"] not in self.action_space.all_actions():
                raise ValueError("Invalid pending execution checkpoint")
            prior_id = pending["prior_observation_id"]
            if isinstance(prior_id, bool) or not isinstance(prior_id, int) or prior_id != identifier:
                raise ValueError("Pending execution observation counter mismatch")
            prior = checked_belief(GaussianBelief(pending["prior"]["mean"], pending["prior"]["covariance"]), self.dim)
            predicted = checked_belief(GaussianBelief(pending["predicted"]["mean"], pending["predicted"]["covariance"]), self.dim)
            context = np.asarray(pending["context"], dtype=float)
            if not self._same_belief(prior, belief) or context.ndim != 1 or not np.all(np.isfinite(context)):
                raise ValueError("Pending execution prior/context mismatch")
            delivery = pending["observation"]
            fields = {"observation_id", "mean", "covariance", "source", "simulated"}
            if not isinstance(delivery, dict) or set(delivery) != fields:
                raise ValueError("Invalid pending observation schema")
            observation = self._checked_delivery(ExecutionObservation(**delivery), after=identifier)
            pending = {"action": pending["action"], "prior": prior, "prior_observation_id": prior_id,
                       "context": context.copy(), "predicted": predicted, "observation": observation}
        self.belief, self.last_observation_id, self._pending_execution = belief, identifier, pending
