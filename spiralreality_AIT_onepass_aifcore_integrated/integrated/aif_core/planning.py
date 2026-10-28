from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable

from ..np_compat import np
from .beliefs import GaussianBelief
from .efe import EFEEngine
from .world_model import ActionSpace


@dataclass
class RolloutResult:
    actions: list[str]
    terms: list[dict]
    aggregate: dict


def checked_belief(value, dimension):
    if not isinstance(value, GaussianBelief):
        raise ValueError("step_fn must return a GaussianBelief forecast")
    mean = np.asarray(value.mu, dtype=float)
    covariance = np.asarray(value.Sigma, dtype=float)
    if mean.shape != (dimension,) or covariance.shape != (dimension, dimension):
        raise ValueError("Belief dimensions do not match the agent")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("Belief must contain finite values")
    if not np.allclose(covariance, covariance.T) or np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
        raise ValueError("Belief covariance must be symmetric positive semidefinite")
    return GaussianBelief(mean.copy(), (covariance + covariance.T) * 0.5)


def checked_observation(mean, covariance, dimension):
    result = checked_belief(GaussianBelief(mean, covariance), dimension)
    if np.min(np.linalg.eigvalsh(result.Sigma)) <= 0:
        raise ValueError("Observation covariance must be positive definite")
    return result.mu, result.Sigma


def forecast_step(step_fn, belief, action, context):
    """Return next-state prior, observation mean, and sensor likelihood noise R.

    The observation is in the latent state's coordinates with identity mapping.
    R excludes latent predictive covariance P; preference risk integrates P+R,
    while assimilation conditions the next-state prior using R alone.
    """
    # All branches receive owned arrays. A forecasting function must itself be
    # pure; no observation/execution callback is invoked during planning.
    next_belief, obs_mean, obs_covariance = step_fn(belief.copy(), action, np.asarray(context, dtype=float).copy())
    next_belief = checked_belief(next_belief, belief.dim)
    obs_mean, obs_covariance = checked_observation(obs_mean, obs_covariance, belief.dim)
    return next_belief, obs_mean, obs_covariance


def _actions(action_space, horizon):
    actions = action_space.all_actions()
    if not actions or len(set(actions)) != len(actions) or any(not isinstance(action, str) or not action for action in actions):
        raise ValueError("Action space must contain distinct nonempty action names")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return actions


def _terms(efe, belief, next_belief, obs_mean, obs_covariance, r3):
    # Expected future evidence is evaluated against the predicted next prior,
    # not the current-state prior. This posterior remains hypothetical.
    anticipated = next_belief.merge_with_observation(obs_mean, obs_covariance)
    predictive_observation_covariance = next_belief.Sigma + obs_covariance
    term = efe.decompose_step(next_belief, obs_mean, predictive_observation_covariance, anticipated, r3_mix=r3)
    if not all(math.isfinite(value) for value in term.values()):
        raise ValueError("Nonfinite rollout score")
    return anticipated, {**term, "evidence": "forecast_only"}


class CEMPlanner:
    def __init__(self, efe: EFEEngine, action_space: ActionSpace, horizon: int = 3,
                 K: int = 48, elite_frac: float = 0.25, iters: int = 2):
        _actions(action_space, horizon)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (K, iters)):
            raise ValueError("K and iters must be positive integers")
        if not math.isfinite(elite_frac) or not 0 < elite_frac <= 1:
            raise ValueError("elite_frac must be in (0, 1]")
        self.efe, self.action_space = efe, action_space
        self.horizon, self.K, self.elite_frac, self.iters = horizon, K, elite_frac, iters

    def plan(self, belief: GaussianBelief, step_fn: Callable, ctx_provider: Callable,
             r3_provider: Callable) -> RolloutResult:
        actions = _actions(self.action_space, self.horizon)
        belief = checked_belief(belief, belief.dim)
        count = len(actions)
        probs = [[1.0 / count for _ in range(count)] for _ in range(self.horizon)]
        rng = np.random.default_rng(42)

        def evaluate(sequence):
            current, terms = belief.copy(), []
            for index, action in enumerate(sequence):
                prefix = sequence[:index + 1]
                next_belief, mean, covariance = forecast_step(step_fn, current, action, ctx_provider(prefix))
                current, term = _terms(self.efe, current, next_belief, mean, covariance, r3_provider(prefix))
                terms.append(term)
            return RolloutResult(list(sequence), terms, self.efe.total_over_rollout(terms))

        best = None
        for _ in range(self.iters):
            samples = [[actions[int(rng.choice(count, p=probs[index]))] for index in range(self.horizon)] for _ in range(self.K)]
            evaluated = [evaluate(sequence) for sequence in samples]
            evaluated.sort(key=lambda result: result.aggregate["total"])
            if best is None or evaluated[0].aggregate["total"] < best.aggregate["total"]:
                best = evaluated[0]
            elite = evaluated[:max(1, int(self.K * self.elite_frac))]
            probs = [[1e-9 for _ in range(count)] for _ in range(self.horizon)]
            for result in elite:
                for index, action in enumerate(result.actions):
                    probs[index][actions.index(action)] += 1
            probs = [[value / sum(row) for value in row] for row in probs]
        modal = evaluate([actions[max(range(count), key=lambda index: probs[t][index])] for t in range(self.horizon)])
        return modal if modal.aggregate["total"] < best.aggregate["total"] else best


class SimpleTreePlanner:
    def __init__(self, efe: EFEEngine, action_space: ActionSpace, horizon: int = 3, branch_limit: int = 4):
        _actions(action_space, horizon)
        if isinstance(branch_limit, bool) or not isinstance(branch_limit, int) or branch_limit < 1:
            raise ValueError("branch_limit must be a positive integer")
        self.efe, self.action_space = efe, action_space
        self.horizon, self.branch_limit = horizon, branch_limit

    def plan(self, belief: GaussianBelief, step_fn, ctx_provider, r3_provider) -> RolloutResult:
        actions = _actions(self.action_space, self.horizon)
        belief = checked_belief(belief, belief.dim)
        best = None

        def visit(prefix, current, terms):
            nonlocal best
            if len(prefix) == self.horizon:
                result = RolloutResult(prefix, terms, self.efe.total_over_rollout(terms))
                if best is None or result.aggregate["total"] < best.aggregate["total"]:
                    best = result
                return
            for action in actions[:self.branch_limit]:
                path = prefix + [action]
                next_belief, mean, covariance = forecast_step(step_fn, current, action, ctx_provider(path))
                anticipated, term = _terms(self.efe, current, next_belief, mean, covariance, r3_provider(path))
                visit(path, anticipated, terms + [term])
        visit([], belief, [])
        return best
