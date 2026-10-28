"""Empirical observation channels for acquiring real right-hand text context.

The channel predicts the next marginal-probability bin conditional on an actual
training boundary label, the currently visible marginal/age bin, and an amount
of context requested. Only supervised, observed prefix extensions may be
recorded. No runtime model guess is promoted to a boundary label.
"""
from __future__ import annotations

from bisect import bisect_right
import math
from numbers import Integral, Real


_VERSION = 1
_MAX_BINS = 64
_MAX_ACTIONS = 16
_MAX_OBSERVATIONS = (1 << 53) - 1


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def _probability_edges(values, name):
    try:
        edges = tuple(_probability(value, name) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of probability edges") from exc
    if (not 2 <= len(edges) <= _MAX_BINS + 1 or edges[0] != 0.0 or edges[-1] != 1.0
            or any(left >= right for left, right in zip(edges, edges[1:]))):
        raise ValueError(f"{name} must be strictly increasing from zero to one with at most {_MAX_BINS} bins")
    return edges


def _entropy(probability):
    if probability == 0 or probability == 1:
        return 0.0
    return -probability * math.log(probability) - (1.0 - probability) * math.log1p(-probability)


class ContextAcquisitionModel:
    """Small label-conditional channel learned from real prefix extensions.

    ``age`` is the number of visible characters to the right of the queried
    boundary. Age edges start bins; their final bin is unbounded above.
    Probability edges delimit bins with the final edge (one) included.

    ``information_gain`` is mutual information under this fitted channel,
    measured in nats. It is an expected model quantity, not realized evidence or
    a guarantee that a particular extension reduces entropy or improves F1.
    """

    def __init__(
        self,
        action_chars=(4, 16),
        probability_edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        age_edges=(0, 4, 8, 16, 32),
        outcome_edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        alpha=0.5,
        min_per_label=5,
    ):
        try:
            actions = tuple(_integer(value, "action_chars", minimum=1) for value in action_chars)
            ages = tuple(_integer(value, "age_edges") for value in age_edges)
        except TypeError as exc:
            raise ValueError("actions and age edges must be integer sequences") from exc
        if not 1 <= len(actions) <= _MAX_ACTIONS or len(set(actions)) != len(actions):
            raise ValueError(f"Declare one to {_MAX_ACTIONS} distinct acquisition actions")
        if (not 1 <= len(ages) <= _MAX_BINS or ages[0] != 0
                or any(left >= right for left, right in zip(ages, ages[1:]))):
            raise ValueError("age_edges must increase strictly from zero")
        probabilities = _probability_edges(probability_edges, "probability_edges")
        outcomes = _probability_edges(outcome_edges, "outcome_edges")
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(float(alpha) * (len(outcomes) - 1)):
            raise ValueError("alpha is too large for the declared outcome bins")
        minimum = _integer(min_per_label, "min_per_label", minimum=1)
        self._action_chars = actions
        self._probability_edges = probabilities
        self._age_edges = ages
        self._outcome_edges = outcomes
        self._alpha = float(alpha)
        self._min_per_label = minimum
        size = len(actions) * (len(probabilities) - 1) * len(ages) * 2 * (len(outcomes) - 1)
        self._counts = [0] * size
        self._observations = 0
        self._frozen = False

    @property
    def action_chars(self):
        return self._action_chars

    @property
    def probability_edges(self):
        return self._probability_edges

    @property
    def age_edges(self):
        return self._age_edges

    @property
    def outcome_edges(self):
        return self._outcome_edges

    @property
    def alpha(self):
        return self._alpha

    @property
    def min_per_label(self):
        return self._min_per_label

    @property
    def observations(self):
        return self._observations

    @property
    def frozen(self):
        return self._frozen

    def _key(self, probability, age, action_chars):
        probability = _probability(probability, "probability")
        age = _integer(age, "age")
        action = _integer(action_chars, "action_chars", minimum=1)
        if action not in self._action_chars:
            raise ValueError("Acquisition action is outside the model configuration")
        probability_bin = min(len(self._probability_edges) - 2, bisect_right(self._probability_edges, probability) - 1)
        age_bin = bisect_right(self._age_edges, age) - 1
        return probability, self._action_chars.index(action), probability_bin, age_bin

    def _offset(self, action_index, probability_bin, age_bin, label):
        return (((action_index * (len(self._probability_edges) - 1) + probability_bin)
                 * len(self._age_edges) + age_bin) * 2 + label) * (len(self._outcome_edges) - 1)

    def record(self, prob_before, prob_after, gold, age, action_chars):
        """Record one labeled boundary after actually observing its extension."""
        if self._frozen:
            raise RuntimeError("The acquisition channel is frozen")
        _, action_index, probability_bin, age_bin = self._key(prob_before, age, action_chars)
        after = _probability(prob_after, "prob_after")
        label = _integer(gold, "gold")
        if label not in (0, 1):
            raise ValueError("gold must be zero or one")
        if self._observations >= _MAX_OBSERVATIONS:
            raise OverflowError("Acquisition observation count exceeds the checkpoint bound")
        outcome_bin = min(len(self._outcome_edges) - 2, bisect_right(self._outcome_edges, after) - 1)
        self._counts[self._offset(action_index, probability_bin, age_bin, label) + outcome_bin] += 1
        self._observations += 1

    def estimate(self, probability, age, action_chars, *, conditioning_probability=None):
        """Predict Bayes risk and information gain from the learned channel.

        Unsupported state/action pairs return no gain. Symmetric Dirichlet
        smoothing is used only after each label has the required real support.
        ``probability`` is the current latent boundary-label prior. The optional
        ``conditioning_probability`` is the last raw CRF marginal used to select
        the trained observation-state bin. Keeping these separate permits a
        controlled Markov observation filter without multiplying CRF posteriors
        as independent evidence. The same latent prior applies to both
        label-conditional outcome likelihoods.
        """
        prior = _probability(probability, "probability")
        conditioning = prior if conditioning_probability is None else _probability(
            conditioning_probability, "conditioning_probability")
        _, action_index, probability_bin, age_bin = self._key(conditioning, age, action_chars)
        width = len(self._outcome_edges) - 1
        rows = [self._counts[self._offset(action_index, probability_bin, age_bin, label):
                             self._offset(action_index, probability_bin, age_bin, label) + width]
                for label in (0, 1)]
        support = [sum(row) for row in rows]
        current_risk = min(prior, 1.0 - prior)
        current_entropy = _entropy(prior)
        result = {
            "eligible": min(support) >= self._min_per_label,
            "support": sum(support), "support_per_label": support,
            "prior_probability": prior, "conditioning_probability": conditioning,
            "current_risk": current_risk, "expected_risk": current_risk,
            "current_entropy": current_entropy, "information_gain": 0.0,
            "predicted_posterior_entropy": current_entropy,
            "likelihoods": [], "outcome_probabilities": [], "posterior_probabilities": [],
            "state_bin": {"probability": probability_bin, "age": age_bin, "action_chars": int(action_chars)},
        }
        if not result["eligible"]:
            return result
        likelihoods = [[(count + self._alpha) / (total + width * self._alpha) for count in row]
                       for row, total in zip(rows, support)]
        outcomes, posteriors = [], []
        entropy_terms, risk_terms = [], []
        for likelihood_0, likelihood_1 in zip(*likelihoods):
            joint_0, joint_1 = (1.0 - prior) * likelihood_0, prior * likelihood_1
            outcome = joint_0 + joint_1
            posterior = joint_1 / outcome if outcome else None
            outcomes.append(outcome)
            posteriors.append(posterior)
            entropy_terms.append(outcome * _entropy(posterior) if outcome else 0.0)
            risk_terms.append(min(joint_0, joint_1))
        expected_entropy = math.fsum(entropy_terms)
        expected_risk = math.fsum(risk_terms)
        # The coherent finite channel obeys these bounds analytically. Clamp
        # floating-point cancellation at the boundaries, not the observations.
        result.update(likelihoods=likelihoods, outcome_probabilities=outcomes,
                      posterior_probabilities=posteriors,
                      expected_risk=max(0.0, min(current_risk, expected_risk)),
                      predicted_posterior_entropy=max(0.0, min(current_entropy, expected_entropy)),
                      information_gain=max(0.0, min(current_entropy, current_entropy - expected_entropy)))
        return result

    def freeze(self):
        """Disable further training/loading in place and return this model."""
        self._frozen = True
        return self

    def snapshot(self):
        """Return an independent frozen copy suitable for evaluation sessions."""
        return self.from_state(self.export_state()).freeze()

    def _configuration(self):
        return {"action_chars": list(self._action_chars), "probability_edges": list(self._probability_edges),
                "age_edges": list(self._age_edges), "outcome_edges": list(self._outcome_edges),
                "alpha": self._alpha, "min_per_label": self._min_per_label}

    def export_state(self):
        return {"version": _VERSION, "config": self._configuration(), "counts": self._counts.copy(),
                "observations": self._observations, "frozen": self._frozen}

    @classmethod
    def from_state(cls, state):
        config = state.get("config") if isinstance(state, dict) else None
        expected = {"action_chars", "probability_edges", "age_edges", "outcome_edges", "alpha", "min_per_label"}
        if not isinstance(config, dict) or set(config) != expected:
            raise ValueError("Acquisition checkpoint has an invalid configuration schema")
        model = cls(**config)
        model.load_state(state)
        return model

    def load_state(self, state):
        """Atomically load a matching, finite, integer-count checkpoint."""
        if self._frozen:
            raise RuntimeError("The acquisition channel is frozen")
        expected = {"version", "config", "counts", "observations", "frozen"}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("Acquisition checkpoint has an invalid schema")
        if isinstance(state["version"], bool) or not isinstance(state["version"], Integral) or state["version"] != _VERSION:
            raise ValueError("Unsupported acquisition checkpoint version")
        if state["config"] != self._configuration():
            raise ValueError("Acquisition checkpoint configuration mismatch")
        # Validate configuration types as well as equality (True == 1 is not a
        # valid way to smuggle a boolean into an integer checkpoint field).
        validated = type(self)(**state["config"])
        if validated._configuration() != self._configuration():
            raise ValueError("Acquisition checkpoint configuration mismatch")
        raw_counts = state["counts"]
        if not isinstance(raw_counts, list) or len(raw_counts) != len(self._counts):
            raise ValueError("Acquisition checkpoint count shape mismatch")
        counts = [_integer(value, "count") for value in raw_counts]
        observations = _integer(state["observations"], "observations")
        if observations != sum(counts) or observations > _MAX_OBSERVATIONS:
            raise ValueError("Acquisition checkpoint observation count mismatch")
        if not isinstance(state["frozen"], bool):
            raise ValueError("Acquisition checkpoint frozen flag must be a boolean")
        self._counts = counts
        self._observations = observations
        self._frozen = state["frozen"]
