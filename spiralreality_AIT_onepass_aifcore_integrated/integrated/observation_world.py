"""A task-outcome world with persistent Bayesian boundary/regime beliefs.

The two regime hypotheses are the fitted label-conditional context channel and
a label-independent null channel. The null is the equal-label mean of the two
fitted likelihood rows, not an empirical correctness rate. Neither hypothesis
uses runtime labels. A caller conditions on actual delivered outcomes by
retaining the corresponding joint posterior; hypothetical estimates are pure.

This makes uncertainty about an observation hypothesis part of task planning.
It does not calibrate correctness, establish semantic information, or improve
the separate encoder-feature dynamics model. The finite controlled-Markov
channel and persistent-regime assumptions still require empirical validation.
"""
from __future__ import annotations

import math
from numbers import Real

from .acquisition import ContextAcquisitionModel


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def _belief(value):
    # Both immutable runtime tuples and their JSON-array round trips are
    # accepted. Do not silently turn arbitrary iterators/maps into a state.
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("belief must have four joint probabilities")
    probabilities = tuple(_probability(item, "belief probability") for item in value)
    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("belief probabilities must sum to one")
    # Remove accumulated arithmetic error only inside the declared tolerance.
    return tuple(probability / total for probability in probabilities)


def _entropy(probability):
    if probability == 0.0 or probability == 1.0:
        return 0.0
    return -probability * math.log(probability) - (1.0 - probability) * math.log1p(-probability)


def _bounded(value, upper):
    return max(0.0, min(upper, value))


class BayesianObservationWorld:
    """Frozen outcome hypotheses with exact, caller-owned joint belief state.

    Beliefs have order ``(null/Y0, null/Y1, informative/Y0, informative/Y1)``.
    ``initial_belief`` starts independent label/regime marginals. Once an outcome
    is observed these variables generally become correlated. Keep the complete
    ``joint_posteriors[outcome]`` for later estimates: rebuilding independent
    marginals would discard evidence and change the generative model.

    Under the informative hypothesis, likelihoods are the frozen channel's
    ``P(Z | Y, raw_state, age, action)``. Under the null hypothesis both labels
    use ``Q(Z) = (P(Z|Y=0) + P(Z|Y=1))/2``. Equal label weighting makes the null
    explicitly uninformative without injecting the calibration sample's label
    prevalence. It is a comparison hypothesis, not a learned claim that the
    runtime source is unreliable. At a fresh label prior of one half, the two
    regimes predict identical outcome marginals and one observation cannot
    identify the regime.

    A regime persists across observations of the *same boundary*. This class
    carries no stream-global regime state and never fits runtime pseudo-labels.
    Outcome probabilities, Bayes risk, label information, and regime information
    are all derived from the same joint distribution. Information is in nats.
    """

    def __init__(self, channel: ContextAcquisitionModel, informative_prior=0.5):
        if not isinstance(channel, ContextAcquisitionModel):
            raise TypeError("channel must be ContextAcquisitionModel")
        self._informative_prior = _probability(informative_prior, "informative_prior")
        self._channel = channel.snapshot()

    @property
    def channel(self):
        return self._channel

    @property
    def informative_prior(self):
        return self._informative_prior

    def initial_belief(self, probability):
        """Start a boundary with declared independent label/regime priors."""
        probability = _probability(probability, "probability")
        informative = self._informative_prior
        return ((1.0 - informative) * (1.0 - probability),
                (1.0 - informative) * probability,
                informative * (1.0 - probability),
                informative * probability)

    @staticmethod
    def label_probability(belief):
        """Return the boundary-label-one marginal of a validated joint state."""
        joint = _belief(belief)
        return _bounded(math.fsum((joint[1], joint[3])), 1.0)

    @staticmethod
    def informative_probability(belief):
        """Return the informative-regime marginal of a validated joint state."""
        joint = _belief(belief)
        return _bounded(math.fsum((joint[2], joint[3])), 1.0)

    def estimate(self, belief, raw_probability, age, action_chars):
        """Purely predict outcomes and their exact joint Bayesian posteriors.

        ``raw_probability`` selects the empirically fitted channel cell; it is
        separate from the maintained label marginal. Unsupported cells return
        no outcome distribution and no modeled gain, matching the base channel.
        No observation or training count changes during this method.

        ``likelihoods`` retains the base API's two label-conditional rows. When
        the belief is correlated, these average using ``P(regime | label)``.
        A zero-mass label has no defined conditional regime; its unused row is
        averaged with the regime marginal. ``joint_likelihoods`` exposes the
        four fixed hypothesis rows, and ``joint_posteriors`` holds one four-item
        tuple per possible outcome (``None`` for a zero-probability outcome).
        """
        joint = _belief(belief)
        prior = _bounded(math.fsum((joint[1], joint[3])), 1.0)
        informative = _bounded(math.fsum((joint[2], joint[3])), 1.0)
        result = self._channel.estimate(prior, age, action_chars, conditioning_probability=raw_probability)
        regime_entropy = _entropy(informative)
        result.update(joint_posteriors=[], joint_likelihoods=[],
                      informative_probability=informative, regime_information_gain=0.0,
                      current_regime_entropy=regime_entropy,
                      predicted_regime_entropy=regime_entropy)
        if not result["eligible"]:
            return result

        likelihood_0, likelihood_1 = result["likelihoods"]
        null = [0.5 * (left + right) for left, right in zip(likelihood_0, likelihood_1)]
        rows = [null.copy(), null.copy(), likelihood_0.copy(), likelihood_1.copy()]
        effective_rows = []
        for label in (0, 1):
            label_mass = math.fsum((joint[label], joint[2 + label]))
            regime_given_label = joint[2 + label] / label_mass if label_mass else informative
            effective_rows.append([(1.0 - regime_given_label) * nuisance + regime_given_label * trained
                                   for nuisance, trained in zip(null, rows[2 + label])])

        outcomes, posteriors, joint_posteriors = [], [], []
        label_entropy_terms, regime_entropy_terms, risk_terms = [], [], []
        for likelihoods in zip(*rows):
            masses = tuple(probability * likelihood for probability, likelihood in zip(joint, likelihoods))
            outcome = math.fsum(masses)
            outcomes.append(outcome)
            if outcome == 0.0:
                # Very small positive smoothing can underflow for unused bins.
                # Such an outcome contributes zero to every expectation.
                posteriors.append(None)
                joint_posteriors.append(None)
                continue
            posterior_joint = tuple(mass / outcome for mass in masses)
            posterior = _bounded(math.fsum((posterior_joint[1], posterior_joint[3])), 1.0)
            posterior_regime = _bounded(math.fsum((posterior_joint[2], posterior_joint[3])), 1.0)
            posteriors.append(posterior)
            joint_posteriors.append(posterior_joint)
            label_entropy_terms.append(outcome * _entropy(posterior))
            regime_entropy_terms.append(outcome * _entropy(posterior_regime))
            risk_terms.append(min(math.fsum((masses[0], masses[2])), math.fsum((masses[1], masses[3]))))

        label_entropy = result["current_entropy"]
        expected_label_entropy = _bounded(math.fsum(label_entropy_terms), label_entropy)
        expected_regime_entropy = _bounded(math.fsum(regime_entropy_terms), regime_entropy)
        result.update(likelihoods=effective_rows, joint_likelihoods=rows,
                      outcome_probabilities=outcomes, posterior_probabilities=posteriors,
                      joint_posteriors=joint_posteriors,
                      expected_risk=_bounded(math.fsum(risk_terms), result["current_risk"]),
                      predicted_posterior_entropy=expected_label_entropy,
                      information_gain=_bounded(label_entropy - expected_label_entropy, label_entropy),
                      predicted_regime_entropy=expected_regime_entropy,
                      regime_information_gain=_bounded(regime_entropy - expected_regime_entropy, regime_entropy))
        return result


__all__ = ["BayesianObservationWorld"]
