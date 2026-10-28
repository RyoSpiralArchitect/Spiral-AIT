"""Grounded context acquisition with pure plans and explicit text observations.

The finite observation channel supplies expected risk/information and conditions
label beliefs on genuinely observed raw-CRF state bins. Overlapping raw CRF
posteriors are never multiplied as independent measurements. Unsupported/new
positions and reframed windows fall back to raw CRF beliefs. World-model learning
and phase tracking happen only after characters actually arrive.
"""
from __future__ import annotations

from collections import Counter, deque
from bisect import bisect_right
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real

from .acquisition import ContextAcquisitionModel
from .observation_world import BayesianObservationWorld
from .np_compat import np


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _entropy(p):
    return -p * math.log(p) - (1 - p) * math.log1p(-p) if 0 < p < 1 else 0.0


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def marginal_tokens(text, probabilities):
    """Binary marginal-MAP segmentation, minimizing modeled boundary error."""
    if not isinstance(text, str) or len(probabilities) != max(0, len(text) - 1):
        raise ValueError("One boundary marginal is required per interior character position")
    tokens, start = [], 0
    for index, probability in enumerate(probabilities):
        p = float(probability)
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Boundary marginals must be finite probabilities")
        if p >= 0.5:
            tokens.append(text[start:index + 1])
            start = index + 1
    if text:
        tokens.append(text[start:])
    return tokens


@dataclass(frozen=True)
class ActiveContextConfig:
    initial_chars: int = 8
    action_chars: tuple = (4, 16)
    max_window_chars: int = 96
    context_chars: int = 32
    max_acquisitions_per_commit: int = 4
    horizon: int = 2
    char_cost: float = 0.002
    call_cost: float = 0.002
    epistemic_weight: float = 0.05
    phase_gain: float = 0.5
    surprise_gain: float = 0.25
    learn_dynamics: bool = True
    trace_limit: int = 256
    observation_model: str = "fixed_channel"
    informative_prior: float = 0.5
    learn_observation_regime: bool = True
    model_epistemic_weight: float = 0.0

    def __post_init__(self):
        for name in ("initial_chars", "max_window_chars", "max_acquisitions_per_commit", "horizon"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        for name in ("context_chars", "trace_limit"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.max_window_chars < 9 or self.initial_chars > self.max_window_chars:
            raise ValueError("The initial observation must fit a window of at least nine characters")
        if self.context_chars >= self.max_window_chars or self.horizon > 3:
            raise ValueError("Left context must leave window capacity, and horizon must be <= 3")
        actions = tuple(_integer(value, "action_chars", 1) for value in self.action_chars)
        if not actions or len(set(actions)) != len(actions) or max(actions) > self.max_window_chars:
            raise ValueError("Declare distinct bounded positive acquisition sizes")
        object.__setattr__(self, "action_chars", actions)
        for name in ("char_cost", "call_cost", "epistemic_weight", "phase_gain", "surprise_gain", "model_epistemic_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.learn_dynamics, bool):
            raise ValueError("learn_dynamics must be boolean")
        if self.observation_model not in ("fixed_channel", "bayesian_regime"):
            raise ValueError("observation_model must be fixed_channel or bayesian_regime")
        if (isinstance(self.informative_prior, bool) or not isinstance(self.informative_prior, Real)
                or not math.isfinite(self.informative_prior) or not 0 <= self.informative_prior <= 1):
            raise ValueError("informative_prior must be a finite probability")
        object.__setattr__(self, "informative_prior", float(self.informative_prior))
        if not isinstance(self.learn_observation_regime, bool):
            raise ValueError("learn_observation_regime must be boolean")


@dataclass(frozen=True)
class TextObservation:
    observation_id: int
    text: str
    eof: bool
    source: str = "provided_text_replay"


class StringTextSource:
    """Pull-only text replay; the actor receives no length/future-content API.

    Repeating the last request ID/size returns its receipt without advancing the
    source. An external implementation should provide the same idempotency or
    retain delivered observations itself when processing is retried.
    """
    def __init__(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._text, self._position, self._last = text, 0, None

    def read(self, requested_chars: int, observation_id: int) -> TextObservation:
        count = _integer(requested_chars, "requested_chars", 1)
        identifier = _integer(observation_id, "observation_id", 1)
        if self._last is not None:
            last_count, last = self._last
            if identifier == last.observation_id and count == last_count:
                return last
            if identifier <= last.observation_id:
                raise ValueError("Source request IDs must increase, except an identical last retry")
        end = min(len(self._text), self._position + count)
        observation = TextObservation(identifier, self._text[self._position:end], end == len(self._text))
        self._position, self._last = end, (count, observation)
        return observation


@dataclass(frozen=True)
class AcquisitionCandidate:
    action: str
    requested_chars: int
    total: float
    expected_risk: float
    expected_information_gain: float
    support: int
    supported: bool
    expected_model_information_gain: float = 0.0


@dataclass(frozen=True)
class ContextDecision:
    state_version: int
    observation_id: int
    action: str
    requested_chars: int
    reason: str
    focus_index: int | None
    focus_offset: int | None
    probability: float | None
    conditioning_probability: float | None
    age: int
    phase_drive: float
    surprise_drive: float
    candidates: tuple[AcquisitionCandidate, ...]
    informative_probability: float | None = None


class ActiveSessionError(RuntimeError):
    """A failed run retains already emitted tokens in ``partial_result``."""
    def __init__(self, message, partial_result):
        super().__init__(message)
        self.partial_result = partial_result


def acquisition_policy(requested_chars, action_chars, *, initial=False):
    if initial:
        return "ProbeMotivation"
    return "SeekEvidence" if requested_chars >= max(action_chars) else "ProbeReliability"


class ActiveTextSession:
    """Single-owner AIT controller for acquiring context before irrevocable cuts.

    ``plan`` is pure and cannot access the source. ``step`` executes exactly one
    plan, retaining a successfully acquired observation if later processing
    fails. Histories are bounded, and the frozen label channel never learns from
    runtime pseudo-labels. Only the separate latent dynamics model adapts online.
    ``model`` must not be externally trained/loaded/observed during a session.
    """
    def __init__(self, model, channel: ContextAcquisitionModel, config=None):
        self.model = model
        self.config = config or ActiveContextConfig()
        if not isinstance(self.config, ActiveContextConfig):
            raise TypeError("config must be ActiveContextConfig")
        self.channel = channel.snapshot()
        self.observation_world = (BayesianObservationWorld(self.channel, informative_prior=self.config.informative_prior)
                                  if self.config.observation_model == "bayesian_regime" else None)
        if not set(self.config.action_chars) <= set(self.channel.action_chars):
            raise ValueError("Configured acquisition sizes must be present in the learned channel")
        if self.config.horizon > 1 and self.channel.probability_edges != self.channel.outcome_edges:
            raise ValueError("Multistep channel planning requires matching observed-state and outcome bins")
        model.student._require_python_head()
        if not model.student._context_ready():
            raise ValueError("Active context requires the contextual Python boundary head")
        self._context = self._pending = ""
        self._observation = None
        self._eof = False
        self._version = self._received = self._committed = self._reads_since_commit = 0
        self._model_observation_id = model.last_observation_id
        self._source_observation_id = model.last_observation_id
        self._latent = np.zeros(model.latent_dim, dtype=float)
        self._surprise = 0.0
        self._retry = None
        self._trace = deque(maxlen=self.config.trace_limit)
        self._events = 0
        self._metrics = Counter()
        self._waits, self._policies = Counter(), Counter()
        self._initial_world_updates = model.dynamics.trained_steps

    @property
    def pending_text(self):
        return self._pending

    @property
    def trace(self):
        # Callers cannot edit a retained event or a nested candidate in place.
        return json.loads(json.dumps(list(self._trace)))

    @property
    def metrics(self):
        names = ("acquire_calls", "acquired_chars", "deliberate_acquire_calls", "commit_actions",
                 "forced_splits", "model_calls", "window_square_sum", "max_window_chars_seen",
                 "budget_forced_cuts", "window_forced_cuts", "truncated_acquisitions", "channel_boundary_updates",
                 "observation_world_updates", "outcome_forecasts")
        return {**{name: self._metrics[name] for name in names},
                "received_chars": self._received, "committed_chars": self._committed,
                "wait_char_histogram": dict(self._waits), "policy_counts": dict(self._policies),
                "world_model_updates": self.model.dynamics.trained_steps - self._initial_world_updates,
                "observations": self._metrics["observations"], "events": self._events,
                "trace_truncated": max(0, self._events - len(self._trace)),
                "supported_decisions": self._metrics["supported_decisions"],
                "forecast_evaluations": self._metrics["forecast_evaluations"],
                "expected_information_gain_sum": float(self._metrics["expected_information_gain_sum"]),
                "realized_entropy_reduction_sum": float(self._metrics["realized_entropy_reduction_sum"]),
                "regime_information_gain_sum": float(self._metrics["regime_information_gain_sum"]),
                "outcome_nll_sum": float(self._metrics["outcome_nll_sum"]),
                "outcome_brier_sum": float(self._metrics["outcome_brier_sum"])}

    def _check_owner(self):
        if self.model.last_observation_id != self._model_observation_id:
            raise RuntimeError("The model received observations outside this active session")

    def _drives(self):
        phase = math.tanh(max(0.0, float(self.model.R3_mix)))
        weight = self.config.epistemic_weight * (1 + self.config.phase_gain * phase + self.config.surprise_gain * self._surprise)
        return phase, weight

    def _estimate(self, probability, raw_probability, age, count, joint=None):
        if self.observation_world is None:
            return self.channel.estimate(probability, age, count, conditioning_probability=raw_probability)
        belief = self.observation_world.initial_belief(probability) if joint is None else joint
        return self.observation_world.estimate(belief, raw_probability, age, count)

    def _continuation_belief(self, estimate, outcome):
        if self.observation_world is None:
            return None
        joint = estimate["joint_posteriors"][outcome]
        # The explicit ablation forgets regime evidence AND label/regime
        # correlation after each delivery, while retaining the label marginal.
        # Full inference must retain the joint instead of repeatedly remixing.
        if not self.config.learn_observation_regime:
            joint = self.observation_world.initial_belief(estimate["posterior_probabilities"][outcome])
        return joint

    def _value(self, probability, raw_probability, age, horizon, capacity, weight, joint=None):
        best = min(probability, 1 - probability)
        if horizon <= 0:
            return best
        for count in self.config.action_chars:
            if count > capacity:
                continue
            estimate = self._estimate(probability, raw_probability, age, count, joint)
            if not estimate["eligible"]:
                continue
            future = math.fsum(mass * self._value(posterior,
                               (self.channel.outcome_edges[index] + self.channel.outcome_edges[index + 1]) / 2,
                               age + count, horizon - 1, capacity - count, weight,
                               self._continuation_belief(estimate, index))
                               for index, (mass, posterior) in enumerate(zip(estimate["outcome_probabilities"], estimate["posterior_probabilities"])) if mass)
            value = (future + count * self.config.char_cost + self.config.call_cost - weight * estimate["information_gain"]
                     - self.config.model_epistemic_weight * estimate.get("regime_information_gain", 0.0))
            best = min(best, value)
        return best

    def plan(self):
        self._check_owner()
        phase, weight = self._drives()
        identifier = max(self._model_observation_id, self._source_observation_id) + 1
        values = dict(state_version=self._version, observation_id=identifier, focus_index=None,
                      focus_offset=None,
                      probability=None, conditioning_probability=None, age=0,
                      phase_drive=phase, surprise_drive=self._surprise, candidates=())
        if not self._pending:
            available = self.config.max_window_chars - len(self._context)
            return ContextDecision(action="Done" if self._eof else "Acquire", requested_chars=0 if self._eof else min(self.config.initial_chars, available),
                                   reason="end_of_source" if self._eof else "initial_context", **values)
        context_len = len(self._context)
        tokens = self._pending_tokens(self._observation, context_len)
        token_length = len(tokens[0])
        first_end = context_len + token_length
        probabilities = self._observation["boundary_probabilities"]
        indices = list(range(context_len, min(first_end, len(probabilities))))
        if indices:
            focus = max(indices, key=lambda i: (_entropy(float(probabilities[i])), -i))
            prior = float(probabilities[focus])
            raw_prior = float(self._observation["raw_boundary_probabilities"][focus])
            age = len(self._context + self._pending) - focus - 1
            values.update(focus_index=focus, focus_offset=self._committed - context_len + focus + 1,
                          probability=prior, conditioning_probability=raw_prior, age=age)
        capacity = self.config.max_window_chars - context_len - len(self._pending)
        boundary_visible = first_end < len(self._context + self._pending)
        if self._eof:
            return ContextDecision(action="Commit", requested_chars=0, reason="observed_eof", **values)
        if self._reads_since_commit >= self.config.max_acquisitions_per_commit or capacity == 0:
            return ContextDecision(action="Commit", requested_chars=0, reason="acquisition_budget" if capacity else "window_pressure", **values)
        if not boundary_visible:
            return ContextDecision(action="Acquire", requested_chars=min(min(self.config.action_chars), capacity),
                                   reason="unobserved_token_end", **values)
        if not indices:
            return ContextDecision(action="Commit", requested_chars=0, reason="no_internal_uncertainty", **values)
        commit_risk = min(prior, 1 - prior)
        horizon = min(self.config.horizon, self.config.max_acquisitions_per_commit - self._reads_since_commit)
        joint = self._observation["world_beliefs"][focus] if self.observation_world is not None else None
        if joint is not None:
            values["informative_probability"] = self.observation_world.informative_probability(joint)
        candidates = [AcquisitionCandidate("Commit", 0, commit_risk, commit_risk, 0.0, 0, True)]
        for count in self.config.action_chars:
            if count > capacity:
                continue
            estimate = self._estimate(prior, raw_prior, age, count, joint)
            if estimate["eligible"]:
                continuation = math.fsum(mass * self._value(posterior,
                                         (self.channel.outcome_edges[index] + self.channel.outcome_edges[index + 1]) / 2,
                                         age + count, horizon - 1, capacity - count, weight,
                                         self._continuation_belief(estimate, index))
                                         for index, (mass, posterior) in enumerate(zip(estimate["outcome_probabilities"], estimate["posterior_probabilities"])) if mass)
            else:
                continuation = commit_risk
            score = (continuation + count * self.config.char_cost + self.config.call_cost - weight * estimate["information_gain"]
                     - self.config.model_epistemic_weight * estimate.get("regime_information_gain", 0.0))
            candidates.append(AcquisitionCandidate("Acquire", count, score, estimate["expected_risk"],
                                                   estimate["information_gain"], estimate["support"], estimate["eligible"],
                                                   estimate.get("regime_information_gain", 0.0)))
        selected = min(candidates, key=lambda candidate: (candidate.total, candidate.requested_chars))
        values["candidates"] = tuple(candidates)
        return ContextDecision(action=selected.action, requested_chars=selected.requested_chars,
                               reason="modeled_value_of_context" if selected.requested_chars else "commit_has_lower_cost", **values)

    @staticmethod
    def _pending_tokens(observation, context_len):
        result, offset = [], 0
        for token in observation["tokens"]:
            end = offset + len(token)
            if end > context_len:
                result.append(token[max(0, context_len - offset):])
            offset = end
        return result

    def _inspect(self, text):
        observation = self.model.student.observe_sequence(text)
        tokens = observation["tokens"]
        probabilities = observation["boundary_probabilities"]
        latent = np.asarray(observation["context_vector"], dtype=float)
        if (any(not isinstance(token, str) or not token for token in tokens) or "".join(tokens) != text
                or len(probabilities) != max(0, len(text) - 1)
                or any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in probabilities)
                or latent.shape != (self.model.latent_dim,) or not np.all(np.isfinite(latent))):
            raise ValueError("Malformed contextual observation")
        raw = [float(probability) for probability in probabilities]
        return {**observation, "raw_boundary_probabilities": raw,
                "boundary_probabilities": raw.copy(), "tokens": marginal_tokens(text, raw),
                "world_beliefs": [self.observation_world.initial_belief(p) for p in raw]
                if self.observation_world is not None else None}

    def _filter_extension(self, observation, text, requested_chars, actual_chars, focus_index=None):
        """Condition the old label belief on a genuinely observed raw-state bin.

        The channel is conditional on the previous raw bin, age, and action. We
        never multiply the old and new CRF posteriors. Unsupported/partial/new
        positions use the raw CRF belief; reframing also resets to that belief.
        """
        if (self._observation is None or actual_chars != requested_chars
                or requested_chars not in self.channel.action_chars):
            return observation, 0, None
        beliefs = observation["boundary_probabilities"].copy()
        world_beliefs = observation["world_beliefs"].copy() if self.observation_world is not None else None
        updated = 0
        focused = None
        before_length = len(self._context + self._pending)
        for index, prior in enumerate(self._observation["boundary_probabilities"]):
            if index < len(self._context):
                continue  # Already committed text is not a mutable hypothesis.
            raw_prior = self._observation["raw_boundary_probabilities"][index]
            joint = self._observation["world_beliefs"][index] if self.observation_world is not None else None
            estimate = self._estimate(prior, raw_prior, before_length - index - 1, requested_chars, joint)
            if estimate["eligible"]:
                outcome = min(len(self.channel.outcome_edges) - 2,
                              bisect_right(self.channel.outcome_edges, observation["raw_boundary_probabilities"][index]) - 1)
                beliefs[index] = estimate["posterior_probabilities"][outcome]
                if world_beliefs is not None:
                    world_beliefs[index] = self._continuation_belief(estimate, outcome)
                updated += 1
                # Score only one predeclared focused outcome per acquisition.
                # Neighboring boundary outcomes share text and are correlated.
                if index == focus_index:
                    masses = estimate["outcome_probabilities"]
                    regime_before = estimate.get("informative_probability")
                    regime_after = (self.observation_world.informative_probability(world_beliefs[index])
                                    if world_beliefs is not None else None)
                    focused = {"outcome_bin": outcome, "outcome_probability": masses[outcome],
                               "outcome_nll": -math.log(masses[outcome]),
                               "outcome_brier": math.fsum((p - int(i == outcome)) ** 2 for i, p in enumerate(masses)),
                               "informative_probability_before": regime_before,
                               "informative_probability_after": regime_after,
                               "realized_regime_entropy_reduction": _entropy(regime_before) - _entropy(regime_after)
                               if regime_before is not None else None,
                               "joint_belief_before": joint,
                               "joint_belief_after": world_beliefs[index] if world_beliefs is not None else None,
                               "expected_regime_information_gain": estimate.get("regime_information_gain", 0.0)}
        return {**observation, "boundary_probabilities": beliefs, "tokens": marginal_tokens(text, beliefs),
                "world_beliefs": world_beliefs}, updated, focused

    def _record_inspection(self, text):
        self._metrics["model_calls"] += 1
        self._metrics["window_square_sum"] += len(text) ** 2
        self._metrics["max_window_chars_seen"] = max(self._metrics["max_window_chars_seen"], len(text))

    def _event(self, decision, **outcome):
        self._events += 1
        self._policies[decision.action + (str(decision.requested_chars) if decision.action == "Acquire" else "")] += 1
        self._trace.append({"event": self._events, **asdict(decision), "received_chars": self._received,
                            "committed_chars": self._committed, "phase_mixing": float(self.model.R3_mix),
                            "world_model_trained_steps": self.model.dynamics.trained_steps, **outcome})

    def execute(self, decision: ContextDecision, source=None, *, observation=None):
        if decision != self.plan():
            raise ValueError("Decision is stale or does not match the current pure plan")
        if decision.action == "Done":
            if observation is not None:
                raise ValueError("A completed session does not accept another observation")
            return {"tokens": [], "done": True, "decision": decision}
        if decision.action == "Commit":
            if observation is not None:
                raise ValueError("Commit does not accept a new observation")
            token = self._pending_tokens(self._observation, len(self._context))[0]
            pending = self._pending[len(token):]
            context = (self._context + token)[-self.config.context_chars:] if self.config.context_chars else ""
            work = context + pending
            inspected = self._observation
            changed = work != self._context + self._pending
            if pending and changed:
                inspected = self._inspect(work)
            forced = not self._eof and len(token) == len(self._pending)
            self._pending, self._context = pending, context
            self._observation = inspected if pending else None
            if pending and changed:
                self._record_inspection(work)
                self._latent = np.array(inspected["context_vector"], dtype=float, copy=True)
            self._committed += len(token)
            self._reads_since_commit = 0
            self._metrics["commit_actions"] += 1
            self._metrics["forced_splits"] += int(forced)
            self._metrics["budget_forced_cuts"] += int(forced and decision.reason == "acquisition_budget")
            self._metrics["window_forced_cuts"] += int(forced and decision.reason == "window_pressure")
            self._waits[self._received - self._committed] += 1
            self._version += 1
            self._event(decision, token_chars=len(token), forced=forced, belief_reframed=bool(pending and changed))
            return {"tokens": [token], "done": False, "decision": decision}
        policy = acquisition_policy(decision.requested_chars, self.config.action_chars,
                                    initial=decision.reason == "initial_context")
        # Forecast before acquiring/inspecting the response. Both the state and
        # context are the last observed frame, never newly revealed features.
        predicted, _ = self.model.predict_next(self._latent, np.eye(self.model.latent_dim), policy, self._latent)
        if observation is None:
            if self._retry is not None:
                saved_decision, observation = self._retry
                if decision != saved_decision:
                    raise RuntimeError("An acquired observation must be processed before another action")
            elif source is None:
                raise ValueError("Acquire requires a source or a delivered observation")
            else:
                observation = source.read(decision.requested_chars, decision.observation_id)
        if (not isinstance(observation, TextObservation) or isinstance(observation.observation_id, bool)
                or not isinstance(observation.observation_id, Integral) or observation.observation_id != decision.observation_id
                or not isinstance(observation.text, str) or len(observation.text) > decision.requested_chars
                or not isinstance(observation.eof, bool) or not isinstance(observation.source, str) or not observation.source
                or (len(observation.text) < decision.requested_chars and not observation.eof)):
            raise ValueError("Source must deliver a matching bounded, nonempty-or-EOF text observation")
        if self._retry is not None and self._retry != (decision, observation):
            raise ValueError("A different observation cannot replace the retained acquired receipt")
        self._retry = (decision, observation)
        work = self._context + self._pending + observation.text
        if len(work) > self.config.max_window_chars:
            raise ValueError("Acquired text exceeds the bounded observation window")
        inspected = self._inspect(work) if observation.text else self._observation
        channel_updates = 0
        outcome_forecast = None
        if observation.text:
            inspected, channel_updates, outcome_forecast = self._filter_extension(
                inspected, work, decision.requested_chars, len(observation.text), decision.focus_index)
        before_p = decision.probability
        after_p = None
        if inspected is not None and decision.focus_index is not None:
            if decision.focus_index < len(inspected["boundary_probabilities"]):
                after_p = float(inspected["boundary_probabilities"][decision.focus_index])
        receipt, error = None, 0.0
        if observation.text:
            latent = np.asarray(inspected["context_vector"], dtype=float)
            error = float(np.linalg.norm(latent - predicted))
            receipt = self.model.observe_transition(self._latent, policy, self._latent, latent,
                                                    observation_id=observation.observation_id, source=observation.source,
                                                    train=self.config.learn_dynamics)
            self._model_observation_id = self.model.last_observation_id
            self._latent = latent.copy()
            self._surprise = error / (1.0 + error)
            self._record_inspection(work)
            self._metrics["observations"] += 1
        self._pending += observation.text
        self._observation = inspected
        self._eof = observation.eof
        self._received += len(observation.text)
        self._source_observation_id = int(observation.observation_id)
        self._reads_since_commit += 1
        self._metrics["acquire_calls"] += 1
        self._metrics["acquired_chars"] += len(observation.text)
        self._metrics["deliberate_acquire_calls"] += int(decision.reason == "modeled_value_of_context")
        selected = next((candidate for candidate in decision.candidates if candidate.action == decision.action
                         and candidate.requested_chars == decision.requested_chars), None)
        realized = _entropy(before_p) - _entropy(after_p) if before_p is not None and after_p is not None else None
        full_extension = len(observation.text) == decision.requested_chars
        matched_forecast = selected is not None and selected.supported and full_extension
        if matched_forecast:
            self._metrics["supported_decisions"] += int(selected.supported)
            self._metrics["forecast_evaluations"] += 1
            self._metrics["expected_information_gain_sum"] += selected.expected_information_gain
        if realized is not None and matched_forecast:
            self._metrics["realized_entropy_reduction_sum"] += realized
        self._metrics["truncated_acquisitions"] += int(not full_extension)
        self._metrics["channel_boundary_updates"] += channel_updates
        if outcome_forecast is not None:
            self._metrics["outcome_forecasts"] += 1
            self._metrics["outcome_nll_sum"] += outcome_forecast["outcome_nll"]
            self._metrics["outcome_brier_sum"] += outcome_forecast["outcome_brier"]
            self._metrics["regime_information_gain_sum"] += outcome_forecast["expected_regime_information_gain"]
        self._metrics["observation_world_updates"] += channel_updates if (self.observation_world is not None
                                                                         and self.config.learn_observation_regime) else 0
        self._version += 1
        self._retry = None
        self._event(decision, acquired_chars=len(observation.text), source=observation.source, eof=observation.eof,
                    input_sha256=hashlib.sha256(observation.text.encode()).hexdigest(),
                    context_sha256=_digest(np.asarray(inspected["context_vector"]).tolist()) if inspected is not None else None,
                    observed_probability=after_p, realized_entropy_reduction=realized,
                    raw_observed_probability=float(inspected["raw_boundary_probabilities"][decision.focus_index])
                    if inspected is not None and decision.focus_index is not None and decision.focus_index < len(inspected["raw_boundary_probabilities"]) else None,
                    forecast_full_extension=full_extension, matched_forecast=matched_forecast,
                    outcome_forecast=outcome_forecast,
                    channel_boundary_updates=channel_updates, prediction_error=error, observation_receipt=receipt)
        return {"tokens": [], "done": False, "decision": decision}

    def step(self, source):
        return self.execute(self.plan(), source)

    def run(self, source, *, max_steps=100000):
        limit = _integer(max_steps, "max_steps", 1)
        tokens = []
        def partial(done):
            return {"tokens": tokens.copy(), "metrics": self.metrics, "trace": self.trace,
                    "decoder": "marginal_map", "done": done}
        try:
            for _ in range(limit):
                result = self.step(source)
                tokens.extend(result["tokens"])
                if result["done"]:
                    return partial(True)
        except Exception as error:
            raise ActiveSessionError("Active run interrupted; emitted tokens are in partial_result and the acquired receipt is retained",
                                     partial(False)) from error
        return partial(False)


__all__ = ["ActiveContextConfig", "ActiveTextSession", "ContextDecision", "TextObservation", "StringTextSource",
           "ActiveSessionError", "marginal_tokens", "acquisition_policy"]
