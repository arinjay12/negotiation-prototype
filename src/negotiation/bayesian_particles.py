"""Continuous Dirichlet-particle opponent beliefs for Bayesian v1.

Only issue weights are inferred. Option values, reservation and response noise
are fixed inputs. Counteroffer preference evidence is a later research step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .domain import Offer, Scenario
from .inference import Action, Event, StandardInferenceCore


@dataclass(frozen=True)
class ParticleConfig:
    alpha: tuple[float, ...]
    count: int
    beta: float
    seed: int
    ess_threshold: float | None = None
    resample: bool = False

    def validate(self, issue_count: int) -> None:
        if len(self.alpha) != issue_count or any(not np.isfinite(a) or a <= 0 for a in self.alpha):
            raise ValueError("Dirichlet alpha must be positive for every issue")
        if self.count < 2 or not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("Particle count and beta must be positive")
        if self.ess_threshold is not None and not 0 < self.ess_threshold <= 1:
            raise ValueError("ESS threshold must be in (0, 1]")
        if self.resample and self.ess_threshold is None:
            raise ValueError("Resampling needs an ESS threshold")


class DirichletParticleOpponent:
    """Weighted particles approximate a non-conjugate posterior over weights."""

    def __init__(
        self, scenario: Scenario, option_values: Mapping[str, Mapping[str, float]],
        reservation: float, config: ParticleConfig,
    ) -> None:
        config.validate(len(scenario.issues))
        if not 0 <= reservation <= 1:
            raise ValueError("Reservation must be in [0, 1]")
        if set(option_values) != {issue.name for issue in scenario.issues}:
            raise ValueError("Fixed option-value functions must cover all issues")
        for issue in scenario.issues:
            if set(option_values[issue.name]) != set(issue.options):
                raise ValueError(f"Missing fixed values for {issue.name}")
        self.issue_names = tuple(issue.name for issue in scenario.issues)
        self.option_values = option_values
        self.reservation = reservation
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.particles = self.rng.dirichlet(np.asarray(config.alpha, dtype=float), size=config.count)
        self.weights = np.full(config.count, 1.0 / config.count, dtype=float)
        self.observation_count = 0

    def _values(self, offer: Offer) -> np.ndarray:
        if len(offer.values) != len(self.issue_names):
            raise ValueError("Offer dimension differs from particle model")
        return np.asarray([
            self.option_values[issue][choice]
            for issue, choice in zip(self.issue_names, offer.values)
        ], dtype=float)

    def _log_accept(self, offer: Offer) -> np.ndarray:
        utility = self.particles @ self._values(offer)
        z = self.config.beta * (utility - self.reservation)
        return -np.logaddexp(0.0, -z)

    def _acceptance(self, offer: Offer) -> np.ndarray:
        return np.exp(self._log_accept(offer))

    def predict_accept(self, offer: Offer, scenario: Scenario) -> float:
        if scenario.violations(offer):
            return 0.0
        return float(self.weights @ self._acceptance(offer))

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(np.square(self.weights)))

    @property
    def posterior_mean(self) -> np.ndarray:
        return self.weights @ self.particles

    @property
    def posterior_variance(self) -> np.ndarray:
        delta = self.particles - self.posterior_mean
        return self.weights @ np.square(delta)

    def summary(self) -> Mapping[str, Any]:
        return {
            "kind": "dirichlet_particles", "particle_count": self.config.count,
            "observation_count": self.observation_count,
            "ess": self.effective_sample_size,
            "posterior_mean_weights": dict(zip(self.issue_names, self.posterior_mean.tolist())),
            "posterior_weight_variance": dict(zip(self.issue_names, self.posterior_variance.tolist())),
        }

    def _systematic_resample(self) -> None:
        count = self.config.count
        positions = (self.rng.random() + np.arange(count)) / count
        indices = np.searchsorted(np.cumsum(self.weights), positions, side="left")
        self.particles = self.particles[np.minimum(indices, count - 1)].copy()
        self.weights.fill(1.0 / count)

    def observe(self, event: Event, scenario: Scenario) -> Mapping[str, Any] | None:
        if event.action not in (Action.ACCEPT, Action.REJECT) or event.offer is None:
            return None
        if scenario.violations(event.offer):
            # Infeasible offers are filtered by the controller and offer search.
            raise ValueError("Cannot update belief from an infeasible offer")
        ess_before = self.effective_sample_size
        log_accept = self._log_accept(event.offer)
        if event.action == Action.ACCEPT:
            log_likelihood = log_accept
        else:
            # log(1-sigmoid(z)) = -logaddexp(0,z)
            utility = self.particles @ self._values(event.offer)
            z = self.config.beta * (utility - self.reservation)
            log_likelihood = -np.logaddexp(0.0, z)
        log_posterior = np.log(self.weights) + log_likelihood
        log_posterior -= np.max(log_posterior)
        unnormalised = np.exp(log_posterior)
        self.weights = unnormalised / np.sum(unnormalised)
        self.observation_count += 1
        ess_after_update = self.effective_sample_size
        resampled = False
        if (self.config.resample and self.config.ess_threshold is not None
                and ess_after_update < self.config.ess_threshold * self.config.count):
            self._systematic_resample()
            resampled = True
        return {
            "action": event.action.value, "particle_count": self.config.count,
            "ess_before": ess_before, "ess_after_update": ess_after_update,
            "ess_after": self.effective_sample_size,
            "resampling_performed": resampled,
            "posterior_mean_weights": dict(zip(self.issue_names, self.posterior_mean.tolist())),
            "posterior_weight_variance": dict(zip(self.issue_names, self.posterior_variance.tolist())),
        }


def bayesian_cores(
    scenario: Scenario, configurations: Mapping[str, Mapping[str, ParticleConfig]],
) -> dict[str, StandardInferenceCore]:
    """Build separate observer/opponent filters; no true weights enter a filter.

    Configuration is indexed by observer ID then opponent ID. Fixed option
    values and reservations are drawn from the synthetic scenario; only weights
    are hidden in the filters. The controller retains them for synthetic metrics.
    """

    if scenario.structural_deadlock_check:
        raise ValueError("Bayesian runs must disable the perfect-information structural deadlock oracle")
    cores: dict[str, StandardInferenceCore] = {}
    for observer in scenario.stakeholders:
        expected = {person.id for person in scenario.stakeholders if person.id != observer.id}
        if set(configurations[observer.id]) != expected:
            raise ValueError(f"Missing opponent configuration for {observer.id}")
        beliefs = {
            opponent.id: DirichletParticleOpponent(
                scenario, opponent.utility.option_values, opponent.utility.reservation,
                configurations[observer.id][opponent.id],
            )
            for opponent in scenario.stakeholders if opponent.id != observer.id
        }
        cores[observer.id] = StandardInferenceCore(observer.id, scenario, beliefs)
    return cores


