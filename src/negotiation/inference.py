"""Common inference interface and deterministic standard implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

from .domain import Offer, Scenario, UtilityModel


class Action(str, Enum):
    PROPOSE = "PROPOSE"
    COUNTER = "COUNTER"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    DECLARE_DEADLOCK = "DECLARE_DEADLOCK"


@dataclass(frozen=True)
class Event:
    actor: str
    action: Action
    offer: Offer | None
    turn: int


@dataclass(frozen=True)
class NegotiationState:
    scenario: Scenario
    feasible_offers: tuple[Offer, ...]
    current_offer: Offer | None
    proposed_offers: frozenset[Offer]
    accepted_by: frozenset[str]
    turn: int
    latest_event: Event | None


@dataclass(frozen=True)
class OfferEvaluation:
    utility: float
    reservation: float
    acceptable: bool
    constraint_violations: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    offer: Offer
    own_utility: float
    own_surplus: float
    opponent_acceptance: Mapping[str, float]
    score: float


@dataclass(frozen=True)
class Decision:
    action: Action
    offer: Offer | None
    reason: str
    candidates: tuple[Candidate, ...] = ()


class OpponentBelief(Protocol):
    def predict_accept(self, offer: Offer, scenario: Scenario) -> float: ...
    def summary(self) -> Mapping[str, Any]: ...
    def observe(self, event: Event, scenario: Scenario) -> Mapping[str, Any] | None: ...


class KnownOpponent:
    """Perfect-information baseline. The utility is private to the core."""

    def __init__(self, utility: UtilityModel):
        self.utility = utility

    def predict_accept(self, offer: Offer, scenario: Scenario) -> float:
        return float(
            not scenario.violations(offer)
            and self.utility.evaluate(offer, scenario.issues) >= self.utility.reservation
        )

    def summary(self) -> Mapping[str, Any]:
        return {"kind": "perfect_information", "reservation": self.utility.reservation}

    def observe(self, event: Event, scenario: Scenario) -> None:
        return None


class InferenceCore(ABC):
    @abstractmethod
    def observe(self, event: Event, state: NegotiationState) -> None:
        """Process only information visible in the public event stream."""

    @abstractmethod
    def evaluate_offer(self, offer: Offer, state: NegotiationState) -> OfferEvaluation:
        """Evaluate own response to an offer with a structured result."""

    @abstractmethod
    def choose_action(self, state: NegotiationState) -> Decision:
        """Return an explicit protocol action."""

    @abstractmethod
    def drain_observations(self) -> tuple[Mapping[str, Any], ...]:
        """Return trace records since this core's preceding turn."""

    @abstractmethod
    def belief_summaries(self) -> Mapping[str, Mapping[str, Any]]:
        """Return inspectable opponent belief summaries."""


class OfferSearch(ABC):
    @abstractmethod
    def rank(
        self, scenario: Scenario, own_utility: UtilityModel,
        beliefs: Mapping[str, OpponentBelief], feasible: tuple[Offer, ...],
        excluded: frozenset[Offer],
    ) -> tuple[Candidate, ...]: ...


class ExhaustiveSearch(OfferSearch):
    """Enumerate every feasible offer; stable ties follow configured option order."""

    def rank(self, scenario, own_utility, beliefs, feasible, excluded):
        candidates: list[Candidate] = []
        for offer in feasible:
            if offer in excluded:
                continue
            own = own_utility.evaluate(offer, scenario.issues)
            surplus = own - own_utility.reservation
            if surplus < -1e-12:
                continue
            probabilities = {id: belief.predict_accept(offer, scenario) for id, belief in beliefs.items()}
            probability_product = 1.0
            for probability in probabilities.values():
                probability_product *= probability
            candidates.append(Candidate(offer, own, max(0.0, surplus), probabilities, max(0.0, surplus) * probability_product))
        # Python's stable sort preserves exhaustive enumeration order on exact ties.
        return tuple(sorted(candidates, key=lambda item: -item.score))


class StandardInferenceCore(InferenceCore):
    """Own utility is known; opponent beliefs can be exact or Bayesian."""

    def __init__(
        self, stakeholder_id: str, scenario: Scenario,
        beliefs: Mapping[str, OpponentBelief], search: OfferSearch | None = None,
    ) -> None:
        self.id = stakeholder_id
        self.scenario = scenario
        self.own_utility = scenario.stakeholder(stakeholder_id).utility
        if set(beliefs) != {person.id for person in scenario.stakeholders} - {stakeholder_id}:
            raise ValueError("One opponent belief is required per other stakeholder")
        self.beliefs = dict(beliefs)
        self.search = search or ExhaustiveSearch()
        self._observations: list[Mapping[str, Any]] = []

    def observe(self, event: Event, state: NegotiationState) -> None:
        if event.actor == self.id:
            return
        belief = self.beliefs[event.actor]
        before = dict(belief.summary())
        update = belief.observe(event, state.scenario)
        self._observations.append({
            "event": {"actor": event.actor, "action": event.action.value,
                      "offer": event.offer.as_dict(state.scenario.issues) if event.offer else None,
                      "turn": event.turn},
            "belief_before": before,
            "evidence": update,
            "belief_after": dict(belief.summary()),
        })

    def evaluate_offer(self, offer: Offer, state: NegotiationState) -> OfferEvaluation:
        violations = state.scenario.violations(offer)
        utility = self.own_utility.evaluate(offer, state.scenario.issues)
        return OfferEvaluation(utility, self.own_utility.reservation,
                               not violations and utility >= self.own_utility.reservation - 1e-12,
                               violations)

    def choose_action(self, state: NegotiationState) -> Decision:
        if state.current_offer is not None:
            evaluation = self.evaluate_offer(state.current_offer, state)
            if evaluation.acceptable:
                return Decision(Action.ACCEPT, state.current_offer,
                                f"Own utility {evaluation.utility:.6f} reaches reservation {evaluation.reservation:.6f}.")
        candidates = self.search.rank(state.scenario, self.own_utility, self.beliefs,
                                      state.feasible_offers, state.proposed_offers)
        if candidates:
            selected = candidates[0]
            action = Action.PROPOSE if state.current_offer is None else Action.COUNTER
            return Decision(action, selected.offer,
                            "Highest configured proposer-surplus times posterior-predictive acceptance score; stable option-order tie break.",
                            candidates)
        if state.current_offer is not None:
            return Decision(Action.REJECT, state.current_offer,
                            "Current offer is below reservation and no untried acceptable offer remains.")
        return Decision(Action.DECLARE_DEADLOCK, None, "No untried acceptable offer remains.")

    def drain_observations(self) -> tuple[Mapping[str, Any], ...]:
        events = tuple(self._observations)
        self._observations.clear()
        return events

    def belief_summaries(self) -> Mapping[str, Mapping[str, Any]]:
        return {id: dict(belief.summary()) for id, belief in self.beliefs.items()}


def perfect_information_cores(scenario: Scenario) -> dict[str, StandardInferenceCore]:
    return {
        person.id: StandardInferenceCore(
            person.id, scenario,
            {other.id: KnownOpponent(other.utility) for other in scenario.stakeholders if other.id != person.id},
        )
        for person in scenario.stakeholders
    }

