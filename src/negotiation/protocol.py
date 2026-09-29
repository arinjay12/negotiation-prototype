"""Inference-independent controller, deadlock outcomes, and decision traces."""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Mapping

from .domain import Offer, Scenario
from .inference import Action, Decision, Event, InferenceCore, NegotiationState
from .public import PublicScenario


@dataclass(frozen=True)
class TurnTrace:
    turn: int
    agent: str
    observed_events: tuple[Mapping[str, Any], ...]
    current_offer: Mapping[str, str] | None
    own_utility: float | None
    opponent_beliefs_before: Mapping[str, Any]
    evidence: tuple[Any, ...]
    posterior_update: tuple[Any, ...]
    opponent_beliefs_after: Mapping[str, Any]
    broadcast_effects: Mapping[str, Any]
    candidate_offer_count: int
    candidate_offers_considered: tuple[Mapping[str, str], ...]
    candidate_scores: tuple[Mapping[str, Any], ...]
    predicted_acceptance: Mapping[str, float]
    selected_action: str
    selected_offer: Mapping[str, str] | None
    decision_reason: str


@dataclass(frozen=True)
class Deadlock:
    type: str
    blocking_conditions: tuple[str, ...]
    possible_unlocks: tuple[str, ...]
    human_decisions_required: tuple[str, ...]


@dataclass(frozen=True)
class NegotiationResult:
    status: str
    agreement: Offer | None
    utilities: Mapping[str, float] | None
    nash_surplus: float | None
    deadlock: Deadlock | None
    traces: tuple[TurnTrace, ...]

    def as_dict(self, scenario: Scenario) -> dict[str, Any]:
        return {
            "status": self.status,
            "agreement": self.agreement.as_dict(scenario.issues) if self.agreement else None,
            "utilities": self.utilities,
            "nash_surplus": self.nash_surplus,
            "deadlock": asdict(self.deadlock) if self.deadlock else None,
            "traces": [asdict(trace) for trace in self.traces],
        }


class NegotiationController:
    """Rotating protocol; knows only the InferenceCore interface."""

    def __init__(self, scenario: Scenario, cores: Mapping[str, InferenceCore]) -> None:
        if set(cores) != set(scenario.turn_order):
            raise ValueError("One inference core is required per stakeholder")
        self.scenario = scenario
        self.public_scenario = PublicScenario.from_scenario(scenario)
        self.cores = dict(cores)

    def _state(self, feasible, current, proposed, accepted, turn, latest):
        return NegotiationState(self.public_scenario, feasible, current,
                                frozenset(proposed), frozenset(accepted), turn, latest)

    def _structural_deadlock(self, feasible: tuple[Offer, ...]) -> bool:
        if not self.scenario.structural_deadlock_check:
            return False
        state = self._state(feasible, None, set(), set(), 0, None)
        return not any(
            all(core.evaluate_offer(offer, state).acceptable for core in self.cores.values())
            for offer in feasible
        )

    def run(self) -> NegotiationResult:
        feasible = self.public_scenario.feasible_offers()
        if self._structural_deadlock(feasible):
            return NegotiationResult(
                "deadlock", None, None, None,
                Deadlock(
                    "structural",
                    ("No configured feasible offer reaches every stakeholder's reservation value.",),
                    ("Review configured constraints and reservation values with stakeholders.",),
                    ("Decide whether any constraint or reservation value can change.",),
                ), (),
            )
        current: Offer | None = None
        accepted: set[str] = set()
        proposed: set[Offer] = set()
        traces: list[TurnTrace] = []
        latest: Event | None = None
        for turn in range(1, self.scenario.max_turns + 1):
            actor = self.scenario.turn_order[(turn - 1) % len(self.scenario.turn_order)]
            core = self.cores[actor]
            state = self._state(feasible, current, proposed, accepted, turn, latest)
            observed = core.drain_observations()
            beliefs_after = dict(core.belief_summaries())
            beliefs_before = dict(beliefs_after)
            for item in observed:
                beliefs_before[item["event"]["actor"]] = item["belief_before"]
            evaluation = core.evaluate_offer(current, state) if current else None
            decision = core.choose_action(state)
            self._validate_decision(decision, state, actor)
            selected = decision.offer.as_dict(self.scenario.issues) if decision.offer else None
            candidate_scores = tuple({
                "offer": candidate.offer.as_dict(self.scenario.issues),
                "own_utility": candidate.own_utility,
                "own_surplus": candidate.own_surplus,
                "opponent_acceptance": dict(candidate.opponent_acceptance),
                "score": candidate.score,
            } for candidate in decision.candidates)
            traces.append(TurnTrace(
                turn=turn, agent=actor, observed_events=observed,
                current_offer=current.as_dict(self.scenario.issues) if current else None,
                own_utility=evaluation.utility if evaluation else None,
                opponent_beliefs_before=beliefs_before,
                evidence=tuple(item["evidence"] for item in observed),
                posterior_update=tuple(item["evidence"] for item in observed if item["evidence"] is not None),
                opponent_beliefs_after=beliefs_after, broadcast_effects={},
                candidate_offer_count=len(decision.candidates),
                candidate_offers_considered=tuple(candidate.offer.as_dict(self.scenario.issues) for candidate in decision.candidates),
                candidate_scores=candidate_scores,
                predicted_acceptance=dict(decision.candidates[0].opponent_acceptance) if decision.candidates else {},
                selected_action=decision.action.value, selected_offer=selected,
                decision_reason=decision.reason,
            ))
            event_offer = decision.offer
            if decision.action in (Action.PROPOSE, Action.COUNTER):
                current = decision.offer
                proposed.add(current)
                accepted = {actor}
            elif decision.action == Action.ACCEPT:
                accepted.add(actor)
            elif decision.action == Action.REJECT:
                current = None
                accepted.clear()
            else:
                return NegotiationResult(
                    "deadlock", None, None, None,
                    Deadlock("strategic", ("No untried acceptable proposal remains for the acting agent.",),
                             ("Review reservations, preferences, or available options.",),
                             ("Decide whether to change the scenario or accept deadlock.",)),
                    tuple(traces),
                )
            latest = Event(actor, decision.action, event_offer, turn)
            # Broadcast public events immediately; record every observer's effect,
            # including updates after the final acceptance.
            effects = {}
            for observer_id, observer in self.cores.items():
                if observer_id != actor:
                    effect = observer.observe(
                        latest, self._state(feasible, current, proposed, accepted, turn, latest)
                    )
                    if effect is not None:
                        effects[observer_id] = effect
            traces[-1] = replace(traces[-1], broadcast_effects=effects)
            if current is not None and accepted == set(self.scenario.turn_order):
                utilities = {person.id: person.utility.evaluate(current, self.scenario.issues)
                             for person in self.scenario.stakeholders}
                nash = 1.0
                for person in self.scenario.stakeholders:
                    nash *= max(0.0, utilities[person.id] - person.utility.reservation)
                return NegotiationResult("agreement", current, utilities, nash, None, tuple(traces))
        return NegotiationResult(
            "deadlock", None, None, None,
            Deadlock("procedural", (f"Maximum of {self.scenario.max_turns} turns reached.",),
                     ("Review proposal history and consider an amended scenario.",),
                     ("Decide whether to extend the turn limit or change constraints.",)),
            tuple(traces),
        )

    def _validate_decision(self, decision: Decision, state: NegotiationState, actor: str) -> None:
        if decision.action in (Action.PROPOSE, Action.COUNTER):
            if decision.offer not in state.feasible_offers or decision.offer in state.proposed_offers:
                raise ValueError("Core proposed an infeasible or repeated offer")
            if not self.cores[actor].evaluate_offer(decision.offer, state).acceptable:
                raise ValueError("Core proposed an offer below its own reservation")
            expected = Action.PROPOSE if state.current_offer is None else Action.COUNTER
            if decision.action != expected:
                raise ValueError("Invalid propose/counter transition")
        elif decision.action == Action.ACCEPT:
            if state.current_offer is None or decision.offer != state.current_offer:
                raise ValueError("Can accept only the current offer")
            if not self.cores[actor].evaluate_offer(decision.offer, state).acceptable:
                raise ValueError("Core accepted an infeasible or unacceptable offer")
        elif decision.action == Action.REJECT:
            if state.current_offer is None or decision.offer != state.current_offer:
                raise ValueError("Can reject only the current offer")
        elif decision.action == Action.DECLARE_DEADLOCK:
            if decision.offer is not None:
                raise ValueError("Deadlock has no offer")
        else:
            raise ValueError("Unknown action")
