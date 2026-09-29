"""Public negotiation state shared with inference cores.

The controller retains the full synthetic scenario for diagnostics. Cores see
only this offer space and protocol configuration; utility models are passed
separately to their owners or, intentionally, to perfect-information beliefs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

from .domain import Constraint, Issue, KnowledgeGraph, Offer, Scenario


@dataclass(frozen=True)
class PublicScenario:
    id: str
    issues: tuple[Issue, ...]
    constraints: tuple[Constraint, ...]
    graph: KnowledgeGraph
    turn_order: tuple[str, ...]
    max_turns: int
    offer_objective: str

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "PublicScenario":
        return cls(
            id=scenario.id, issues=scenario.issues, constraints=scenario.constraints,
            graph=scenario.graph, turn_order=scenario.turn_order,
            max_turns=scenario.max_turns, offer_objective=scenario.offer_objective,
        )

    def offer(self, values: Mapping[str, str]) -> Offer:
        if set(values) != {issue.name for issue in self.issues}:
            raise ValueError("Offer keys must match all issues exactly")
        for issue in self.issues:
            if values[issue.name] not in issue.options:
                raise ValueError(f"Invalid option for {issue.name}")
        return Offer(tuple(values[issue.name] for issue in self.issues))

    def violations(self, offer: Offer) -> tuple[str, ...]:
        values = offer.as_dict(self.issues)
        return tuple(rule.id for rule in self.constraints if not rule.holds(values))

    def feasible_offers(self) -> tuple[Offer, ...]:
        return tuple(
            offer for values in product(*(issue.options for issue in self.issues))
            if not self.violations(offer := Offer(tuple(values)))
        )
