"""Typed domain records and configuration loading.

All scenario-specific option sets, values and constraints live in data files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Issue:
    name: str
    options: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.options or len(set(self.options)) != len(self.options):
            raise ValueError(f"Invalid issue {self.name!r}")


@dataclass(frozen=True)
class Offer:
    """Values follow Scenario.issues order, making equality deterministic."""

    values: tuple[str, ...]

    def as_dict(self, issues: tuple[Issue, ...]) -> dict[str, str]:
        if len(self.values) != len(issues):
            raise ValueError("Offer and issue lengths differ")
        return {issue.name: value for issue, value in zip(issues, self.values)}


@dataclass(frozen=True)
class Constraint:
    """Declarative hard rule: forbidden value or conditional requirement."""

    id: str
    kind: str
    issue: str
    value: str
    required_issue: str | None = None
    required_values: tuple[str, ...] = ()
    rationale: str = ""

    def validate(self, issues: Mapping[str, Issue]) -> None:
        if self.issue not in issues or self.value not in issues[self.issue].options:
            raise ValueError(f"Constraint {self.id}: unknown issue or value")
        if self.kind == "requires":
            if self.required_issue not in issues or not self.required_values:
                raise ValueError(f"Constraint {self.id}: invalid requirement")
            if not set(self.required_values) <= set(issues[self.required_issue].options):
                raise ValueError(f"Constraint {self.id}: unknown required value")
        elif self.kind != "forbid":
            raise ValueError(f"Constraint {self.id}: unknown kind {self.kind}")

    def holds(self, values: Mapping[str, str]) -> bool:
        if self.kind == "forbid":
            return values[self.issue] != self.value
        return values[self.issue] != self.value or values[self.required_issue] in self.required_values


@dataclass(frozen=True)
class UtilityModel:
    weights: Mapping[str, float]
    option_values: Mapping[str, Mapping[str, float]]
    reservation: float

    def validate(self, issues: tuple[Issue, ...]) -> None:
        names = {issue.name for issue in issues}
        if set(self.weights) != names or set(self.option_values) != names:
            raise ValueError("Utility issues must exactly match scenario issues")
        if any(not 0 <= weight <= 1 for weight in self.weights.values()):
            raise ValueError("Issue weights must be in [0, 1]")
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("Issue weights must sum to one")
        for issue in issues:
            if set(self.option_values[issue.name]) != set(issue.options):
                raise ValueError(f"Option values do not cover {issue.name}")
            if any(not 0 <= value <= 1 for value in self.option_values[issue.name].values()):
                raise ValueError("Option values must be in [0, 1]")
        if not 0 <= self.reservation <= 1:
            raise ValueError("Reservation must be in [0, 1]")

    def evaluate(self, offer: Offer, issues: tuple[Issue, ...]) -> float:
        return sum(
            self.weights[issue.name] * self.option_values[issue.name][choice]
            for issue, choice in zip(issues, offer.values)
        )


@dataclass(frozen=True)
class Stakeholder:
    id: str
    goals: tuple[str, ...]
    known_facts: tuple[str, ...]
    utility: UtilityModel
    # Beliefs and negotiation memory are held by the runtime inference core.


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str


@dataclass(frozen=True)
class GraphEdge:
    source: str
    relation: str
    target: str


@dataclass(frozen=True)
class KnowledgeGraph:
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()

    def validate(self) -> None:
        ids = {node.id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("Graph node IDs must be unique")
        for edge in self.edges:
            if edge.source not in ids or edge.target not in ids:
                raise ValueError(f"Graph edge references missing node: {edge}")

    def outgoing(self, node_id: str) -> tuple[GraphEdge, ...]:
        return tuple(edge for edge in self.edges if edge.source == node_id)


@dataclass(frozen=True)
class Scenario:
    id: str
    issues: tuple[Issue, ...]
    stakeholders: tuple[Stakeholder, ...]
    system_constraints: tuple[Constraint, ...]
    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    turn_order: tuple[str, ...] = ()
    max_turns: int = 30
    offer_objective: str = "proposer_surplus_times_acceptance"
    structural_deadlock_check: bool = True

    def __post_init__(self) -> None:
        names = [issue.name for issue in self.issues]
        ids = [person.id for person in self.stakeholders]
        if not self.id or not self.issues or len(set(names)) != len(names):
            raise ValueError("Scenario requires unique issues")
        if not ids or len(set(ids)) != len(ids) or set(self.turn_order) != set(ids):
            raise ValueError("Turn order must contain each stakeholder exactly once")
        if len(self.turn_order) != len(ids) or self.max_turns < 1:
            raise ValueError("Invalid turn order or maximum turns")
        if self.offer_objective != "proposer_surplus_times_acceptance":
            raise ValueError("Unimplemented offer objective")
        issue_map = {issue.name: issue for issue in self.issues}
        system_constraint_ids = {rule.id for rule in self.system_constraints}
        if len(system_constraint_ids) != len(self.system_constraints):
            raise ValueError("Constraint IDs must be unique")
        for rule in self.system_constraints:
            rule.validate(issue_map)
        for person in self.stakeholders:
            person.utility.validate(self.issues)
        self.graph.validate()

    def offer(self, values: Mapping[str, str]) -> Offer:
        if set(values) != {issue.name for issue in self.issues}:
            raise ValueError("Offer keys must match all issues exactly")
        for issue in self.issues:
            if values[issue.name] not in issue.options:
                raise ValueError(f"Invalid option for {issue.name}")
        return Offer(tuple(values[issue.name] for issue in self.issues))

    def violations(self, offer: Offer) -> tuple[str, ...]:
        values = offer.as_dict(self.issues)
        return tuple(rule.id for rule in self.system_constraints if not rule.holds(values))

    def feasible_offers(self) -> tuple[Offer, ...]:
        return tuple(
            offer for values in product(*(issue.options for issue in self.issues))
            if not self.violations(offer := Offer(tuple(values)))
        )

    def stakeholder(self, id: str) -> Stakeholder:
        return next(person for person in self.stakeholders if person.id == id)


def load_scenario(path: str | Path) -> Scenario:
    """Load a fully declarative JSON scenario; reject malformed configurations."""

    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if "constraints" in data:
        raise ValueError("Use system_constraints for rules that apply to every offer")
    if any("constraint_ids" in item for item in data["stakeholders"]):
        raise ValueError(
            "Stakeholder constraint_ids are unsupported; define global rules in system_constraints"
        )
    issues = tuple(Issue(item["name"], tuple(item["options"])) for item in data["issues"])
    people = tuple(
        Stakeholder(
            id=item["id"],
            goals=tuple(item.get("goals", [])),
            known_facts=tuple(item.get("known_facts", [])),
            utility=UtilityModel(
                weights=item["utility"]["weights"],
                option_values=item["utility"]["option_values"],
                reservation=item["utility"]["reservation"],
            ),
        )
        for item in data["stakeholders"]
    )
    system_constraints = tuple(
        Constraint(
            id=item["id"], kind=item["kind"], issue=item["issue"],
            value=item["value"], required_issue=item.get("required_issue"),
            required_values=tuple(item.get("required_values", [])),
            rationale=item.get("rationale", ""),
        )
        for item in data["system_constraints"]
    )
    graph_data = data.get("knowledge_graph", {})
    graph = KnowledgeGraph(
        nodes=tuple(GraphNode(**item) for item in graph_data.get("nodes", [])),
        edges=tuple(GraphEdge(**item) for item in graph_data.get("edges", [])),
    )
    return Scenario(
        id=data["id"], issues=issues, stakeholders=people, system_constraints=system_constraints,
        graph=graph, turn_order=tuple(data["turn_order"]),
        max_turns=data.get("max_turns", 30),
        offer_objective=data.get("offer_objective", "proposer_surplus_times_acceptance"),
        structural_deadlock_check=data.get("structural_deadlock_check", True),
    )
