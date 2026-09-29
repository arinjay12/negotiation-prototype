"""End-to-end synthetic evidence flow through a three-agent negotiation."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from negotiation.bayesian_particles import ParticleConfig, bayesian_cores
from negotiation.domain import Issue, Offer, Scenario, Stakeholder, UtilityModel
from negotiation.inference import Action, Decision, Event, InferenceCore, NegotiationState, OfferEvaluation
from negotiation.protocol import NegotiationController
from negotiation.public import PublicScenario


class ScriptedResponder(InferenceCore):
    """Deterministic synthetic response from a hidden configured utility."""

    def __init__(self, utility: UtilityModel, initial_offer: Offer | None = None):
        self.utility = utility
        self.initial_offer = initial_offer

    def observe(self, event: Event, state: NegotiationState) -> None:
        assert isinstance(state.scenario, PublicScenario)

    def evaluate_offer(self, offer: Offer, state: NegotiationState) -> OfferEvaluation:
        assert isinstance(state.scenario, PublicScenario)
        value = self.utility.evaluate(offer, state.scenario.issues)
        violations = state.scenario.violations(offer)
        return OfferEvaluation(value, self.utility.reservation,
                               not violations and value >= self.utility.reservation,
                               violations)

    def choose_action(self, state: NegotiationState) -> Decision:
        assert isinstance(state.scenario, PublicScenario)
        if state.current_offer is None:
            if self.initial_offer is None:
                return Decision(Action.DECLARE_DEADLOCK, None, "No scripted proposal")
            return Decision(Action.PROPOSE, self.initial_offer, "Synthetic scripted proposal")
        if self.evaluate_offer(state.current_offer, state).acceptable:
            return Decision(Action.ACCEPT, state.current_offer, "Synthetic utility reaches reservation")
        return Decision(Action.REJECT, state.current_offer, "Synthetic utility is below reservation")

    def drain_observations(self):
        return ()

    def belief_summaries(self):
        return {}


def three_agent_case():
    issues = (Issue("speed", ("slow", "fast", "delayed")),
              Issue("cost", ("cheap", "costly")))
    alice = Stakeholder("alice", (), (), UtilityModel(
        {"speed": .8, "cost": .2},
        {"speed": {"slow": 1, "fast": .4, "delayed": 0},
         "cost": {"cheap": .7, "costly": 1}}, .4))
    bob = Stakeholder("bob", (), (), UtilityModel(
        {"speed": .9, "cost": .1},
        {"speed": {"slow": 0, "fast": 1, "delayed": .2},
         "cost": {"cheap": 1, "costly": 0}}, .55))
    carl = Stakeholder("carl", (), (), UtilityModel(
        {"speed": .5, "cost": .5},
        {"speed": {"slow": 1, "fast": 1, "delayed": 1},
         "cost": {"cheap": 1, "costly": 1}}, .7))
    scenario = Scenario("multi_round_synthetic", issues, (alice, bob, carl), (),
                        turn_order=("alice", "bob", "carl"), max_turns=6,
                        structural_deadlock_check=False)
    offers = {
        "initial": scenario.offer({"speed": "slow", "cost": "cheap"}),
        "cheap_fast": scenario.offer({"speed": "fast", "cost": "cheap"}),
        "costly_fast": scenario.offer({"speed": "fast", "cost": "costly"}),
        "carl_proposal": scenario.offer({"speed": "delayed", "cost": "costly"}),
    }
    return scenario, offers


class MultiAgentIntegrationTests(unittest.TestCase):
    def test_public_view_and_separate_opponent_models(self):
        scenario, offers = three_agent_case()
        configs = {
            observer.id: {
                opponent.id: ParticleConfig((1, 1), 2000, 12, 100 + 3 * i + j)
                for j, opponent in enumerate(scenario.stakeholders) if opponent.id != observer.id
            }
            for i, observer in enumerate(scenario.stakeholders)
        }
        cores = bayesian_cores(scenario, configs)
        alice = cores["alice"]
        self.assertIsInstance(alice.scenario, PublicScenario)
        self.assertFalse(hasattr(alice.scenario, "stakeholders"))
        self.assertIsNot(alice.beliefs["bob"], alice.beliefs["carl"])
        self.assertIsNot(alice.beliefs["bob"], cores["carl"].beliefs["bob"])
        wrong_state = NegotiationState(scenario, scenario.feasible_offers(), offers["initial"],
                                       frozenset(), frozenset(), 1, None)
        with self.assertRaises(TypeError):
            alice.evaluate_offer(offers["initial"], wrong_state)

    def test_observed_rejection_changes_later_counteroffer(self):
        scenario, offers = three_agent_case()
        configs = {
            observer.id: {
                opponent.id: ParticleConfig((1, 1), 3000, 12, 200 + 3 * i + j)
                for j, opponent in enumerate(scenario.stakeholders) if opponent.id != observer.id
            }
            for i, observer in enumerate(scenario.stakeholders)
        }
        cores = bayesian_cores(scenario, configs)
        alice = cores["alice"]
        cores["bob"] = ScriptedResponder(scenario.stakeholder("bob").utility)
        cores["carl"] = ScriptedResponder(scenario.stakeholder("carl").utility,
                                          offers["carl_proposal"])
        result = NegotiationController(scenario, cores).run()
        self.assertEqual(result.status, "agreement")
        self.assertEqual(result.agreement, offers["costly_fast"])
        self.assertEqual([t.selected_action for t in result.traces],
                         ["PROPOSE", "REJECT", "PROPOSE", "COUNTER", "ACCEPT", "ACCEPT"])
        self.assertEqual(result.traces[0].selected_offer, offers["initial"].as_dict(scenario.issues))
        first_scores = {tuple(item["offer"].values()): item["score"]
                        for item in result.traces[0].candidate_scores}
        later_scores = {tuple(item["offer"].values()): item["score"]
                        for item in result.traces[3].candidate_scores}
        cheap = offers["cheap_fast"].values
        costly = offers["costly_fast"].values
        self.assertGreater(first_scores[cheap], first_scores[costly])
        self.assertGreater(later_scores[costly], later_scores[cheap])
        update = result.traces[3].posterior_update
        self.assertEqual(len(update), 1)
        self.assertEqual(update[0]["action"], "REJECT")
        initial_mean = result.traces[0].opponent_beliefs_after["bob"]["posterior_mean_weights"]["speed"]
        later_mean = result.traces[3].opponent_beliefs_after["bob"]["posterior_mean_weights"]["speed"]
        self.assertGreater(later_mean, initial_mean)
        self.assertEqual(alice.beliefs["bob"].observation_count, 2)
        self.assertEqual(alice.beliefs["carl"].observation_count, 1)
        for trace in result.traces:
            if trace.selected_action in ("PROPOSE", "COUNTER"):
                self.assertFalse(scenario.violations(scenario.offer(trace.selected_offer)))
        self.assertEqual(result.traces[1].broadcast_effects["alice"]["evidence"]["action"], "REJECT")
        self.assertEqual(result.traces[-1].broadcast_effects["alice"]["evidence"]["action"], "ACCEPT")


if __name__ == "__main__":
    unittest.main()
