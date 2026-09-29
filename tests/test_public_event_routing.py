"""A public event updates only the belief about its actor."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from negotiation.bayesian_particles import DirichletParticleOpponent, ParticleConfig, bayesian_cores
from negotiation.inference import Action, Event, NegotiationState
from negotiation.public import PublicScenario
from test_multiagent_integration import three_agent_case


class PublicEventRoutingTests(unittest.TestCase):
    def test_actor_specific_updates_and_private_view_guard(self):
        scenario, offers = three_agent_case()
        public = PublicScenario.from_scenario(scenario)
        config = ParticleConfig((1, 1), 2000, 12, 42)
        with self.assertRaises(TypeError):
            DirichletParticleOpponent(scenario, scenario.stakeholder("bob").utility.option_values,
                                      .55, config)
        configs = {
            observer: {opponent: ParticleConfig((1, 1), 2000, 12, seed)
                       for seed, opponent in enumerate((name for name in scenario.turn_order if name != observer), 1)}
            for observer in scenario.turn_order
        }
        cores = bayesian_cores(scenario, configs)
        state = NegotiationState(public, public.feasible_offers(), offers["initial"],
                                 frozenset(), frozenset(), 1, None)
        rejection = Event("bob", Action.REJECT, offers["initial"], 1)
        cores["alice"].observe(rejection, state)
        cores["carl"].observe(rejection, state)
        self.assertEqual(cores["alice"].beliefs["bob"].observation_count, 1)
        self.assertEqual(cores["carl"].beliefs["bob"].observation_count, 1)
        self.assertEqual(cores["alice"].beliefs["carl"].observation_count, 0)
        self.assertEqual(cores["bob"].beliefs["alice"].observation_count, 0)
        acceptance = Event("carl", Action.ACCEPT, offers["costly_fast"], 2)
        cores["alice"].observe(acceptance, state)
        cores["bob"].observe(acceptance, state)
        self.assertEqual(cores["alice"].beliefs["carl"].observation_count, 1)
        self.assertEqual(cores["bob"].beliefs["carl"].observation_count, 1)
        proposal = Event("alice", Action.PROPOSE, offers["cheap_fast"], 3)
        cores["bob"].observe(proposal, state)
        cores["carl"].observe(proposal, state)
        self.assertEqual(cores["bob"].beliefs["alice"].observation_count, 0)
        self.assertEqual(cores["carl"].beliefs["alice"].observation_count, 0)


if __name__ == "__main__":
    unittest.main()
