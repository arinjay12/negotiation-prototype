import dataclasses
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from negotiation.bayesian_config import load_particle_config
from negotiation.bayesian_particles import bayesian_cores
from negotiation.domain import load_scenario
from negotiation.inference import perfect_information_cores
from negotiation.protocol import NegotiationController


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationAndProtocolTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario(ROOT / "configs/toxic_waste.json")

    def test_bayesian_configuration_requires_oracle_disabled(self):
        configs = load_particle_config(ROOT / "configs/bayesian_particles.json", self.scenario)
        self.assertEqual(set(configs), set(self.scenario.turn_order))
        with self.assertRaisesRegex(ValueError, "structural deadlock oracle"):
            bayesian_cores(self.scenario, configs)

    def test_turn_limit_produces_procedural_deadlock(self):
        scenario = dataclasses.replace(self.scenario, max_turns=1)
        result = NegotiationController(scenario, perfect_information_cores(scenario)).run()
        self.assertEqual(result.status, "deadlock")
        self.assertEqual(result.deadlock.type, "procedural")
        self.assertEqual(len(result.traces), 1)
        self.assertEqual(result.traces[0].selected_action, "PROPOSE")


if __name__ == "__main__":
    unittest.main()
