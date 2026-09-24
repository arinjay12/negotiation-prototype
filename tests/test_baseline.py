import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from negotiation.domain import load_scenario
from negotiation.inference import perfect_information_cores
from negotiation.protocol import NegotiationController


ROOT = Path(__file__).resolve().parents[1]


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((ROOT / "tests/fixtures/deterministic_case.json").read_text())
        self.scenario = load_scenario(ROOT / self.fixture["scenario"])

    def test_golden_agreement_and_full_trace(self):
        result = NegotiationController(self.scenario, perfect_information_cores(self.scenario)).run()
        self.assertEqual(result.status, self.fixture["expected_status"])
        self.assertEqual(len(self.scenario.feasible_offers()), self.fixture["feasible_offer_count"])
        self.assertEqual(result.agreement.as_dict(self.scenario.issues), self.fixture["expected_offer"])
        for id, expected in self.fixture["expected_utilities"].items():
            self.assertAlmostEqual(result.utilities[id], expected)
        self.assertEqual([trace.selected_action for trace in result.traces], self.fixture["expected_actions"])
        self.assertEqual([trace.agent for trace in result.traces], list(self.scenario.turn_order))
        self.assertGreater(result.traces[0].candidate_offer_count, 0)
        self.assertEqual(result.traces[0].candidate_scores[0]["offer"], self.fixture["expected_offer"])
        self.assertFalse(self.scenario.violations(result.agreement))
        json.dumps(result.as_dict(self.scenario))

    def test_structural_deadlock(self):
        # A deliberately impossible reservation combination; no agreement is forced.
        import dataclasses
        people = tuple(dataclasses.replace(p, utility=dataclasses.replace(p.utility, reservation=1.0))
                       for p in self.scenario.stakeholders)
        scenario = dataclasses.replace(self.scenario, stakeholders=people)
        result = NegotiationController(scenario, perfect_information_cores(scenario)).run()
        self.assertEqual(result.status, "deadlock")
        self.assertEqual(result.deadlock.type, "structural")
        self.assertIsNone(result.agreement)


if __name__ == "__main__":
    unittest.main()
