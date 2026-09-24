import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from negotiation.domain import load_scenario


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "toxic_waste.json"


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario(CONFIG)

    def test_offer_validation_and_feasible_count(self):
        self.assertEqual(len(self.scenario.feasible_offers()), 162)
        first = self.scenario.feasible_offers()[0]
        self.assertEqual(first.as_dict(self.scenario.issues)["handler"], "specialist")
        self.assertTrue(all(not self.scenario.violations(x) for x in self.scenario.feasible_offers()))
        with self.assertRaises(ValueError):
            self.scenario.offer({"handler": "specialist"})

    def test_constraints_and_graph(self):
        values = {issue.name: issue.options[0] for issue in self.scenario.issues}
        offer = self.scenario.offer(values)
        self.assertEqual(set(self.scenario.violations(offer)), {"synthetic_specialist_only", "synthetic_documentation_required"})
        self.assertEqual(self.scenario.graph.outgoing("bob")[0].relation, "REPRESENTS")

    def test_configuration_rejects_invalid_weights(self):
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        data["stakeholders"][0]["utility"]["weights"]["handler"] = 0.4
        path = Path(__file__).resolve().parents[1] / "work" / "invalid_scenario.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_scenario(path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
