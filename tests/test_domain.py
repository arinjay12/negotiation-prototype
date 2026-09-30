import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from negotiation.domain import load_scenario
from negotiation.public import PublicScenario


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
        public = PublicScenario.from_scenario(self.scenario)
        self.assertEqual(public.system_constraints, self.scenario.system_constraints)
        self.assertEqual(public.violations(offer), self.scenario.violations(offer))
        self.assertEqual(self.scenario.graph.outgoing("bob")[0].relation, "REPRESENTS")

    def test_ambiguous_constraint_fields_are_rejected(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        path = Path(__file__).resolve().parents[1] / "work" / "invalid_scenario.json"
        path.parent.mkdir(exist_ok=True)
        try:
            for field in ("constraints", "constraint_ids"):
                with self.subTest(field=field):
                    data = json.loads(json.dumps(base))
                    if field == "constraints":
                        data["constraints"] = data.pop("system_constraints")
                    else:
                        data["stakeholders"][0]["constraint_ids"] = ["synthetic_specialist_only"]
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, field):
                        load_scenario(path)
            data = json.loads(json.dumps(base))
            del data["system_constraints"]
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(KeyError):
                load_scenario(path)
        finally:
            path.unlink(missing_ok=True)

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
