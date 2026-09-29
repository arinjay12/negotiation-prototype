"""Independent checks of the configured baseline equations and boundary rule."""

import dataclasses
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from negotiation.domain import load_scenario
from negotiation.inference import ExhaustiveSearch, KnownOpponent, NegotiationState, perfect_information_cores
from negotiation.public import PublicScenario


ROOT = Path(__file__).resolve().parents[1]


class BaselineMathReviewTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario(ROOT / "configs/toxic_waste.json")
        self.offer = self.scenario.offer({
            "handler": "specialist", "timing": "24h", "cost_responsibility": "split",
            "employment_protection": "written", "work_pay_protection": "full",
            "documentation": "full",
        })

    def test_hand_calculated_utilities_and_score(self):
        alice = self.scenario.stakeholder("alice").utility
        bob = self.scenario.stakeholder("bob").utility
        carl = self.scenario.stakeholder("carl").utility
        self.assertAlmostEqual(alice.evaluate(self.offer, self.scenario.issues),
                               .35 + .05 + .02 * .8 + .30 + .20 + .08)
        self.assertAlmostEqual(bob.evaluate(self.offer, self.scenario.issues),
                               .35 + .05 + .05 * .7 + .25 + .10 + .20)
        self.assertAlmostEqual(carl.evaluate(self.offer, self.scenario.issues),
                               .15 * .4 + .30 * .7 + .30 * .7 + .05 * .4 + .10 * 0 + .10 * .4)
        beliefs = {"bob": KnownOpponent(bob), "carl": KnownOpponent(carl)}
        public = PublicScenario.from_scenario(self.scenario)
        candidates = ExhaustiveSearch().rank(public, alice, beliefs,
                                             public.feasible_offers(), frozenset())
        self.assertEqual(candidates[0].offer, self.offer)
        self.assertAlmostEqual(candidates[0].score, .996 - .65)

    def test_below_reservation_is_rejected_even_with_small_gap(self):
        alice = self.scenario.stakeholder("alice")
        value = alice.utility.evaluate(self.offer, self.scenario.issues)
        stricter = dataclasses.replace(alice, utility=dataclasses.replace(alice.utility,
                                                                          reservation=value + 5e-13))
        scenario = dataclasses.replace(self.scenario, stakeholders=(stricter,) + self.scenario.stakeholders[1:])
        core = perfect_information_cores(scenario)["alice"]
        public = PublicScenario.from_scenario(scenario)
        state = NegotiationState(public, public.feasible_offers(), self.offer,
                                 frozenset(), frozenset(), 1, None)
        self.assertFalse(core.evaluate_offer(self.offer, state).acceptable)
        self.assertEqual(ExhaustiveSearch().rank(public, stricter.utility, core.beliefs,
                                                (self.offer,), frozenset()), ())


if __name__ == "__main__":
    unittest.main()
