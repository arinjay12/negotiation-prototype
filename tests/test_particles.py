import json
import dataclasses
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from negotiation.bayesian_particles import ParticleConfig, DirichletParticleOpponent, bayesian_cores
from negotiation.domain import Issue, Offer, Scenario, Stakeholder, UtilityModel, load_scenario
from negotiation.inference import Action, Event, ExhaustiveSearch
from negotiation.protocol import NegotiationController
from negotiation.public import PublicScenario


ROOT = Path(__file__).resolve().parents[1]


def synthetic_two_issue_case():
    issues = (Issue("speed", ("slow", "fast")), Issue("cost", ("costly", "cheap")))
    own = UtilityModel(
        {"speed": 0.5, "cost": 0.5},
        {"speed": {"slow": 1.0, "fast": 1.0}, "cost": {"costly": 0.2, "cheap": 0.8}},
        0.1,
    )
    # The test simulator owns the hidden true weight vector separately.
    opponent = UtilityModel(
        {"speed": 0.5, "cost": 0.5},
        {"speed": {"slow": 0.0, "fast": 1.0}, "cost": {"costly": 0.0, "cheap": 1.0}},
        0.55,
    )
    scenario = Scenario("two_issue_synthetic", issues,
                        (Stakeholder("proposer", (), (), own), Stakeholder("opponent", (), (), opponent)),
                        (), turn_order=("proposer", "opponent"))
    fast_costly = scenario.offer({"speed": "fast", "cost": "costly"})
    slow_cheap = scenario.offer({"speed": "slow", "cost": "cheap"})
    return scenario, own, opponent, fast_costly, slow_cheap


class ParticleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((ROOT / "tests/fixtures/bayesian_case.json").read_text())
        self.scenario, self.own, self.opponent, self.fast_costly, self.slow_cheap = synthetic_two_issue_case()
        self.public = PublicScenario.from_scenario(self.scenario)
        self.config = ParticleConfig(tuple(self.fixture["alpha"]), self.fixture["particle_count"],
                                     self.fixture["beta"], self.fixture["seed"])

    def make_model(self):
        return DirichletParticleOpponent(self.public, self.opponent.option_values,
                                         self.fixture["reservation"], self.config)

    def test_prior_sampling_normalisation_and_seed(self):
        first, second = self.make_model(), self.make_model()
        np.testing.assert_array_equal(first.particles, second.particles)
        np.testing.assert_array_equal(first.weights, second.weights)
        np.testing.assert_allclose(first.particles.sum(axis=1), 1.0)
        self.assertAlmostEqual(first.weights.sum(), 1.0)
        self.assertAlmostEqual(first.effective_sample_size, self.config.count)

    def test_hidden_synthetic_behaviour_changes_prediction_and_offer_ranking(self):
        model = self.make_model()
        offers = (self.fast_costly, self.slow_cheap)
        search = ExhaustiveSearch()
        before = search.rank(self.public, self.own, {"opponent": model}, offers, frozenset())
        self.assertEqual(before[0].offer, self.slow_cheap)
        prior_fast = model.predict_accept(self.fast_costly, self.public)
        prior_slow = model.predict_accept(self.slow_cheap, self.public)
        hidden = np.asarray(self.fixture["hidden_true_weights"])
        rng = np.random.default_rng(self.fixture["observation_seed"])
        observed = []
        for turn in range(self.fixture["repetitions_per_offer"]):
            for offer in offers:
                values = model._values(offer)
                true_probability = 1 / (1 + np.exp(-self.config.beta * (hidden @ values - self.fixture["reservation"])))
                action = Action.ACCEPT if rng.random() < true_probability else Action.REJECT
                observed.append(action)
                update = model.observe(Event("opponent", action, offer, turn), self.public)
                self.assertAlmostEqual(model.weights.sum(), 1.0)
                self.assertGreater(update["ess_after"], 0)
        self.assertIn(Action.ACCEPT, observed)
        self.assertIn(Action.REJECT, observed)
        self.assertGreater(model.posterior_mean[0], 0.7)
        self.assertGreater(model.predict_accept(self.fast_costly, self.public), prior_fast)
        self.assertLess(model.predict_accept(self.slow_cheap, self.public), prior_slow)
        after = search.rank(self.public, self.own, {"opponent": model}, offers, frozenset())
        self.assertEqual(after[0].offer, self.fast_costly)
        self.assertEqual(model.observation_count, 2 * self.fixture["repetitions_per_offer"])

    def test_systematic_resampling_and_ignored_proposal(self):
        config = ParticleConfig((1, 1), 2000, 12, 3, ess_threshold=0.99, resample=True)
        model = DirichletParticleOpponent(self.public, self.opponent.option_values, 0.55, config)
        self.assertIsNone(model.observe(Event("opponent", Action.PROPOSE, self.fast_costly, 1), self.public))
        self.assertEqual(model.observation_count, 0)
        update = model.observe(Event("opponent", Action.ACCEPT, self.fast_costly, 2), self.public)
        self.assertTrue(update["resampling_performed"])
        self.assertAlmostEqual(model.effective_sample_size, 2000)
        self.assertAlmostEqual(model.weights.sum(), 1.0)

    def test_three_agent_wiring_and_controller_interface(self):
        scenario = dataclasses.replace(load_scenario(ROOT / "configs/toxic_waste.json"), structural_deadlock_check=False)
        configs = {
            observer.id: {
                opponent.id: ParticleConfig((1.0,) * len(scenario.issues), 2000, 12.0,
                                            seed=100 + i * 3 + j)
                for j, opponent in enumerate(scenario.stakeholders) if opponent.id != observer.id
            }
            for i, observer in enumerate(scenario.stakeholders)
        }
        cores = bayesian_cores(scenario, configs)
        self.assertIsNot(cores["alice"].beliefs["bob"], cores["carl"].beliefs["bob"])
        result = NegotiationController(scenario, cores).run()
        self.assertIn(result.status, ("agreement", "deadlock"))
        for trace in result.traces:
            if trace.selected_action in ("PROPOSE", "COUNTER"):
                self.assertFalse(scenario.violations(scenario.offer(trace.selected_offer)))
        json.dumps(result.as_dict(scenario))


if __name__ == "__main__":
    unittest.main()


