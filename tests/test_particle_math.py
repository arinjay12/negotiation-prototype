"""Hand-computed particle likelihood checks, separate from synthetic behaviour."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from negotiation.bayesian_particles import DirichletParticleOpponent, ParticleConfig
from negotiation.inference import Action, Event
from test_particles import synthetic_two_issue_case


class ParticleMathTests(unittest.TestCase):
    def test_accept_and_reject_match_hand_calculation(self):
        scenario, _, opponent, fast_costly, _ = synthetic_two_issue_case()
        config = ParticleConfig((1.0, 1.0), 2, 10.0, 5)
        accept = DirichletParticleOpponent(scenario, opponent.option_values, 0.5, config)
        reject = DirichletParticleOpponent(scenario, opponent.option_values, 0.5, config)
        particles = np.array([[0.8, 0.2], [0.2, 0.8]])
        accept.particles = particles.copy()
        reject.particles = particles.copy()
        self.assertAlmostEqual(accept.predict_accept(fast_costly, scenario), 0.5)
        accept.observe(Event("opponent", Action.ACCEPT, fast_costly, 1), scenario)
        reject.observe(Event("opponent", Action.REJECT, fast_costly, 1), scenario)
        expected = 1 / (1 + math.exp(-3.0))
        self.assertAlmostEqual(accept.weights[0], expected)
        self.assertAlmostEqual(reject.weights[0], 1 - expected)
        self.assertAlmostEqual(accept.effective_sample_size,
                               1 / (expected ** 2 + (1 - expected) ** 2))


if __name__ == "__main__":
    unittest.main()
