"""Load explicit per-observer particle settings without hidden defaults."""

import json
from pathlib import Path

from .bayesian_particles import ParticleConfig
from .domain import Scenario


def load_particle_config(path: str | Path, scenario: Scenario) -> dict[str, dict[str, ParticleConfig]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    observers = data["observers"]
    if set(observers) != set(scenario.turn_order):
        raise ValueError("Bayesian configuration must cover each observer")
    result: dict[str, dict[str, ParticleConfig]] = {}
    for observer, opponents in observers.items():
        expected = set(scenario.turn_order) - {observer}
        if set(opponents) != expected:
            raise ValueError(f"Bayesian configuration must cover each opponent for {observer}")
        result[observer] = {}
        for opponent, item in opponents.items():
            config = ParticleConfig(
                alpha=tuple(item["alpha"]), count=item["count"], beta=item["beta"],
                seed=item["seed"], ess_threshold=item.get("ess_threshold"),
                resample=item.get("resample", False),
            )
            config.validate(len(scenario.issues))
            result[observer][opponent] = config
    return result
