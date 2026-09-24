"""Run a synthetic Bayesian negotiation with the structural oracle disabled."""

import argparse
from dataclasses import replace
import json

from .bayesian_config import load_particle_config
from .bayesian_particles import bayesian_cores
from .domain import load_scenario
from .protocol import NegotiationController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="Path to JSON scenario configuration")
    parser.add_argument("particles", help="Path to per-observer particle configuration")
    args = parser.parse_args()
    source_scenario = load_scenario(args.scenario)
    configurations = load_particle_config(args.particles, source_scenario)
    # Exact structural deadlock requires hidden true preferences and is unavailable online.
    scenario = replace(source_scenario, structural_deadlock_check=False)
    result = NegotiationController(scenario, bayesian_cores(scenario, configurations)).run()
    output = result.as_dict(scenario)
    output["mode"] = "bayesian_issue_weights"
    output["structural_deadlock_oracle"] = False
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
