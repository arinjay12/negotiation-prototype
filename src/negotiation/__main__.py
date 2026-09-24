"""Run the deterministic fixture and emit a machine-readable complete trace."""

import argparse
import json

from .domain import load_scenario
from .inference import perfect_information_cores
from .protocol import NegotiationController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="Path to JSON scenario configuration")
    args = parser.parse_args()
    scenario = load_scenario(args.scenario)
    result = NegotiationController(scenario, perfect_information_cores(scenario)).run()
    print(json.dumps(result.as_dict(scenario), indent=2))


if __name__ == "__main__":
    main()
