"""
Entry for doing quantitative analysis on strategies.

Excepts a single config file detailing the experiments
  Runs 2 types of experiments:
    - Optimizing a strategy
    - Evaluating the robustness of a strategy
"""

# System Imports
import argparse
import yaml
import gc
import os

# Local Imports
from utils.config_model import Config
import optimize_strategy
import robustness_test


def main(config_file):
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    config = Config(**config_data)
    root_folder = os.path.dirname(config_file)

    for experiment in config.experiments:
        if experiment.type == "optimize":
            optimize_strategy.main(experiment, root_folder)
        elif experiment.type == "robustness":
            robustness_test.main(experiment, root_folder)
        # These functions can use excessive memory
        # Make sure to clean it up for the next experiment
        gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Main file for running quantitative analysis."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
