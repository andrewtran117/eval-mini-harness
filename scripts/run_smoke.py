#!/usr/bin/env python3
"""
Main CLI for running smoke tests against vLLM endpoints.

Loads config + dataset, orchestrates requests, collects metrics, renders report.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from evals
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.config import (
    load_config,
    ConfigError,
    get_endpoint_url,
    get_model_name,
    get_dataset_path,
    get_output_dir,
)


def main():
    """Run smoke tests."""
    parser = argparse.ArgumentParser(description="Run vLLM smoke tests")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Compare two run JSON files")

    args = parser.parse_args()

    # Load and validate config
    try:
        config = load_config(args.config)
        print(f"✓ Config loaded from {args.config}")
        print(f"  Endpoint: {get_endpoint_url(config)}")
        print(f"  Model: {get_model_name(config)}")
        print(f"  Dataset: {get_dataset_path(config)}")
        print(f"  Output: {get_output_dir(config)}")
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # TODO: Load dataset, run tests, generate report


if __name__ == "__main__":
    main()
