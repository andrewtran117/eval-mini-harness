#!/usr/bin/env python3
"""
Main CLI for running smoke tests against vLLM endpoints.

Loads config + dataset, orchestrates requests, collects metrics, renders report.
"""

import argparse


def main():
    """Run smoke tests."""
    parser = argparse.ArgumentParser(description="Run vLLM smoke tests")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Compare two run JSON files")

    args = parser.parse_args()

    # TODO: Implement
    print(f"Loading config from {args.config}")


if __name__ == "__main__":
    main()
