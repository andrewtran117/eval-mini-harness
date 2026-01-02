#!/usr/bin/env python3
"""
Performance probing tool for measuring TTFT, latency, and throughput.

Measures performance across different prompt lengths and concurrency levels.
"""

import argparse


def main():
    """Run performance probes."""
    parser = argparse.ArgumentParser(description="Run vLLM performance probes")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")

    args = parser.parse_args()

    # TODO: Implement
    print(f"Running performance probes with config {args.config}")


if __name__ == "__main__":
    main()
