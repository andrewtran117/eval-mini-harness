#!/usr/bin/env python3
"""
Main CLI for running smoke tests with Ollama.

Loads config + dataset, orchestrates requests, collects metrics, renders report.
"""

import argparse
import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import from evals
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.config import (
    load_config,
    ConfigError,
    get_model_name,
    get_dataset_path,
    get_output_dir,
    get_parallelism,
)
from evals.dataset import load_dataset, get_test_case_count, DatasetError
from evals.client import run_inference, test_connection, ClientError
from evals.scorers import score_response
from evals.metrics import calculate_summary_stats, get_failed_tests


async def _run_single_test(
    config: dict,
    test_case: dict,
    case_index: int,
    total: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Run a single test case with bounded concurrency.

    Uses a semaphore to limit concurrent calls; offloads the synchronous
    Ollama client to a thread via asyncio.to_thread.
    """
    case_id = test_case.get("id", case_index)
    prompt = test_case["prompt"]
    logs = []

    async with semaphore:
        logs.append(f"\n[{case_index}/{total}] Test case {case_id}")
        logs.append(f"  Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")

        try:
            # Run inference in a worker thread to avoid blocking the event loop
            response_text, metrics = await asyncio.to_thread(run_inference, config, prompt)
            logs.append(f"  Latency: {metrics['latency_s']:.2f}s")

            # Score response
            ok, reason = score_response(response_text, test_case)
            status = "PASS" if ok else "FAIL"
            logs.append(f"  [{status}] {reason}")

            # Show response on failure for debugging
            if not ok:
                response_preview = response_text[:100].replace('\n', ' ')
                logs.append(f"  Response: \"{response_preview}{'...' if len(response_text) > 100 else ''}\")")

            result = {
                "id": case_id,
                "prompt": prompt,
                "response": response_text,
                "ok": ok,
                "reason": reason,
                "tags": test_case.get("tags", []),
                **metrics,
            }
        except ClientError as e:
            print(f"  Client error: {e}")
            result = {
                "id": case_id,
                "prompt": prompt,
                "response": None,
                "ok": False,
                "reason": f"Client error: {e}",
                "tags": test_case.get("tags", []),
                "latency_s": None,
                "ttft_ms": None,
                "tokens_in": None,
                "tokens_out": None,
            }

    # Preserve original ordering by including index for later sort
    result["_index"] = case_index
    result["_logs"] = logs
    return result


async def run_smoke_tests_async(config: dict) -> dict:
    """
    Run smoke tests against Ollama with async bounded concurrency.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with test results and metadata
    """
    # Load dataset
    print("\nLoading dataset...")
    dataset_path = get_dataset_path(config)
    try:
        test_cases = load_dataset(dataset_path)
        print(f"Loaded {get_test_case_count(test_cases)} test cases from {dataset_path}")
    except DatasetError as e:
        print(f"Dataset error: {e}")
        sys.exit(1)

    # Test Ollama connection
    print("\nTesting Ollama connection...")
    if not test_connection(config):
        print("Cannot connect to Ollama")
        print("  Make sure Ollama is running: ollama serve")
        print(f"  And model is pulled: ollama pull {get_model_name(config)}")
        sys.exit(1)
    print(f"Connected to Ollama (model: {get_model_name(config)})")

    # Run tests
    print(f"\nRunning smoke tests...")
    parallelism = max(1, get_parallelism(config))
    total = len(test_cases)

    semaphore = asyncio.Semaphore(parallelism)
    tasks = [
        asyncio.create_task(_run_single_test(config, test_case, i, total, semaphore))
        for i, test_case in enumerate(test_cases, 1)
    ]

    # Gather results and restore original order
    raw_results = await asyncio.gather(*tasks)
    results_sorted = sorted(raw_results, key=lambda x: x["_index"])

    # Emit logs in order to avoid interleaving noise
    for r in results_sorted:
        for line in r.pop("_logs", []):
            print(line)

    results = []
    for r in results_sorted:
        r.pop("_index", None)
        results.append(r)

    return {
        "config": {
            "model": get_model_name(config),
            "temperature": config["sampling"].get("temperature"),
            "seed": config["run"].get("seed"),
        },
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }


def run_smoke_tests(config: dict) -> dict:
    """Synchronous wrapper to run async tests."""
    return asyncio.run(run_smoke_tests_async(config))


def display_summary(run_data: dict):
    """Display test summary to console."""
    results = run_data["results"]
    stats = calculate_summary_stats(results)

    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)

    # Overall stats
    print(f"\nOverall Results:")
    print(f"  Total tests:  {stats['total_tests']}")
    print(f"  Passed:       {stats['passed']} ({stats['pass_rate']:.1f}%)")
    print(f"  Failed:       {stats['failed']}")

    # Latency stats
    latency = stats["latency"]
    if latency["mean"] is not None:
        print(f"\nLatency:")
        print(f"  Mean:  {latency['mean']:.2f}s")
        print(f"  p50:   {latency['p50']:.2f}s")
        print(f"  p95:   {latency['p95']:.2f}s")

    # Tokens/sec
    if stats["tokens_per_sec"]:
        print(f"\nThroughput: {stats['tokens_per_sec']:.1f} tokens/sec")

    # By tag
    if stats["by_tag"]:
        print(f"\nResults by Tag:")
        for tag, tag_stats in sorted(stats["by_tag"].items()):
            print(f"  {tag:15s} {tag_stats['pass_rate']:5.1f}% ({tag_stats['passed']}/{tag_stats['total']})")

    # Failed tests
    failed = get_failed_tests(results)
    if failed:
        print(f"\nFailed Tests:")
        for result in failed:
            print(f"  [{result['id']}] {result['prompt'][:50]}...")
            print(f"      Reason: {result['reason']}")

    print("\n" + "=" * 60)


def save_results(run_data: dict, config: dict):
    """Save results to JSON file."""
    output_dir = Path(get_output_dir(config))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    run_name = config["run"].get("run_name", "run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{run_name}_{timestamp}.json"
    filepath = output_dir / filename

    # Save
    with open(filepath, "w") as f:
        json.dump(run_data, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def main():
    """Run smoke tests."""
    parser = argparse.ArgumentParser(description="Run Ollama smoke tests")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Compare two run JSON files")

    args = parser.parse_args()

    # Handle compare mode
    if args.compare:
        print("Compare mode not yet implemented")
        sys.exit(1)

    # Load config
    print("=" * 60)
    print("OLLAMA SMOKE TEST RUNNER")
    print("=" * 60)

    try:
        config = load_config(args.config)
        print(f"\nConfig loaded from {args.config}")
        print(f"  Model: {get_model_name(config)}")
        print(f"  Dataset: {get_dataset_path(config)}")
    except ConfigError as e:
        print(f"Config error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("  Make sure you're running from the repo root directory")
        sys.exit(1)

    # Run tests
    start_time = time.time()
    run_data = run_smoke_tests(config)
    elapsed = time.time() - start_time

    # Display results
    display_summary(run_data)
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    save_results(run_data, config)

    # Exit with appropriate code
    stats = calculate_summary_stats(run_data["results"])
    sys.exit(0 if stats["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
