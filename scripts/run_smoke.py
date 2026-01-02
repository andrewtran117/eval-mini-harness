#!/usr/bin/env python3
"""
Main CLI for running smoke tests with an OpenAI-compatible endpoint (e.g., vLLM).

Loads config + dataset, orchestrates requests, collects metrics, renders report.
"""

import argparse
import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

# Add parent directory to path to import from evals
sys.path.insert(0, str(Path(__file__).parent.parent))

from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound

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
from evals.judge import llm_judge


async def _run_single_test(
    config: dict,
    test_case: dict,
    case_index: int,
    total: int,
    semaphore: asyncio.Semaphore,
    use_llm_judge_default: bool,
    judge_fn,
) -> dict:
    """
    Run a single test case with bounded concurrency.

    Uses a semaphore to limit concurrent calls; offloads the synchronous
    client to a thread via asyncio.to_thread.
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
            use_judge = test_case.get("use_llm_judge")
            if use_judge is None:
                use_judge = use_llm_judge_default

            ok, reason = score_response(
                response_text,
                test_case,
                use_llm_judge=use_judge,
                judge_fn=judge_fn,
            )
            status = "PASS" if ok else "FAIL"
            logs.append(f"  [{status}] {reason}")

            # Show response on failure for debugging
            if not ok:
                # Longer preview to help debug regex/format mismatches
                response_preview = response_text[:300].replace('\n', ' ')
                logs.append(f"  Response: \"{response_preview}{'...' if len(response_text) > 300 else ''}\")")

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
    Run smoke tests against an OpenAI-compatible endpoint with async bounded concurrency.

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

    # Test connection
    print("\nTesting model endpoint connection...")
    if not test_connection(config):
        print("Cannot connect to endpoint")
        print("  Make sure your vLLM/OpenAI-compatible server is reachable and model is loaded")
        sys.exit(1)
    print(f"Connected to endpoint (model: {get_model_name(config)})")

    # Run tests
    print(f"\nRunning smoke tests...")
    parallelism = max(1, get_parallelism(config))
    total = len(test_cases)

    # Prepare judge callable if configured
    scoring_cfg = config.get("scoring", {})
    judge_cfg = scoring_cfg.get("judge", {})
    default_use_llm_judge = scoring_cfg.get("default") == "llm_judge"

    judge_fn = None
    if judge_cfg:
        def judge_fn(test_case, prompt, response):
            return llm_judge(judge_cfg, test_case, prompt, response)

    semaphore = asyncio.Semaphore(parallelism)
    tasks = [
        asyncio.create_task(
            _run_single_test(
                config,
                test_case,
                i,
                total,
                semaphore,
                default_use_llm_judge,
                judge_fn,
            )
        )
        for i, test_case in enumerate(test_cases, 1)
    ]

    # Gather results; print logs as soon as we can while preserving order
    results = []
    buffer = {}
    next_to_emit = 1
    for finished in asyncio.as_completed(tasks):
        r = await finished
        buffer[r["_index"]] = r

        # Emit in order while contiguous indexes are available
        while next_to_emit in buffer:
            current = buffer.pop(next_to_emit)
            for line in current.pop("_logs", []):
                print(line)
            current.pop("_index", None)
            results.append(current)
            next_to_emit += 1

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
    """Save results to JSON file and render reports."""
    output_dir = Path(get_output_dir(config))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    run_name = config["run"].get("run_name", "run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{run_name}_{timestamp}"
    json_path = output_dir / f"{base_filename}.json"

    # Save
    with open(json_path, "w") as f:
        json.dump(run_data, f, indent=2)

    print(f"\nResults saved to: {json_path}")

    # Render Markdown and HTML reports if templates exist
    render_reports(run_data, config, output_dir, base_filename)


def _get_template_env() -> Environment:
    """Return a Jinja2 environment pointing at templates/."""
    templates_dir = Path(__file__).parent.parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_reports(run_data: Dict[str, Any], config: Dict[str, Any], output_dir: Path, base_filename: str) -> None:
    """Render Markdown and HTML reports if templates are available."""
    env = _get_template_env()
    context = _build_report_context(run_data, config)

    for name, ext in [("report.md.j2", "md"), ("report.html.j2", "html")]:
        try:
            template = env.get_template(name)
        except TemplateNotFound:
            continue

        output_path = output_dir / f"{base_filename}.{ext}"
        rendered = template.render(**context)
        with open(output_path, "w") as f:
            f.write(rendered)
        print(f"Report saved to: {output_path}")


def _build_report_context(run_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for templating."""
    results = run_data["results"]
    stats = calculate_summary_stats(results)
    failed = get_failed_tests(results)
    return {
        "run": run_data,
        "config": config,
        "stats": stats,
        "failed": failed,
        "timestamp": run_data.get("timestamp"),
        "model": run_data.get("config", {}).get("model"),
        "dataset": get_dataset_path(config),
        "run_name": config["run"].get("run_name"),
    }


def main():
    """Run smoke tests."""
    parser = argparse.ArgumentParser(description="Run smoke tests against an OpenAI-compatible endpoint (e.g., vLLM)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Compare two run JSON files")

    args = parser.parse_args()

    # Handle compare mode
    if args.compare:
        print("Compare mode not yet implemented")
        sys.exit(1)

    # Load config
    print("=" * 60)
    print("OPENAI-COMPATIBLE SMOKE TEST RUNNER")
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
