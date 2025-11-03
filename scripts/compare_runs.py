#!/usr/bin/env python3
"""
Compare two smoke test runs.

Usage:
  python scripts/compare_runs.py reports/run1.json reports/run2.json
"""

import json
import sys
from pathlib import Path


def load_run(filepath: str) -> dict:
    """Load run JSON file."""
    with open(filepath) as f:
        return json.load(f)


def compare_runs(run1_path: str, run2_path: str):
    """Compare two test runs."""
    run1 = load_run(run1_path)
    run2 = load_run(run2_path)

    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Basic info
    print(f"\nRun A: {run1_path}")
    print(f"  Model: {run1['config']['model']}")
    print(f"  Timestamp: {run1['timestamp']}")

    print(f"\nRun B: {run2_path}")
    print(f"  Model: {run2['config']['model']}")
    print(f"  Timestamp: {run2['timestamp']}")

    # Calculate stats for each run
    results1 = run1['results']
    results2 = run2['results']

    total1 = len(results1)
    passed1 = sum(1 for r in results1 if r['ok'])
    pass_rate1 = (passed1 / total1 * 100) if total1 > 0 else 0

    total2 = len(results2)
    passed2 = sum(1 for r in results2 if r['ok'])
    pass_rate2 = (passed2 / total2 * 100) if total2 > 0 else 0

    # Overall comparison
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(f"\nPass Rate:")
    print(f"  Run A: {pass_rate1:.1f}% ({passed1}/{total1})")
    print(f"  Run B: {pass_rate2:.1f}% ({passed2}/{total2})")
    diff = pass_rate2 - pass_rate1
    print(f"  Difference: {diff:+.1f}%")

    # Latency comparison
    latencies1 = [r['latency_s'] for r in results1 if r.get('latency_s')]
    latencies2 = [r['latency_s'] for r in results2 if r.get('latency_s')]

    if latencies1 and latencies2:
        avg_lat1 = sum(latencies1) / len(latencies1)
        avg_lat2 = sum(latencies2) / len(latencies2)

        print(f"\nAverage Latency:")
        print(f"  Run A: {avg_lat1:.2f}s")
        print(f"  Run B: {avg_lat2:.2f}s")
        speedup = ((avg_lat1 - avg_lat2) / avg_lat1 * 100) if avg_lat1 > 0 else 0
        print(f"  Speedup: {speedup:+.1f}%")

    # Per-test differences
    print("\n" + "=" * 60)
    print("PER-TEST DIFFERENCES")
    print("=" * 60)

    changed = []
    for r1, r2 in zip(results1, results2):
        if r1['ok'] != r2['ok']:
            status1 = "PASS" if r1['ok'] else "FAIL"
            status2 = "PASS" if r2['ok'] else "FAIL"
            changed.append((r1['id'], r1['prompt'][:50], status1, status2))

    if changed:
        print(f"\nTests with different results: {len(changed)}")
        for test_id, prompt, status1, status2 in changed:
            print(f"\n  [{test_id}] {prompt}...")
            print(f"    Run A: {status1}")
            print(f"    Run B: {status2}")
    else:
        print("\nNo differences - both runs had identical pass/fail results")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_runs.py <run1.json> <run2.json>")
        sys.exit(1)

    run1_path = sys.argv[1]
    run2_path = sys.argv[2]

    if not Path(run1_path).exists():
        print(f"Error: File not found: {run1_path}")
        sys.exit(1)

    if not Path(run2_path).exists():
        print(f"Error: File not found: {run2_path}")
        sys.exit(1)

    compare_runs(run1_path, run2_path)


if __name__ == "__main__":
    main()
