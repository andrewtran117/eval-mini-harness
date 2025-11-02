"""
Metrics calculation and aggregation.

Computes pass rates, latency percentiles, tokens/sec, etc.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict


def calculate_pass_rate(results: List[Dict[str, Any]]) -> float:
    """
    Calculate overall pass rate as a percentage.

    Args:
        results: List of test results with 'ok' field

    Returns:
        Pass rate as percentage (0-100)
    """
    if not results:
        return 0.0

    passed = sum(1 for r in results if r.get("ok", False))
    return (passed / len(results)) * 100


def calculate_pass_rate_by_tag(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate pass rates grouped by tags.

    Args:
        results: List of test results with 'ok' and 'tags' fields

    Returns:
        Dictionary mapping tag to stats dict with:
        - pass_rate: percentage
        - passed: number passed
        - total: total tests with this tag
    """
    tag_stats = defaultdict(lambda: {"passed": 0, "total": 0})

    for result in results:
        tags = result.get("tags", [])
        ok = result.get("ok", False)

        for tag in tags:
            tag_stats[tag]["total"] += 1
            if ok:
                tag_stats[tag]["passed"] += 1

    # Calculate percentages
    for tag, stats in tag_stats.items():
        if stats["total"] > 0:
            stats["pass_rate"] = (stats["passed"] / stats["total"]) * 100
        else:
            stats["pass_rate"] = 0.0

    return dict(tag_stats)


def calculate_percentile(values: List[float], percentile: int) -> Optional[float]:
    """
    Calculate percentile (p50, p95, etc.).

    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)

    Returns:
        Percentile value, or None if values is empty
    """
    if not values:
        return None

    if percentile < 0 or percentile > 100:
        raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    # Use linear interpolation method
    rank = (percentile / 100) * (n - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, n - 1)
    fraction = rank - lower_idx

    lower_value = sorted_values[lower_idx]
    upper_value = sorted_values[upper_idx]

    return lower_value + fraction * (upper_value - lower_value)


def calculate_tokens_per_sec(results: List[Dict[str, Any]]) -> Optional[float]:
    """
    Calculate average tokens per second across all results.

    Args:
        results: List of test results with 'tokens_out' and 'latency_s' fields

    Returns:
        Average tokens/sec, or None if no valid data
    """
    valid_results = [
        r for r in results
        if r.get("tokens_out") and r.get("latency_s") and r.get("latency_s") > 0
    ]

    if not valid_results:
        return None

    total_tokens = sum(r["tokens_out"] for r in valid_results)
    total_time = sum(r["latency_s"] for r in valid_results)

    return total_tokens / total_time if total_time > 0 else None


def calculate_latency_stats(results: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Calculate latency statistics (min, max, mean, p50, p95, p99).

    Args:
        results: List of test results with 'latency_s' field

    Returns:
        Dictionary with latency statistics
    """
    latencies = [r["latency_s"] for r in results if "latency_s" in r]

    if not latencies:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }

    return {
        "min": min(latencies),
        "max": max(latencies),
        "mean": sum(latencies) / len(latencies),
        "p50": calculate_percentile(latencies, 50),
        "p95": calculate_percentile(latencies, 95),
        "p99": calculate_percentile(latencies, 99),
    }


def calculate_ttft_stats(results: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Calculate Time To First Token (TTFT) statistics.

    Args:
        results: List of test results with 'ttft_ms' field

    Returns:
        Dictionary with TTFT statistics in milliseconds
    """
    ttfts = [r["ttft_ms"] for r in results if "ttft_ms" in r and r["ttft_ms"] is not None]

    if not ttfts:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
        }

    return {
        "min": min(ttfts),
        "max": max(ttfts),
        "mean": sum(ttfts) / len(ttfts),
        "p50": calculate_percentile(ttfts, 50),
        "p95": calculate_percentile(ttfts, 95),
    }


def calculate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive summary statistics.

    Args:
        results: List of test results

    Returns:
        Dictionary with all summary statistics
    """
    return {
        "total_tests": len(results),
        "passed": sum(1 for r in results if r.get("ok", False)),
        "failed": sum(1 for r in results if not r.get("ok", False)),
        "pass_rate": calculate_pass_rate(results),
        "latency": calculate_latency_stats(results),
        "ttft": calculate_ttft_stats(results),
        "tokens_per_sec": calculate_tokens_per_sec(results),
        "by_tag": calculate_pass_rate_by_tag(results),
    }


def get_failed_tests(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get all failed test results.

    Args:
        results: List of test results

    Returns:
        List of failed test results
    """
    return [r for r in results if not r.get("ok", False)]


def get_passed_tests(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get all passed test results.

    Args:
        results: List of test results

    Returns:
        List of passed test results
    """
    return [r for r in results if r.get("ok", False)]
