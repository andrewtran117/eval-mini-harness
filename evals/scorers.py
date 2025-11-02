"""
Scoring functions for validating LLM outputs.

Supports: regex, keyword matching, JSON validation, numeric tolerance.
"""

import re
import json
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path


def score_regex(response: str, pattern: str) -> Tuple[bool, str]:
    """
    Check if response matches regex pattern.

    Args:
        response: LLM output text
        pattern: Regular expression pattern to match

    Returns:
        Tuple of (pass/fail, reason)
    """
    if not pattern:
        return False, "No pattern provided"

    try:
        match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
        if match:
            matched_text = match.group(0)
            preview = matched_text[:50] + "..." if len(matched_text) > 50 else matched_text
            return True, f"Regex matched: '{preview}'"
        else:
            return False, f"Regex pattern not found: {pattern[:50]}"
    except re.error as e:
        return False, f"Invalid regex pattern: {e}"


def score_keywords(response: str, keywords: List[str], any_match: bool = False) -> Tuple[bool, str]:
    """
    Check if response contains keywords (case-insensitive).

    Args:
        response: LLM output text
        keywords: List of keywords to search for
        any_match: If True, any keyword match passes; if False, all keywords must be present

    Returns:
        Tuple of (pass/fail, reason)
    """
    if not keywords:
        return False, "No keywords provided"

    response_lower = response.lower()
    found_keywords = []
    missing_keywords = []

    for keyword in keywords:
        if keyword.lower() in response_lower:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    if any_match:
        # At least one keyword must match
        if found_keywords:
            return True, f"Found keyword(s): {', '.join(found_keywords)}"
        else:
            return False, f"None of the keywords found: {', '.join(keywords)}"
    else:
        # All keywords must match
        if not missing_keywords:
            return True, f"All keywords found: {', '.join(found_keywords)}"
        else:
            return False, f"Missing keyword(s): {', '.join(missing_keywords)}"


def score_json(response: str, schema_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate response is valid JSON, optionally against schema.

    Args:
        response: LLM output text
        schema_path: Optional path to JSON schema file for validation

    Returns:
        Tuple of (pass/fail, reason)
    """
    # Try to parse as JSON
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    # If no schema provided, just check it's valid JSON
    if not schema_path:
        json_type = type(parsed).__name__
        return True, f"Valid JSON ({json_type})"

    # Validate against schema if provided
    try:
        import jsonschema
        schema_file = Path(schema_path)
        if not schema_file.exists():
            return False, f"Schema file not found: {schema_path}"

        with open(schema_file) as f:
            schema = json.load(f)

        jsonschema.validate(instance=parsed, schema=schema)
        return True, "Valid JSON matching schema"
    except ImportError:
        return True, "Valid JSON (jsonschema not installed, skipping schema validation)"
    except jsonschema.ValidationError as e:
        return False, f"JSON schema validation failed: {e.message}"
    except Exception as e:
        return False, f"Schema validation error: {e}"


def score_number(response: str, target: float, tolerance: float = 0.0) -> Tuple[bool, str]:
    """
    Extract first number from response and check if within tolerance of target.

    Args:
        response: LLM output text
        target: Expected numeric value
        tolerance: Allowed difference from target (default: 0.0 for exact match)

    Returns:
        Tuple of (pass/fail, reason)
    """
    # Extract all numbers from response (integers and floats)
    # Pattern matches: integers, floats, negative numbers, scientific notation
    number_pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
    matches = re.findall(number_pattern, response)

    if not matches:
        return False, f"No number found in response (expected {target})"

    # Try each number found (prioritize later numbers as they're more likely to be answers)
    for match in reversed(matches):
        try:
            found_number = float(match)
            diff = abs(found_number - target)

            if diff <= tolerance:
                if tolerance == 0:
                    return True, f"Number matches exactly: {found_number}"
                else:
                    return True, f"Number within tolerance: {found_number} (target: {target}, tolerance: ±{tolerance})"
            else:
                # Continue checking other numbers
                continue
        except ValueError:
            continue

    # If we get here, no matching number was found
    # Report the last number we tried
    try:
        last_num = float(matches[-1])
        diff = abs(last_num - target)
        return False, f"Number {last_num} outside tolerance (expected {target} ±{tolerance}, diff: {diff})"
    except:
        return False, f"Could not parse numbers in response (expected {target})"


def score_response(response: str, test_case: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Score a response against a test case using the appropriate scorer.

    Applies scorers in order of precedence:
    1. expect_regex
    2. expect_number
    3. expect_json
    4. keywords

    Args:
        response: LLM output text
        test_case: Test case dictionary from dataset

    Returns:
        Tuple of (pass/fail, reason)
    """
    # Try regex first
    if test_case.get("expect_regex"):
        return score_regex(response, test_case["expect_regex"])

    # Try number
    if test_case.get("expect_number") is not None:
        tolerance = test_case.get("tolerance", 0.0)
        return score_number(response, test_case["expect_number"], tolerance)

    # Try JSON
    if test_case.get("expect_json"):
        schema_path = test_case.get("json_schema")
        return score_json(response, schema_path)

    # Try keywords
    if test_case.get("keywords"):
        any_match = test_case.get("any_match", False)
        return score_keywords(response, test_case["keywords"], any_match)

    # No validation criteria specified
    return False, "No validation criteria specified in test case"
