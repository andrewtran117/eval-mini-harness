"""
Scoring functions for validating LLM outputs.

Supports: regex, keyword matching, JSON validation, numeric tolerance.
"""

import re
import json
from typing import Dict, Any, Tuple


def score_regex(response: str, pattern: str) -> Tuple[bool, str]:
    """Check if response matches regex pattern."""
    # TODO: Implement
    pass


def score_keywords(response: str, keywords: list[str], any_match: bool = False) -> Tuple[bool, str]:
    """Check if response contains keywords (case-insensitive)."""
    # TODO: Implement
    pass


def score_json(response: str, schema_path: str = None) -> Tuple[bool, str]:
    """Validate response is valid JSON, optionally against schema."""
    # TODO: Implement
    pass


def score_number(response: str, target: float, tolerance: float = 0.0) -> Tuple[bool, str]:
    """Extract first number from response and check if within tolerance of target."""
    # TODO: Implement
    pass
