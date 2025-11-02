"""
Dataset CSV parser.

Loads test cases from CSV files with prompts and expected outputs.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional


class DatasetError(Exception):
    """Raised when dataset loading or parsing fails."""
    pass


def load_dataset(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load test cases from CSV file.

    Args:
        csv_path: Path to CSV dataset file

    Returns:
        List of test case dictionaries

    Raises:
        DatasetError: If file not found or parsing fails

    CSV Schema:
        - id: Optional test case identifier
        - prompt: The prompt to send to LLM (required)
        - expect_regex: Regex pattern to match (optional)
        - keywords: Semicolon-separated keywords (optional)
        - expect_json: Whether to expect valid JSON (optional, bool)
        - expect_number: Expected numeric answer (optional, float)
        - tolerance: Tolerance for numeric comparison (optional, float)
        - tags: Semicolon-separated tags (optional)
    """
    path = Path(csv_path)

    if not path.exists():
        raise DatasetError(f"Dataset file not found: {csv_path}")

    cases = []

    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise DatasetError(f"CSV file has no header row: {csv_path}")

            if "prompt" not in reader.fieldnames:
                raise DatasetError(f"CSV missing required 'prompt' column: {csv_path}")

            for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                try:
                    case = _parse_row(row, row_num)
                    cases.append(case)
                except Exception as e:
                    raise DatasetError(f"Error parsing row {row_num}: {e}")

    except csv.Error as e:
        raise DatasetError(f"CSV parsing error: {e}")
    except UnicodeDecodeError as e:
        raise DatasetError(f"File encoding error: {e}")

    if not cases:
        raise DatasetError(f"Dataset is empty: {csv_path}")

    return cases


def _parse_row(row: Dict[str, str], row_num: int) -> Dict[str, Any]:
    """
    Parse a single CSV row into a test case dictionary.

    Args:
        row: Raw CSV row as dict
        row_num: Row number (for error messages)

    Returns:
        Parsed test case dictionary
    """
    # Required field
    prompt = row.get("prompt", "").strip()
    if not prompt:
        raise ValueError(f"Row {row_num}: 'prompt' is required and cannot be empty")

    # Build test case
    case = {
        "id": row.get("id", "").strip() or str(row_num - 1),  # Auto-generate ID if missing
        "prompt": prompt,
        "expect_regex": _parse_optional_string(row.get("expect_regex")),
        "keywords": _parse_semicolon_list(row.get("keywords")),
        "expect_json": _parse_bool(row.get("expect_json")),
        "expect_number": _parse_float(row.get("expect_number")),
        "tolerance": _parse_float(row.get("tolerance")),
        "tags": _parse_semicolon_list(row.get("tags")),
    }

    return case


def _parse_optional_string(value: Optional[str]) -> Optional[str]:
    """Parse optional string field."""
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _parse_semicolon_list(value: Optional[str]) -> List[str]:
    """Parse semicolon-separated list."""
    if not value or not value.strip():
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    """Parse boolean field."""
    if not value or not value.strip():
        return None
    value_lower = value.strip().lower()
    if value_lower in ("true", "1", "yes", "y"):
        return True
    if value_lower in ("false", "0", "no", "n"):
        return False
    return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    """Parse float field."""
    if not value or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def get_test_case_count(cases: List[Dict[str, Any]]) -> int:
    """Get total number of test cases."""
    return len(cases)


def get_tags(cases: List[Dict[str, Any]]) -> List[str]:
    """Get all unique tags from test cases."""
    tags = set()
    for case in cases:
        tags.update(case.get("tags", []))
    return sorted(tags)


def filter_by_tag(cases: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    """Filter test cases by tag."""
    return [case for case in cases if tag in case.get("tags", [])]
