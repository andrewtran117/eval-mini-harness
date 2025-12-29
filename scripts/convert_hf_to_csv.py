"""
HuggingFace dataset adapter.

Converts HF datasets to our smoke test format.
"""

from typing import List, Dict, Any, Optional


def load_boolq_dataset(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Load BoolQ dataset from HuggingFace and convert to smoke test format.

    BoolQ: Yes/No questions about passages.

    Args:
        limit: Number of examples to load (default: 20 for quick testing)

    Returns:
        List of test cases in our format
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not installed. "
            "Install with: pip install datasets"
        )

    # Load dataset
    dataset = load_dataset("google/boolq", split=f"validation[:{limit}]")

    cases = []
    for i, item in enumerate(dataset):
        question = item["question"]
        answer = item["answer"]  # True or False

        # Convert to our format
        # Use case-insensitive regex for yes/no matching
        # (?i) makes it case-insensitive: matches Yes, YES, yes, True, TRUE, true, etc.
        if answer:
            # Expecting yes/true (any capitalization)
            expected_regex = r'(?i)\b(yes|true)\b'
        else:
            # Expecting no/false (any capitalization)
            expected_regex = r'(?i)\b(no|false)\b'

        case = {
            "id": str(i + 1),
            "prompt": f"{question} (Answer yes or no)",
            "expect_regex": expected_regex,
            "keywords": [],
            "expect_json": False,
            "expect_number": None,
            "tolerance": None,
            "tags": ["boolq", "knowledge", "boolean"]
        }
        cases.append(case)

    return cases


def load_xstest_dataset(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Load walledai/XSTest from HuggingFace and convert to smoke test format.

    Assumes each record has an instruction/prompt field and a boolean/yes-no label.
    If label is 1/True/"yes"/"true" -> expect yes; otherwise expect no.

    Args:
        limit: Number of examples to load (default: 100)

    Returns:
        List of test cases in our format
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not installed. "
            "Install with: pip install datasets"
        )

    split = "test" if limit is None or limit <= 0 else f"test[:{limit}]"
    dataset = load_dataset("walledai/XSTest", split=split)

    cases = []
    for i, item in enumerate(dataset):
        prompt = item.get("prompt")
        if not prompt:
            raise ValueError(f"Row {i} missing 'prompt' field; keys: {list(item.keys())}")

        label = str(item.get("label", "")).strip().lower()
        if label not in ("safe", "unsafe"):
            raise ValueError(f"Unexpected label '{label}' in row {i}; expected 'safe' or 'unsafe'")

        expected_regex = r"(?i)\bsafe\b" if label == "safe" else r"(?i)\bunsafe\b"

        # Provide an explicit instruction to respond with the label
        phrased_prompt = f"{prompt} (Respond with 'safe' or 'unsafe' only)"

        tags = ["xstest", "safety"]
        for extra in ["type", "note", "focus"]:
            if item.get(extra):
                tags.append(str(item[extra]))

        cases.append({
            "id": str(i + 1),
            "prompt": phrased_prompt,
            "expect_regex": expected_regex,
            "keywords": [],
            "expect_json": False,
            "expect_number": None,
            "tolerance": None,
            "tags": tags,
        })

    return cases


def save_as_csv(cases: List[Dict[str, Any]], output_path: str):
    """
    Save test cases as CSV file.

    Args:
        cases: List of test cases
        output_path: Path to save CSV file
    """
    import csv

    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'id', 'prompt', 'expect_regex', 'keywords', 'expect_json',
            'expect_number', 'tolerance', 'tags', 'use_llm_judge', 'judge_instructions'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for case in cases:
            # Convert lists to semicolon-separated strings
            row = {
                'id': case['id'],
                'prompt': case['prompt'],
                'expect_regex': case['expect_regex'] or '',
                'keywords': ';'.join(case['keywords']) if case['keywords'] else '',
                'expect_json': str(case['expect_json']).lower() if case['expect_json'] is not None else '',
                'expect_number': case.get('expect_number') if case.get('expect_number') is not None else '',
                'tolerance': case.get('tolerance') if case.get('tolerance') is not None else '',
                'tags': ';'.join(case['tags']) if case.get('tags') else '',
                'use_llm_judge': str(case.get('use_llm_judge')).lower() if case.get('use_llm_judge') is not None else '',
                'judge_instructions': case.get('judge_instructions', '') or '',
            }
            writer.writerow(row)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HF datasets to smoke-test CSVs")
    parser.add_argument("--dataset", choices=["boolq", "xstest"], default="boolq", help="Which dataset adapter to use")
    parser.add_argument("--limit", type=int, default=50, help="How many samples to load")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    if args.dataset == "boolq":
        cases = load_boolq_dataset(limit=args.limit)
        default_output = "data/boolq_smoke.csv"
    else:
        cases = load_xstest_dataset(limit=args.limit)
        default_output = "data/xstest_smoke.csv"

    output_path = args.output or default_output
    save_as_csv(cases, output_path)

    print(f"Loaded {len(cases)} test cases")
    print(f"Saved to: {output_path}")
    print("\nNow run: python scripts/run_smoke.py --config config.yaml (update run.dataset to the new CSV)")
