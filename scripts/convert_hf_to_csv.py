"""
HuggingFace dataset adapter.

Converts HF datasets to our smoke test format.
"""

from typing import List, Dict, Any


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


def save_as_csv(cases: List[Dict[str, Any]], output_path: str):
    """
    Save test cases as CSV file.

    Args:
        cases: List of test cases
        output_path: Path to save CSV file
    """
    import csv

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['id', 'prompt', 'expect_regex', 'keywords', 'expect_json',
                      'expect_number', 'tolerance', 'tags']
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
                'expect_number': case['expect_number'] if case['expect_number'] is not None else '',
                'tolerance': case['tolerance'] if case['tolerance'] is not None else '',
                'tags': ';'.join(case['tags']) if case['tags'] else ''
            }
            writer.writerow(row)


# Example usage
if __name__ == "__main__":
    print("Loading BoolQ dataset from HuggingFace...")
    cases = load_boolq_dataset(limit=20)

    print(f"Loaded {len(cases)} test cases")
    print("\nFirst 3 examples:")
    for case in cases[:3]:
        print(f"\n  ID: {case['id']}")
        print(f"  Prompt: {case['prompt']}")
        print(f"  Expected: {case['keywords']}")

    # Save to CSV
    output_path = "data/boolq_smoke.csv"
    save_as_csv(cases, output_path)
    print(f"\nSaved to: {output_path}")
    print("\nNow run: python scripts/run_smoke.py --config config.yaml")
    print("(After updating config.yaml dataset path to 'data/boolq_smoke.csv')")
