#!/usr/bin/env python3
"""
Merge multiple run JSONs into a single answers file keyed by test case id.

Example:
  python scripts/merge_runs.py \
    --label A_llama31 reports/xstest_llama31_8b_q4km_*.json \
    --label B_qwen reports/xstest_qwen25_7b_*.json \
    --label C_mistral reports/xstest_mistral_nemo_12b_*.json \
    --output reports/merged_answers.json
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_run(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def pick_file(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return matches[-1]  # latest if timestamped


def merge_runs(labeled_paths: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for label, path in labeled_paths:
        run = load_run(path)
        for r in run.get("results", []):
            case_id = str(r.get("id"))
            if case_id not in merged:
                merged[case_id] = {
                    "id": case_id,
                    "prompt": r.get("prompt"),
                    "tags": r.get("tags", []),
                    "answers": {},
                }
            merged[case_id]["answers"][label] = r.get("response")
    # Return sorted by id
    return [merged[k] for k in sorted(merged, key=lambda x: int(x) if x.isdigit() else x)]


def main():
    parser = argparse.ArgumentParser(description="Merge multiple run JSONs into one answers file")
    parser.add_argument(
        "--label",
        action="append",
        nargs=2,
        metavar=("LABEL", "GLOB"),
        required=True,
        help="Label and glob pattern for a run JSON (e.g., A_llama31 reports/xstest_llama31_*.json)",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    labeled_paths = []
    for label, pattern in args.label:
        path = pick_file(pattern)
        labeled_paths.append((label, path))

    merged = merge_runs(labeled_paths)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(labeled_paths)} runs into {out_path} ({len(merged)} cases)")


if __name__ == "__main__":
    main()
