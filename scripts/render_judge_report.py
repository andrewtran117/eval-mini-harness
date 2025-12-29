#!/usr/bin/env python3
"""
Render an HTML report from judge JSONL output.

Usage:
  python scripts/render_judge_report.py \
    --input reports/judge_results.jsonl \
    --output reports/judge_results.html
"""

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "total": len(records),
        "by_judge": defaultdict(lambda: defaultdict(Counter)),  # judge -> pair -> winner counts
        "parse_errors": 0,
        "by_rubric": defaultdict(int),
    }
    for r in records:
        judge = r.get("judge_model", "unknown")
        pair = r.get("pair", "unknown")
        winner = r.get("winner", "unknown")
        rubric = r.get("rubric", "unknown")

        summary["by_judge"][judge][pair][winner] += 1
        summary["by_rubric"][rubric] += 1
        if r.get("parse_error"):
            summary["parse_errors"] += 1
    return summary


def build_env() -> Environment:
    templates_dir = Path(__file__).parent.parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Render HTML report from judge JSONL")
    parser.add_argument("--input", required=True, help="Judge JSONL file")
    parser.add_argument("--output", required=True, help="HTML output file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_jsonl(input_path)
    summary = summarize(records)

    env = build_env()
    template = env.get_template("judge_report.html.j2")
    rendered = template.render(records=records, summary=summary, input_path=str(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    print(f"Rendered {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
