#!/usr/bin/env python3
"""
Split a merged answers file (JSON or JSONL) into N shards for parallel judging.

Examples:
  python scripts/split_dataset.py --input reports/merged_answers.json --output_dir shards --shard_count 4
  python scripts/split_dataset.py --input reports/merged_answers.jsonl --output_dir shards --shard_count 4 --as_jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_items(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        return data
    except json.JSONDecodeError:
        items: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items


def write_shard(items: List[Dict[str, Any]], out_path: Path, as_jsonl: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if as_jsonl:
        with out_path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
    else:
        out_path.write_text(json.dumps(items, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Split merged answers into shards for parallel processing")
    ap.add_argument("--input", required=True, help="Merged answers file (JSON or JSONL)")
    ap.add_argument("--output_dir", required=True, help="Directory to write shard files")
    ap.add_argument("--shard_count", type=int, required=True, help="Number of shards to create")
    ap.add_argument("--prefix", default="shard", help="Output filename prefix (default: shard)")
    ap.add_argument("--as_jsonl", action="store_true", help="Write shards as JSONL instead of JSON array")
    args = ap.parse_args()

    if args.shard_count < 1:
        raise ValueError("shard_count must be >= 1")

    src = Path(args.input)
    out_dir = Path(args.output_dir)

    items = load_items(src)
    total = len(items)
    if total == 0:
        print("No items found; nothing to shard.")
        return

    print(f"Loaded {total} items from {src}")

    shards: List[List[Dict[str, Any]]] = [[] for _ in range(args.shard_count)]
    for idx, item in enumerate(items):
        shards[idx % args.shard_count].append(item)

    for shard_idx, shard_items in enumerate(shards):
        filename = f"{args.prefix}_{shard_idx:02d}.jsonl" if args.as_jsonl else f"{args.prefix}_{shard_idx:02d}.json"
        out_path = out_dir / filename
        write_shard(shard_items, out_path, args.as_jsonl)
        print(f"Shard {shard_idx + 1}/{args.shard_count}: {len(shard_items)} items -> {out_path}")


if __name__ == "__main__":
    main()
