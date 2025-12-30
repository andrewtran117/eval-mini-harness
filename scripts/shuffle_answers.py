#!/usr/bin/env python3
"""
Shuffle answer order per item to remove fixed A/B/C bias.
Usage:
  python scripts/shuffle_answers.py --input reports/xstest_merged_answers.json \
                                    --output reports/xstest_merged_shuffled.json \
                                    --seed 42
"""
import json, random, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    items = json.loads(Path(args.input).read_text())
    shuffled = []

    for it in items:
        ans_list = list(it["answers"].items())  # [(orig_label, text), ...]
        random.shuffle(ans_list)
        letters = ["A", "B", "C"]
        new_answers, candidates = {}, {}
        for letter, (orig_label, text) in zip(letters, ans_list):
            new_answers[letter] = text
            candidates[letter] = orig_label
        it["answers"] = new_answers
        it["candidates"] = candidates
        shuffled.append(it)

    Path(args.output).write_text(json.dumps(shuffled, indent=2))
    print(f"Shuffled file written to {args.output} with {len(shuffled)} items")

if __name__ == "__main__":
    main()
