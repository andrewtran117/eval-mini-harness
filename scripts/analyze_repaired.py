#!/usr/bin/env python3
"""
Quick analysis for judge_results_repaired.jsonl.

Computes:
- Parse error rates by judge/rubric
- Winner counts per judge/rubric/pair (resolved to model keys)
- Overall model win counts (ignoring ties)
- Judge agreement fractions per rubric/pair

Usage:
  python scripts/analyze_repaired.py --input path/to/judge_results_repaired.jsonl
"""

import argparse
import json
import pandas as pd


def load_df(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def resolve_winner_model(row) -> str:
    """
    Map winner letter (A/B/tie) to the actual model key for the given pair.
    """
    winner = row.get("winner")
    if winner == "tie":
        return "tie"
    mapping = row.get("candidates", {})
    pair = row.get("pair")
    if pair == "A_vs_B":
        pair_map = {"A": mapping.get("A"), "B": mapping.get("B")}
    elif pair == "A_vs_C":
        pair_map = {"A": mapping.get("A"), "B": mapping.get("C")}
    elif pair == "B_vs_C":
        pair_map = {"A": mapping.get("B"), "B": mapping.get("C")}
    else:
        return None
    return pair_map.get(winner)


def agreement(group: pd.DataFrame) -> pd.Series:
    vals = group["winner_model"].tolist()
    uniq = set(vals)
    return pd.Series({
        "n_judges": len(vals),
        "agree": len(uniq) == 1,
    })


def main():
    parser = argparse.ArgumentParser(description="Analyze repaired judge results")
    parser.add_argument("--input", required=True, help="Path to judge_results_repaired.jsonl")
    args = parser.parse_args()

    df = load_df(args.input)

    # Resolve winner_model
    df["winner_model"] = df.apply(resolve_winner_model, axis=1)

    # Parse error rate
    print("Parse error rate by judge/rubric:")
    print(df.groupby(["judge_model", "rubric"])["parse_error"].mean(), "\n")

    # Winner counts per judge/rubric/pair
    df_ok = df[df["parse_error"] == False]
    win_counts = (
        df_ok.groupby(["judge_model", "rubric", "pair"])["winner_model"]
             .value_counts()
             .unstack(fill_value=0)
    )
    print("Winner counts per judge/rubric/pair:")
    print(win_counts, "\n")

    # Overall model win counts (ignoring ties)
    model_wins = (
        df_ok[df_ok["winner_model"] != "tie"]
             .groupby(["winner_model", "rubric"])
             .size()
             .unstack(fill_value=0)
    )
    print("Model win counts (ignoring ties):")
    print(model_wins, "\n")

    # Agreement fractions
    agree = (
        df_ok.groupby(["id", "rubric", "pair"])
             .apply(agreement)
             .groupby(["rubric", "pair"])["agree"]
             .mean()
    )
    print("Agreement fraction by rubric/pair:")
    print(agree, "\n")


if __name__ == "__main__":
    main()
