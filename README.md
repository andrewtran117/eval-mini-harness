# eval-mini-harness

A mini, fast regression harness for local LLMs using Ollama. Feed it a CSV of prompts and lightweight evals, and get back detailed reports showing pass/fail rates, latency stats, and per-test results. Perfect for testing different models or catching quality regressions.

## Scripts
- `scripts/merge_runs.py` — merge multiple model answer JSONs into one combined answers file keyed by test id.
- `scripts/shuffle_answers.py` — shuffle answer labels (A/B/C) to reduce position bias before judging.
- `scripts/split_dataset.py` — split a merged answers file into N shards for parallel judging.
- `scripts/run_judge.py` — run strict/lenient pairwise judging over answers using Ollama models.
- `scripts/compare_runs.py` — compare two result sets to see what changed between runs.
- `scripts/render_judge_report.py` — render HTML/Markdown reports from judgment outputs.
- `scripts/perf_probe.py` — quick latency/throughput probe against a model/config.
- `scripts/run_eval.py` — small eval test runner to sanity-check the harness.
- `scripts/convert_hf_to_csv.py` — convert a Hugging Face-style dataset to the CSV format this harness expects.

## Endpoint setup (Ollama)
- The harness talks to a local Ollama instance. Make sure `ollama serve` is running and the model names in your configs match what you have pulled (see `ollama list`).
