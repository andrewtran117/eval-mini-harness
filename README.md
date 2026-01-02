# eval-mini-harness

A mini, fast regression harness for OpenAI-compatible LLM endpoints (e.g., vLLM). Feed it a CSV of prompts and lightweight evals, and get back detailed reports showing pass/fail rates, latency stats, and per-test results. Perfect for testing different models or catching quality regressions.

## Scripts
- `scripts/merge_runs.py` — merge multiple model answer JSONs into one combined answers file keyed by test id.
- `scripts/shuffle_answers.py` — shuffle answer labels (A/B/C) to reduce position bias before judging.
- `scripts/split_dataset.py` — split a merged answers file into N shards for parallel judging.
- `scripts/run_judge.py` — run strict/lenient pairwise judging over answers using OpenAI-compatible models (e.g., vLLM).
- `scripts/compare_runs.py` — compare two result sets to see what changed between runs.
- `scripts/render_judge_report.py` — render HTML/Markdown reports from judgment outputs.
- `scripts/perf_probe.py` — quick latency/throughput probe against a model/config.
- `scripts/run_smoke.py` — small smoke test runner to sanity-check the harness.
- `scripts/convert_hf_to_csv.py` — convert a Hugging Face-style dataset to the CSV format this harness expects.

## Endpoint setup (vLLM/OpenAI-compatible)
- The harness talks to an OpenAI-compatible endpoint (e.g., vLLM). Set `OPENAI_BASE_URL` (default `http://localhost:8000/v1`) and `OPENAI_API_KEY` (use `EMPTY` if your server is unauthenticated).
- You can also pass `--base_url` and `--api_key` flags to `scripts/run_judge.py`; other scripts read the env vars.
- Make sure the model name in your config matches what your vLLM server exposes (see `GET /v1/models`).
