# eval-mini-harness

A mini, fast regression harness for self‑hosted LLMs. Just point it at a vLLM endpoint, feed it a CSV of prompts and lightweight evals, and get back a markdown report showing pass/fail rates, diffs, and latency. Use it before or after serving changes (e.g. quantization) to catch quality regressions and performance drift.
