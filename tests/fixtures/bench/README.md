# Bench fixtures

Small, synthetic files in the exact schemas of the real benchmarks. They exist
so the loaders can be tested without network access or dataset redistribution.

| file | schema of |
|---|---|
| `zsre.jsonl` | zsRE development (`input`, `output[0].answer`, `rephrases`) |
| `mmlu.parquet` | MMLU test (`question`, `subject`, `choices`, `answer`) |
| `wikitext.parquet` | WikiText-2 raw (`text`) |

Real data is fetched by `scripts/fetch_benchmarks.py` and never committed.
