# Demo Travel Agency — multilingual travel assistant

A text-to-text conversational assistant for a fictional travel agency, answering in French, English
and Arabic. It parses a travel request (possibly containing several sub-requests) into
structured data, calls external data sources, and writes one answer grounded in that data.

Status: work in progress.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.11 is installed by uv if missing).

```bash
uv sync
cp .env.example .env   
```

The `ml` dependency group (sentence-transformers, torch) is only needed for embedding-based
steps such as the data leakage check and the scope gate: `uv sync --group ml`.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## Data

See [DATA.md](DATA.md). Useful commands:

```bash
uv run python scripts/build_lookups.py                 # data/lookups/ from the configs
uv run python scripts/build_mocks.py                   # fictional flights and hotels
uv run python scripts/fetch_massive.py                 # off-topic test queries from MASSIVE
uv run --group ml python scripts/validate_data.py      # all dataset checks
```

Off-topic test queries come from [MASSIVE](https://github.com/alexa/massive) (FitzGerald et al.,
2022), released by Amazon under the CC BY 4.0 license.

## Language ID and scope gate

The first two pipeline stages pick the reply language (fastText `lid.176`, restricted to French,
English and Arabic) and decline requests outside the 15 subcategories (One-Class SVM on
multilingual sentence embeddings). Declined requests get a reply listing what the assistant can
help with, in the user's language (`configs/refusal_templates.yaml`).

```bash
uv run --group ml python scripts/download_models.py --embeddings   # models/ (git-ignored)
uv run --group ml python scripts/train_scope_gate.py               # models/scope_gate.joblib
uv run --group ml python scripts/eval_language_scope.py            # reports/language_scope_val.md
uv run --group ml python scripts/eval_language_scope.py --split test
```

## Limitations

- The test set is model-generated and checked by the author rather than written by real users.
- Questions outside the 15 supported subcategories are not handled (the assistant declines them).
  Travel requests just outside them (car rental, trains, buses, taxis, cruises) are often not
  recognized as off-topic by the scope gate.
- Destinations are limited to a fixed catalog of 32 cities.
