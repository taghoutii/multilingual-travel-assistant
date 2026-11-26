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
steps such as the data leakage check: `uv sync --group ml`.

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

## Limitations

- The test set is model-generated and checked by the author rather than written by real users.
- Questions outside the 15 supported subcategories are not handled (the assistant declines them).
- Destinations are limited to a fixed catalog of 32 cities.
