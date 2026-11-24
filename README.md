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

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
