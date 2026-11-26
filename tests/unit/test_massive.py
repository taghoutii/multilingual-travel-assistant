"""MASSIVE selection on a tiny fake archive (offline)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from travel_agent.massive import SCENARIOS, read_locales, select


def _rows(locale: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rid = 0
    for scenario in [*SCENARIOS, "transport", "weather"]:
        for partition in ("train", "test", "test", "test", "test", "test", "test", "dev"):
            rid += 1
            rows.append(
                {
                    "id": str(rid),
                    "locale": locale,
                    "partition": partition,
                    "scenario": scenario,
                    "utt": f"{locale} {scenario} {rid}",
                }
            )
    return rows


def _archive(tmp_path: Path) -> Path:
    path = tmp_path / "massive.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for locale in ("fr-FR", "en-US", "ar-SA", "de-DE"):
            data = "\n".join(json.dumps(r) for r in _rows(locale)).encode()
            info = tarfile.TarInfo(f"1.1/data/{locale}.jsonl")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_select_filters_split_scenarios_and_locales(tmp_path: Path) -> None:
    records = select(read_locales(_archive(tmp_path)), per_scenario=5, seed=42)
    assert len(records) == 3 * len(SCENARIOS) * 5
    assert {r["lang"] for r in records} == {"fr", "en", "ar"}
    assert {r["meta"]["scenario"] for r in records} == set(SCENARIOS)
    assert all(r["source"] == "massive" and r["in_scope"] is False for r in records)
    assert all(r["label"] is None for r in records)
    assert len({r["id"] for r in records}) == len(records)


def test_select_only_uses_test_split(tmp_path: Path) -> None:
    rows = read_locales(_archive(tmp_path))
    test_ids = {f"massive-fr-FR-{r['id']}" for r in rows["fr-FR"] if r["partition"] == "test"}
    records = select(rows, per_scenario=5, seed=42)
    assert {r["id"] for r in records if r["lang"] == "fr"} <= test_ids


def test_select_is_deterministic(tmp_path: Path) -> None:
    rows = read_locales(_archive(tmp_path))
    assert select(rows, 3, seed=1) == select(rows, 3, seed=1)
    assert select(rows, 3, seed=1) != select(rows, 3, seed=2)


def test_not_enough_utterances(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only 6 test utterances"):
        select(read_locales(_archive(tmp_path)), per_scenario=7, seed=42)
