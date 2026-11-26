from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from travel_agent.data_checks import (
    Dataset,
    check_duplicates,
    check_ids_and_cross_duplicates,
    check_records,
    counts,
    find_leaks,
    load_jsonl,
)
from travel_agent.lookups import load_lookups
from travel_agent.taxonomy import load_taxonomy

REF = date(2026, 12, 1)
TAX, LOOKUPS = load_taxonomy(), load_lookups()


def _rec(
    rid: str, text: str, requests: list[dict[str, Any]] | None, **extra: Any
) -> dict[str, Any]:
    return {
        "id": rid,
        "text": text,
        "lang": extra.pop("lang", "fr"),
        "in_scope": requests is not None,
        "label": {"requests": requests} if requests is not None else None,
        "source": extra.pop("source", "synthetic"),
        **extra,
    }


def _ds(
    records: list[dict[str, Any]], kind: str = "synthetic", source: str = "synthetic"
) -> Dataset:
    return Dataset("x.jsonl", kind, source, records)  # type: ignore[arg-type]


def _checks(ds: Dataset) -> list[str]:
    issues, _ = check_records(ds, TAX, LOOKUPS, REF)
    return [f"{i.level}:{i.check}" for i in issues]


def test_valid_record_has_no_issue() -> None:
    req = {
        "subcategory": "hotel_search",
        "entities": {"destination": "Lisbonne", "date_start": "3 mars", "date_end": "2 nuits"},
    }
    assert _checks(_ds([_rec("a", "hotel lisbonne", [req])])) == []


def test_schema_errors() -> None:
    ds = _ds(
        [
            {"id": "a", "text": "x"},  # missing keys
            _rec("b", "x", None, label={"requests": []}),  # off-topic with a label
            _rec("c", "x", [{"subcategory": "car_rental", "entities": {}}]),
            _rec("d", "x", [{"subcategory": "food", "entities": {"city": "Rome"}}]),
            _rec("e", "x", [{"subcategory": "food", "entities": {}}], lang="de"),
            _rec("f", "x", [{"subcategory": "food", "entities": {}}], source="human"),
        ]
    )
    found = _checks(ds)
    assert found.count("error:schema") == 5
    assert "error:subcategory" in found


def test_non_slot_entity_is_an_error() -> None:
    req = {"subcategory": "travel_insurance", "entities": {"destination": "Rome"}}
    assert _checks(_ds([_rec("a", "x", [req])])) == ["error:slots"]


def test_catalog_check_and_out_of_catalog_flag() -> None:
    req = {"subcategory": "weather", "entities": {"destination": "Reykjavik"}}
    assert _checks(_ds([_rec("a", "x", [req])])) == ["error:catalog"]
    flagged = _rec("b", "x", [req], meta={"out_of_catalog": True})
    assert _checks(_ds([flagged], "test", "synthetic")) == []
    wrong_flag = _rec(
        "c",
        "x",
        [{"subcategory": "weather", "entities": {"destination": "Rome"}}],
        meta={"out_of_catalog": True},
    )
    assert _checks(_ds([wrong_flag])) == ["warning:catalog"]


def test_typos_and_countries_resolve() -> None:
    ds = _ds(
        [
            _rec("a", "x", [{"subcategory": "weather", "entities": {"destination": "edinburg"}}]),
            _rec(
                "b", "x", [{"subcategory": "visa_documents", "entities": {"destination": "Japon"}}]
            ),
        ]
    )
    assert _checks(ds) == []


def test_bad_dates_are_reported() -> None:
    req = {"subcategory": "weather", "entities": {"destination": "Rome", "date_start": "bientôt"}}
    ds = _ds([_rec("a", "x", [req])])
    issues, bad = check_records(ds, TAX, LOOKUPS, REF)
    assert bad == [("a", "date_start", "bientôt")]
    assert issues[0].level == "error"
    issues, _ = check_records(
        _ds([_rec("a", "x", [req], source="human")], "test", "human"), TAX, LOOKUPS, REF
    )
    assert issues[0].level == "warning"  # test files: reported, not blocking


def test_duplicates() -> None:
    ds = _ds(
        [
            _rec("a", "Quel temps à Rome demain ?", None),
            _rec("b", "quel temps a rome demain", None),
            _rec("c", "Quel temps à Rome demain ??", None),
            _rec("d", "Raconte-moi une blague", None),
        ]
    )
    found = [(i.level, i.check, i.id) for i in check_duplicates(ds)]
    assert ("error", "duplicates", "b") in found
    assert ("error", "duplicates", "c") in found


def test_ids_unique_across_files() -> None:
    a = Dataset("a.jsonl", "synthetic", "synthetic", [_rec("x1", "hello", None)])
    b = Dataset("b.jsonl", "test", "human", [_rec("x1", "hello", None)])
    checks = sorted(i.check for i in check_ids_and_cross_duplicates([a, b]))
    assert checks == ["duplicates", "ids"]


def test_find_leaks_with_fake_embeddings() -> None:
    vectors = {"rome": [1.0, 0.0], "rome!": [0.99, 0.05], "paris": [0.0, 1.0]}
    syn = Dataset(
        "train.jsonl",
        "synthetic",
        "synthetic",
        [_rec("s1", "rome", None), _rec("s2", "paris", None)],
    )
    tst = Dataset("t.jsonl", "test", "human", [_rec("t1", "rome!", None)])
    leaks = find_leaks([syn], [tst], lambda texts: [vectors[t] for t in texts])
    assert [(x.synthetic_id, x.test_id) for x in leaks] == [("s1", "t1")]


def test_counts() -> None:
    ds = _ds(
        [
            _rec(
                "a",
                "x",
                [
                    {"subcategory": "food", "entities": {}},
                    {"subcategory": "weather", "entities": {}},
                ],
            ),
            _rec("b", "y", [{"subcategory": "food", "entities": {}}], lang="ar"),
            _rec("c", "z", None),
        ]
    )
    stats = counts([ds])
    assert stats["requests_by_source_lang_subcategory"]["synthetic"]["food"] == {"fr": 1, "ar": 1}
    assert stats["in_scope_records_by_request_count"] == {"x.jsonl|1": 1, "x.jsonl|2": 1}


def test_load_jsonl_reports_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "human.jsonl"
    path.write_text(json.dumps(_rec("a", "x", None)) + "\n{not json\n[1]\n", encoding="utf-8")
    ds, issues = load_jsonl(path, "test")
    assert ds.expected_source == "human" and len(ds.records) == 1
    assert [i.check for i in issues] == ["json", "json"]
