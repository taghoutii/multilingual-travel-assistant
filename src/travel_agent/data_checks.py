"""Dataset checks used by scripts/validate_data.py.

Errors make validation fail; warnings are reported for review. Test files are never modified;
only synthetic files may be rewritten (dropping leaked examples) by the script.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError
from rapidfuzz import fuzz, process

from travel_agent.dates import resolve_dates
from travel_agent.lookups import Lookups
from travel_agent.schemas import ParsedQuery
from travel_agent.taxonomy import Taxonomy
from travel_agent.text_norm import fold

Level = Literal["error", "warning"]
Embed = Callable[[list[str]], Sequence[Sequence[float]]]

LANGS = frozenset({"fr", "en", "ar"})
REQUIRED_KEYS = ("id", "text", "lang", "in_scope", "label", "source")
TEST_SOURCES = {
    "generated_validated.jsonl": "generated_validated",
    "human.jsonl": "human",
    "massive_off_topic.jsonl": "massive",
}
NEAR_DUP_RATIO = 95
LEAK_COSINE = 0.9
DATE_FIELDS = ("date_start", "date_end")
CURRENCY_FIELDS = ("currency_from", "currency_to", "budget_currency")


@dataclass(frozen=True)
class Issue:
    level: Level
    check: str
    file: str
    id: str | None
    detail: str


@dataclass
class Dataset:
    name: str
    kind: Literal["synthetic", "test"]
    expected_source: str
    records: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Leak:
    synthetic_file: str
    synthetic_id: str
    test_file: str
    test_id: str
    cosine: float


def load_jsonl(path: Path, kind: Literal["synthetic", "test"]) -> tuple[Dataset, list[Issue]]:
    source = "synthetic" if kind == "synthetic" else TEST_SOURCES.get(path.name, "human")
    ds = Dataset(path.name, kind, source)
    issues: list[Issue] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(Issue("error", "json", path.name, f"line {n}", str(exc)))
                continue
            if not isinstance(rec, dict):
                issues.append(Issue("error", "json", path.name, f"line {n}", "not an object"))
                continue
            ds.records.append(rec)
    return ds, issues


def _out_of_catalog(rec: dict[str, Any]) -> bool:
    meta = rec.get("meta")
    return isinstance(meta, dict) and meta.get("out_of_catalog") is True


def check_records(
    ds: Dataset, taxonomy: Taxonomy, lookups: Lookups, ref: date
) -> tuple[list[Issue], list[tuple[str, str, str]]]:
    """Schema, subcategories, slots, catalog places, currencies, dates.

    Returns issues and the list of (id, field, span) date spans that do not normalize.
    """
    issues: list[Issue] = []
    bad_dates: list[tuple[str, str, str]] = []
    strict: Level = "error" if ds.kind == "synthetic" else "warning"

    def add(level: Level, check: str, rid: str | None, detail: str) -> None:
        issues.append(Issue(level, check, ds.name, rid, detail))

    for rec in ds.records:
        rid = str(rec.get("id", "?"))
        missing = [k for k in REQUIRED_KEYS if k not in rec]
        if missing:
            add("error", "schema", rid, f"missing keys {missing}")
            continue
        if not isinstance(rec["text"], str) or not rec["text"].strip():
            add("error", "schema", rid, "empty text")
        if rec["lang"] not in LANGS:
            add("error", "schema", rid, f"lang {rec['lang']!r} not in fr/en/ar")
        if rec["source"] != ds.expected_source:
            add(
                "error", "schema", rid, f"source {rec['source']!r}, expected {ds.expected_source!r}"
            )
        if "meta" in rec and not isinstance(rec["meta"], dict):
            add("error", "schema", rid, "meta must be an object")
        if rec["in_scope"] is False:
            if rec["label"] is not None:
                add("error", "schema", rid, "off-topic record must have label null")
            continue
        if rec["in_scope"] is not True:
            add("error", "schema", rid, "in_scope must be true or false")
            continue
        unknown = [
            r.get("subcategory")
            for r in (rec["label"] or {}).get("requests", [])
            if isinstance(r, dict) and r.get("subcategory") not in taxonomy.subcategories
        ]
        if unknown:
            add("error", "subcategory", rid, f"unknown subcategories {unknown}")
            continue
        try:
            query = ParsedQuery.model_validate(rec["label"])
        except ValidationError as exc:
            add("error", "schema", rid, f"label: {exc.errors()[0]['msg']}")
            continue
        flagged = _out_of_catalog(rec)
        unresolved_places = 0
        for req in query.requests:
            sub = req.subcategory.value
            ents = req.entities.model_dump(exclude_none=True)
            extra = set(ents) - set(taxonomy.subcategories[sub].slots)
            if extra:
                add("error", "slots", rid, f"{sub}: not slots of this subcategory {sorted(extra)}")
            for key in ("origin", "destination"):
                if key not in ents:
                    continue
                kinds = ("city", "country") if sub == "visa_documents" else ("city",)
                if lookups.resolve_place(ents[key], kinds) is None:  # type: ignore[arg-type]
                    unresolved_places += 1
                    if not flagged:
                        add("error", "catalog", rid, f"{sub}.{key} {ents[key]!r} not in catalog")
            for key in CURRENCY_FIELDS:
                if key in ents and lookups.resolve_currency(ents[key]) is None:
                    add(strict, "currency", rid, f"{sub}.{key} {ents[key]!r} not recognized")
            if any(k in ents for k in DATE_FIELDS):
                result = resolve_dates(
                    ents.get("date_start"), ents.get("date_end"), rec["lang"], ref
                )
                for key, value in (("date_start", result.start), ("date_end", result.end)):
                    if key in ents and value is None:
                        bad_dates.append((rid, key, ents[key]))
                        add(strict, "dates", rid, f"{sub}.{key} {ents[key]!r} does not normalize")
                if result.start and result.end and result.end < result.start:
                    add(strict, "dates", rid, f"{sub}: date_end before date_start")
        if flagged and unresolved_places == 0:
            add(
                "warning",
                "catalog",
                rid,
                "flagged out_of_catalog but every place is in the catalog",
            )
    return issues, bad_dates


def check_duplicates(ds: Dataset) -> list[Issue]:
    issues: list[Issue] = []
    seen: dict[str, str] = {}
    keys: list[str] = []
    ids: list[str] = []
    for rec in ds.records:
        rid, key = str(rec.get("id")), fold(str(rec.get("text", "")))
        if key in seen:
            issues.append(Issue("error", "duplicates", ds.name, rid, f"same text as {seen[key]}"))
            continue
        match = process.extractOne(key, keys, scorer=fuzz.ratio, score_cutoff=NEAR_DUP_RATIO)
        if match is not None:
            other = ids[match[2]]
            issues.append(
                Issue(
                    "warning",
                    "near_duplicates",
                    ds.name,
                    rid,
                    f"{match[1]:.0f}% similar to {other}",
                )
            )
        seen[key] = rid
        keys.append(key)
        ids.append(rid)
    return issues


def check_ids_and_cross_duplicates(datasets: list[Dataset]) -> list[Issue]:
    issues: list[Issue] = []
    owner: dict[str, str] = {}
    text_owner: dict[str, tuple[str, str]] = {}
    for ds in datasets:
        for rec in ds.records:
            rid = str(rec.get("id"))
            if rid in owner:
                issues.append(Issue("error", "ids", ds.name, rid, f"id also used in {owner[rid]}"))
            owner[rid] = ds.name
            key = fold(str(rec.get("text", "")))
            if key in text_owner and text_owner[key][0] != ds.name:
                other_file, other_id = text_owner[key]
                issues.append(
                    Issue(
                        "error",
                        "duplicates",
                        ds.name,
                        rid,
                        f"same text as {other_id} in {other_file}",
                    )
                )
            text_owner.setdefault(key, (ds.name, rid))
    return issues


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def find_leaks(
    synthetic: list[Dataset], test: list[Dataset], embed: Embed, threshold: float = LEAK_COSINE
) -> list[Leak]:
    """Synthetic examples whose embedding cosine to any test example exceeds `threshold`."""
    syn = [(ds.name, str(r["id"]), str(r["text"])) for ds in synthetic for r in ds.records]
    tst = [(ds.name, str(r["id"]), str(r["text"])) for ds in test for r in ds.records]
    if not syn or not tst:
        return []
    syn_vecs = embed([t for *_, t in syn])
    tst_vecs = embed([t for *_, t in tst])
    leaks: list[Leak] = []
    try:
        import numpy as np
    except ImportError:  # small inputs (tests): pure Python
        a = [_normalize(v) for v in syn_vecs]
        b = [_normalize(v) for v in tst_vecs]
        for i, va in enumerate(a):
            sims = [sum(x * y for x, y in zip(va, vb, strict=True)) for vb in b]
            j = max(range(len(sims)), key=sims.__getitem__)
            if sims[j] > threshold:
                leaks.append(Leak(syn[i][0], syn[i][1], tst[j][0], tst[j][1], round(sims[j], 4)))
        return leaks
    a_np = np.asarray(syn_vecs, dtype=np.float32)
    b_np = np.asarray(tst_vecs, dtype=np.float32)
    a_np /= np.linalg.norm(a_np, axis=1, keepdims=True) + 1e-12
    b_np /= np.linalg.norm(b_np, axis=1, keepdims=True) + 1e-12
    for start in range(0, len(a_np), 1024):
        block = a_np[start : start + 1024] @ b_np.T
        best = block.argmax(axis=1)
        for offset, j in enumerate(best):
            score = float(block[offset, j])
            if score > threshold:
                i = start + offset
                leaks.append(Leak(syn[i][0], syn[i][1], tst[j][0], tst[j][1], round(score, 4)))
    return leaks


def counts(datasets: list[Dataset]) -> dict[str, Any]:
    """Counts per source x language x subcategory, plus scope and request-count breakdowns."""
    by_label: Counter[tuple[str, str, str]] = Counter()
    by_scope: Counter[tuple[str, str, str]] = Counter()
    requests: Counter[tuple[str, int]] = Counter()
    for ds in datasets:
        for rec in ds.records:
            src, lang = str(rec.get("source")), str(rec.get("lang"))
            if rec.get("in_scope") is not True or not isinstance(rec.get("label"), dict):
                by_scope[(ds.name, lang, "off_topic")] += 1
                continue
            reqs = rec["label"].get("requests", [])
            by_scope[(ds.name, lang, "in_scope")] += 1
            requests[(ds.name, len(reqs))] += 1
            for req in reqs:
                by_label[(src, lang, str(req.get("subcategory")))] += 1
    table: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for (src, lang, sub), n in sorted(by_label.items()):
        table[src][sub][lang] = n
    return {
        "requests_by_source_lang_subcategory": {k: dict(v) for k, v in table.items()},
        "records_by_file_lang_scope": {
            f"{f}|{lang}|{s}": n for (f, lang, s), n in sorted(by_scope.items())
        },
        "in_scope_records_by_request_count": {
            f"{f}|{n}": c for (f, n), c in sorted(requests.items())
        },
    }
