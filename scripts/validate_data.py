"""Validate the synthetic and test datasets.

Checks: JSON format and schema, known subcategories, slots, catalog places (except records flagged
meta.out_of_catalog), currency words, gold date spans that do not normalize, exact and near
duplicates, leakage between synthetic data and all test sources (embedding cosine > 0.9), and
counts per source x language x subcategory.

    uv run --group ml python scripts/validate_data.py              # report; exit 1 on errors
    uv run --group ml python scripts/validate_data.py --drop-leaks # also drop leaked synthetic rows

Test files are never modified. Leaked synthetic examples are reported, and removed only with
--drop-leaks. The leakage check needs the `ml` dependency group (sentence-transformers).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from travel_agent.config import PROJECT_ROOT, get_settings
from travel_agent.data_checks import (
    Dataset,
    Issue,
    check_duplicates,
    check_ids_and_cross_duplicates,
    check_records,
    counts,
    find_leaks,
    load_jsonl,
)
from travel_agent.lookups import get_lookups
from travel_agent.taxonomy import get_taxonomy

SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
TEST_DIR = PROJECT_ROOT / "data" / "test"
MODELS_DIR = PROJECT_ROOT / "models"


def embedder() -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        get_settings().embeddings_model, cache_folder=str(MODELS_DIR / "hf")
    )

    def embed(texts: list[str]) -> Any:
        return model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )

    return embed


def load_all() -> tuple[list[Dataset], list[Dataset], list[Issue]]:
    issues: list[Issue] = []
    synthetic, test = [], []
    for path in sorted(SYNTHETIC_DIR.glob("*.jsonl")):
        ds, found = load_jsonl(path, "synthetic")
        synthetic.append(ds)
        issues += found
    for path in sorted(TEST_DIR.glob("*.jsonl")):
        ds, found = load_jsonl(path, "test")
        test.append(ds)
        issues += found
    return synthetic, test, issues


def rewrite(ds: Dataset, drop: set[str]) -> int:
    kept = [r for r in ds.records if str(r["id"]) not in drop]
    with (SYNTHETIC_DIR / ds.name).open("w", encoding="utf-8", newline="\n") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(ds.records) - len(kept)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--drop-leaks", action="store_true", help="remove leaked synthetic examples"
    )
    parser.add_argument("--skip-leakage", action="store_true", help="skip the embedding check")
    parser.add_argument("--report", type=Path, help="also write the full report as JSON")
    args = parser.parse_args()

    taxonomy, lookups, ref = get_taxonomy(), get_lookups(), get_settings().eval_reference_date
    synthetic, test, issues = load_all()
    datasets = synthetic + test
    if not synthetic:
        print(f"no synthetic files in {SYNTHETIC_DIR}")
        return 1
    bad_dates: dict[str, list[tuple[str, str, str]]] = {}
    for ds in datasets:
        found, bad = check_records(ds, taxonomy, lookups, ref)
        issues += found + check_duplicates(ds)
        bad_dates[ds.name] = bad
    issues += check_ids_and_cross_duplicates(datasets)

    leaks: list[Any] = []
    if not test:
        print("leakage: no test files yet in data/test/ (run again once they exist)")
    elif args.skip_leakage:
        issues.append(Issue("warning", "leakage", "-", None, "skipped (--skip-leakage)"))
    else:
        try:
            embed = embedder()
        except ImportError:
            print("leakage check needs the ml group: uv run --group ml python <this script>")
            return 1
        leaks = find_leaks(synthetic, test, embed)
        if leaks and args.drop_leaks:
            by_file: dict[str, set[str]] = {}
            for leak in leaks:
                by_file.setdefault(leak.synthetic_file, set()).add(leak.synthetic_id)
            for ds in synthetic:
                if ds.name in by_file:
                    print(f"dropped {rewrite(ds, by_file[ds.name])} leaked examples from {ds.name}")
        elif leaks:
            issues.append(
                Issue(
                    "error",
                    "leakage",
                    "-",
                    None,
                    f"{len(leaks)} synthetic examples leak; rerun with --drop-leaks",
                )
            )

    stats = counts(datasets)
    _print_report(datasets, issues, bad_dates, leaks, stats)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "issues": [vars(i) for i in issues],
                    "leaks": [vars(x) for x in leaks],
                    "bad_dates": bad_dates,
                    "counts": stats,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 1 if any(i.level == "error" for i in issues) else 0


def _print_report(
    datasets: list[Dataset],
    issues: list[Issue],
    bad_dates: dict[str, Any],
    leaks: list[Any],
    stats: dict[str, Any],
) -> None:
    print("== files")
    for ds in datasets:
        langs = Counter(str(r.get("lang")) for r in ds.records)
        print(f"  {ds.name:28s} {len(ds.records):5d} records  {dict(sorted(langs.items()))}")
    print("== requests per source x subcategory x language")
    for src, subs in stats["requests_by_source_lang_subcategory"].items():
        print(f"  [{src}]")
        for sub, langs in subs.items():
            print(f"    {sub:22s} " + "  ".join(f"{lang}={n}" for lang, n in sorted(langs.items())))
    print("== in-scope records by request count")
    for key, n in stats["in_scope_records_by_request_count"].items():
        print(f"  {key:30s} {n}")
    print("== date spans that do not normalize")
    for name, bad in bad_dates.items():
        print(f"  {name}: {len(bad)}")
        for rid, key, span in bad[:50]:
            print(f"    {rid} {key}: {span!r}")
    print(f"== leakage (cosine > 0.9): {len(leaks)}")
    for leak in leaks[:30]:
        print(f"  {leak.synthetic_id} ~ {leak.test_id} ({leak.cosine})")
    summary = Counter((i.level, i.check) for i in issues)
    print("== issues")
    for (level, check), n in sorted(summary.items()):
        print(f"  {level:7s} {check:16s} {n}")
    for issue in [i for i in issues if i.level == "error"][:40]:
        print(f"  ERROR {issue.file} {issue.id}: {issue.detail}")
    errors = sum(1 for i in issues if i.level == "error")
    print(
        f"== result: {'FAIL' if errors else 'PASS'} ({errors} errors, "
        f"{sum(1 for i in issues if i.level == 'warning')} warnings)"
    )


if __name__ == "__main__":
    sys.exit(main())
