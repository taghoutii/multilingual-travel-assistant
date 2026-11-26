"""Select the off-topic test queries from MASSIVE (AmazonScience/massive, CC BY 4.0).

Official test split, locales fr-FR / en-US / ar-SA, a fixed list of non-travel scenarios, seeded
sampling of `per_scenario` utterances per scenario and locale. MASSIVE is used for nothing else.
"""

from __future__ import annotations

import json
import random
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
)
LOCALES = {"fr-FR": "fr", "en-US": "en", "ar-SA": "ar"}
SCENARIOS = (
    "alarm",
    "music",
    "audio",
    "iot",
    "email",
    "lists",
    "cooking",
    "takeaway",
    "play",
    "news",
    "social",
    "calendar",
)


def read_locales(
    tar_path: Path, locales: Iterable[str] = LOCALES
) -> dict[str, list[dict[str, Any]]]:
    """Read `<locale>.jsonl` members from the MASSIVE tarball without extracting it."""
    wanted = {f"{loc}.jsonl": loc for loc in locales}
    rows: dict[str, list[dict[str, Any]]] = {loc: [] for loc in wanted.values()}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name in wanted and member.isfile():
                handle = tar.extractfile(member)
                assert handle is not None
                for line in handle:
                    if line.strip():
                        rows[wanted[name]].append(json.loads(line))
    missing = [loc for loc, items in rows.items() if not items]
    if missing:
        raise ValueError(f"locales not found in {tar_path}: {missing}")
    return rows


def select(
    rows: dict[str, list[dict[str, Any]]], per_scenario: int, seed: int
) -> list[dict[str, Any]]:
    """Seeded sample of test-split utterances per (locale, scenario), as test-set records."""
    records: list[dict[str, Any]] = []
    for locale in sorted(rows):
        for scenario in SCENARIOS:
            pool = sorted(
                (r for r in rows[locale] if r["partition"] == "test" and r["scenario"] == scenario),
                key=lambda r: int(r["id"]),
            )
            if len(pool) < per_scenario:
                raise ValueError(f"{locale}/{scenario}: only {len(pool)} test utterances")
            rng = random.Random(f"{seed}-{locale}-{scenario}")
            for row in sorted(rng.sample(pool, per_scenario), key=lambda r: int(r["id"])):
                records.append(
                    {
                        "id": f"massive-{locale}-{row['id']}",
                        "text": row["utt"],
                        "lang": LOCALES[locale],
                        "in_scope": False,
                        "label": None,
                        "source": "massive",
                        "meta": {
                            "scenario": scenario,
                            "massive_id": str(row["id"]),
                            "locale": locale,
                        },
                    }
                )
    return records
