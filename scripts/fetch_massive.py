"""Build data/test/massive_off_topic.jsonl from MASSIVE (AmazonScience/massive, CC BY 4.0).

Downloads the official MASSIVE 1.1 archive once into data/raw/ (git-ignored), then samples the
off-topic test queries (see travel_agent.massive). The output is never edited by hand.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

from travel_agent.config import PROJECT_ROOT, get_settings
from travel_agent.massive import MASSIVE_URL, read_locales, select

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "data" / "test" / "massive_off_topic.jsonl"


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp.rename(dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-scenario", type=int, default=5)
    parser.add_argument("--archive", type=Path, help="use a local copy of the MASSIVE archive")
    args = parser.parse_args()
    archive = args.archive or download(MASSIVE_URL, RAW_DIR / Path(MASSIVE_URL).name)
    records = select(read_locales(archive), args.per_scenario, get_settings().seed)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
