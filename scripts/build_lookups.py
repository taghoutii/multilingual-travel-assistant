"""Build data/lookups/cities.json and currencies.json from configs/destinations.yaml and
configs/currencies.yaml. Run after editing either config; `--check` fails if the files are stale."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from travel_agent.config import CONFIG_DIR, read_yaml
from travel_agent.destinations import load_catalog
from travel_agent.lookups import LOOKUPS_DIR, build_cities, build_currencies


def _dump(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def build() -> dict[Path, str]:
    return {
        LOOKUPS_DIR / "cities.json": _dump(build_cities(load_catalog())),
        LOOKUPS_DIR / "currencies.json": _dump(
            build_currencies(read_yaml(CONFIG_DIR / "currencies.yaml"))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only check the files are current")
    args = parser.parse_args()
    outputs = build()
    if args.check:
        stale = [p for p, text in outputs.items() if not p.exists() or p.read_text("utf-8") != text]
        for path in stale:
            print(f"stale: {path}")
        return 1 if stale else 0
    LOOKUPS_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
