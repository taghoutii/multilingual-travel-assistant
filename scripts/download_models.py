"""Download the pretrained models used at runtime into models/ (git-ignored).

    uv run python scripts/download_models.py                  # fastText language ID
    uv run --group ml python scripts/download_models.py --embeddings   # also the embeddings model

The fastText file is skipped when already present. The embeddings model is cached by
sentence-transformers under models/hf.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request

from travel_agent.config import get_settings
from travel_agent.language_id import LID_URL, model_path


def download_language_id(force: bool = False) -> None:
    target = model_path()
    if target.exists() and not force:
        print(f"{target} already present")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    print(f"downloading {LID_URL}")
    with urllib.request.urlopen(LID_URL, timeout=60) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(target)
    print(f"saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


def download_embeddings() -> None:
    from travel_agent.embeddings import load_embedder

    load_embedder()
    print(f"embeddings model ready: {get_settings().embeddings_model}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embeddings", action="store_true", help="also fetch the embeddings model")
    parser.add_argument("--force", action="store_true", help="re-download the fastText model")
    args = parser.parse_args(argv)
    download_language_id(args.force)
    if args.embeddings:
        download_embeddings()
    return 0


if __name__ == "__main__":
    sys.exit(main())
