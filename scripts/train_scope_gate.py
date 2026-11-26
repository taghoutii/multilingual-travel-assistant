"""Train the scope gate and choose nu, gamma and the threshold on validation.

    uv run --group ml python scripts/train_scope_gate.py

Fits a One-Class SVM on embeddings of the in-scope training queries (data/synthetic/train.jsonl),
then, for every (nu, gamma) in configs/settings.yaml:
  - threshold = score that accepts `target_in_scope_accept` of the in-scope validation queries
    (data/synthetic/val.jsonl);
  - off-topic rejection on data/synthetic/off_topic.jsonl at that threshold.
Keeps the pair with the highest rejection and saves it to models/<scope_gate.model_file>.
No test file is read.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from travel_agent.config import PROJECT_ROOT, get_settings
from travel_agent.data_checks import load_jsonl
from travel_agent.embeddings import load_embedder
from travel_agent.scope_gate import ScopeGate, model_path, tune
from travel_agent.seed import set_seed

SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"


def texts(name: str, in_scope: bool) -> list[str]:
    ds, issues = load_jsonl(SYNTHETIC_DIR / name, "synthetic")
    if issues:
        raise SystemExit(f"{name}: {len(issues)} format errors, run scripts/validate_data.py")
    return [str(r["text"]) for r in ds.records if bool(r["in_scope"]) == in_scope]


def main() -> int:
    settings = get_settings()
    cfg = settings.scope_gate
    set_seed(settings.seed)

    train_in = texts("train.jsonl", True)
    val_in = texts("val.jsonl", True)
    val_off = texts("off_topic.jsonl", False)
    print(f"train in-scope {len(train_in)}, val in-scope {len(val_in)}, off-topic {len(val_off)}")

    embed = load_embedder()
    start = time.perf_counter()
    x_train, x_val_in, x_val_off = (np.asarray(embed(t)) for t in (train_in, val_in, val_off))
    print(f"embedded in {time.perf_counter() - start:.1f}s, dim {x_train.shape[1]}")

    result = tune(
        x_train, x_val_in, x_val_off, cfg.nu_grid, cfg.gamma_grid, cfg.target_in_scope_accept
    )
    print(f"{'nu':>6} {'gamma':>6} {'thresh':>8} {'in acc':>7} {'off rej':>8}")
    for row in result.grid:
        print(
            f"{row.nu:>6} {row.gamma!s:>6} {row.threshold:>8.4f} "
            f"{row.in_scope_accept:>7.3f} {row.off_topic_reject:>8.3f}"
        )
    best = next(r for r in result.grid if (r.nu, r.gamma) == (result.nu, result.gamma))
    print(f"chosen: nu={result.nu} gamma={result.gamma} threshold={result.threshold:.4f}")

    meta = {
        "embeddings_model": settings.embeddings_model,
        "nu": result.nu,
        "gamma": result.gamma,
        "target_in_scope_accept": cfg.target_in_scope_accept,
        "n_train": len(train_in),
        "val_in_scope_accept": best.in_scope_accept,
        "val_off_topic_reject": best.off_topic_reject,
        "grid": [row.__dict__ for row in result.grid],
    }
    path = model_path()
    ScopeGate(result.svm, result.threshold, meta=meta).save(path)
    print(f"saved {path}")
    print(json.dumps({k: v for k, v in meta.items() if k != "grid"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
