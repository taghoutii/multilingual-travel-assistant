"""Evaluate language ID (stage 1) and the scope gate (stage 2).

    uv run --group ml python scripts/eval_language_scope.py              # validation (synthetic)
    uv run --group ml python scripts/eval_language_scope.py --split test # test files in data/test/

Validation uses data/synthetic/val.jsonl (in scope) and data/synthetic/off_topic.jsonl (off topic).
Both were used to choose the gate's threshold and hyper-parameters, so validation numbers are
tuning numbers; the test split gives the unbiased ones. Writes reports/language_scope_<split>.md.
Rates come with Wilson 95% intervals.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any

from travel_agent.config import PROJECT_ROOT, get_settings
from travel_agent.data_checks import load_jsonl
from travel_agent.embeddings import load_embedder
from travel_agent.eval.gate_metrics import Rate, grouped, percentile, rate
from travel_agent.language_id import LanguageIdentifier, LanguageResult, fasttext_predictor
from travel_agent.scope_gate import ScopeGate, model_path

DATA = PROJECT_ROOT / "data"
REPORTS = PROJECT_ROOT / "reports"

# Small probe of messages in languages the assistant does not support (expected: "unsupported").
OTHER_LANGUAGE_PROBE = [
    ("es", "Quiero reservar un vuelo de Madrid a Roma para el próximo viernes"),
    ("es", "¿Qué tiempo hará en Barcelona mañana?"),
    ("es", "vuelo madrid paris"),
    ("de", "Ich möchte ein Hotel in Berlin für drei Nächte buchen"),
    ("de", "Wie ist das Wetter in München?"),
    ("de", "flug nach rom"),
    ("it", "Vorrei prenotare un volo per Parigi la settimana prossima"),
    ("it", "Quali documenti servono per il visto?"),
    ("pt", "Quero um hotel barato em Lisboa para dois adultos"),
    ("pt", "Quanto custa a bagagem extra?"),
    ("tr", "İstanbul'dan Paris'e uçak bileti arıyorum"),
    ("tr", "Otel rezervasyonumu iptal etmek istiyorum"),
    ("nl", "Ik wil graag een vlucht naar Londen boeken"),
    ("ru", "Я хочу забронировать отель в Дубае"),
    ("fa", "می‌خواهم یک بلیط هواپیما به دبی رزرو کنم"),
    ("ur", "مجھے دبئی کے لیے ہوٹل چاہیے"),
    ("zh", "我想预订去东京的机票"),
    ("ja", "東京のホテルを予約したいです"),
]


def load_split(split: str) -> list[dict[str, Any]]:
    if split == "val":
        files = [DATA / "synthetic" / "val.jsonl", DATA / "synthetic" / "off_topic.jsonl"]
        kind = "synthetic"
    else:
        files = sorted((DATA / "test").glob("*.jsonl"))
        kind = "test"
    records: list[dict[str, Any]] = []
    for path in files:
        if not path.exists():
            continue
        ds, issues = load_jsonl(path, kind)  # type: ignore[arg-type]
        if issues:
            raise SystemExit(f"{path.name}: format errors, run scripts/validate_data.py")
        records += ds.records
    return records


def style(rec: dict[str, Any]) -> str:
    meta = rec.get("meta") or {}
    if meta.get("telegraphic"):
        return "telegraphic"
    if meta.get("noisy"):
        return "noisy"
    return "clean" if rec["source"] == "synthetic" else rec["source"]


def n_requests(rec: dict[str, Any]) -> str:
    return str(len(rec["label"]["requests"])) if rec.get("label") else "-"


def off_kind(rec: dict[str, Any]) -> str:
    meta = rec.get("meta") or {}
    return str(meta.get("kind") or meta.get("scenario") or rec["source"])


def timed(fn: Callable[[str], Any], texts: list[str]) -> tuple[list[Any], list[float]]:
    out, ms = [], []
    for text in texts:
        start = time.perf_counter()
        out.append(fn(text))
        ms.append(1000 * (time.perf_counter() - start))
    return out, ms


def table(title: str, rows: dict[str, Rate], label: str = "group") -> list[str]:
    lines = [f"**{title}**", "", f"| {label} | rate [95% CI] (hits/n) |", "|---|---|"]
    lines += [f"| {g} | {r.fmt()} |" for g, r in rows.items()]
    return [*lines, ""]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max-examples", type=int, default=15)
    args = parser.parse_args()

    records = load_split(args.split)
    if not records:
        print(f"no records for split {args.split}")
        return 1
    cfg = get_settings().language_id
    lid = LanguageIdentifier(
        fasttext_predictor(), cfg.unsupported_min_prob, cfg.keep_previous_below
    )
    gate = ScopeGate.load(model_path(), load_embedder())
    if gate.meta.get("embeddings_model") != get_settings().embeddings_model:
        raise SystemExit("scope gate was trained with another embeddings model; retrain it")

    texts = [str(r["text"]) for r in records]
    lang_results, lid_ms = timed(lid.identify, texts)
    gate.decide(texts[0])  # warm-up
    decisions, gate_ms = timed(gate.decide, texts)

    in_scope = [i for i, r in enumerate(records) if r["in_scope"]]
    off_topic = [i for i, r in enumerate(records) if not r["in_scope"]]
    lid_ok = [lang_results[i].lang == records[i]["lang"] for i in range(len(records))]
    # MASSIVE labels follow the locale; ar-SA contains some English utterances.
    lid_scored = [i for i, r in enumerate(records) if r["source"] != "massive"]

    out: list[str] = [
        f"# Language ID and scope gate: {args.split}",
        "",
        "Rates with Wilson 95% intervals. "
        + (
            "Validation = data/synthetic/val.jsonl (in scope) + data/synthetic/off_topic.jsonl; "
            "both were used to choose the gate's threshold and hyper-parameters, so these are "
            "tuning numbers."
            if args.split == "val"
            else "Test = every file in data/test/. Language accuracy excludes MASSIVE "
            "(its labels follow the locale)."
        ),
        "",
        f"Records: {len(records)} ({len(in_scope)} in scope, {len(off_topic)} off topic).",
        "",
        "## Language ID",
        "",
        f"fastText `{cfg.model}`, unsupported if another language has p >= "
        f"{cfg.unsupported_min_prob}.",
        "",
        f"Accuracy (supported-language messages): {rate(lid_ok[i] for i in lid_scored).fmt()}",
        "",
        f"Wrongly flagged as unsupported: "
        f"{rate(not lang_results[i].supported for i in lid_scored).fmt()}",
        "",
    ]
    out += table(
        "Accuracy per language", grouped((records[i]["lang"], lid_ok[i]) for i in lid_scored)
    )
    out += table("Accuracy per style", grouped((style(records[i]), lid_ok[i]) for i in lid_scored))
    reasons = grouped((lang_results[i].reason, lid_ok[i]) for i in lid_scored)
    out += table("Accuracy per decision rule", reasons, "rule")

    probe: list[LanguageResult] = [lid.identify(t) for _, t in OTHER_LANGUAGE_PROBE]
    out += [
        f"Other-language probe ({len(probe)} messages in es/de/it/pt/tr/nl/ru/fa/ur/zh/ja): "
        f"detected as unsupported {rate(not p.supported for p in probe).fmt()}",
        "",
    ]
    lid_errors = [i for i in lid_scored if not lid_ok[i]]
    if lid_errors:
        out += ["Errors:", "", "| gold | predicted | fastText top | text |", "|---|---|---|---|"]
        for i in lid_errors[: args.max_examples]:
            res = lang_results[i]
            out.append(
                f"| {records[i]['lang']} | {res.lang} | {res.detected} {res.confidence:.2f} "
                f"| {records[i]['text']} |"
            )
        out.append("")

    accepted = [d.in_scope for d in decisions]
    meta = gate.meta
    out += [
        "## Scope gate",
        "",
        f"One-Class SVM (nu={meta.get('nu')}, gamma={meta.get('gamma')}) on "
        f"`{meta.get('embeddings_model')}` embeddings of {meta.get('n_train')} in-scope training "
        f"queries; threshold {gate.threshold:.4f} (target in-scope acceptance "
        f"{meta.get('target_in_scope_accept')}).",
        "",
        f"In-scope accepted: {rate(accepted[i] for i in in_scope).fmt()}",
        "",
        f"Off-topic rejected: {rate(not accepted[i] for i in off_topic).fmt()}",
        "",
    ]
    out += table(
        "In-scope accepted per language",
        grouped((records[i]["lang"], accepted[i]) for i in in_scope),
    )
    out += table(
        "In-scope accepted per request count",
        grouped((n_requests(records[i]), accepted[i]) for i in in_scope),
        "requests",
    )
    out += table(
        "In-scope accepted per style", grouped((style(records[i]), accepted[i]) for i in in_scope)
    )
    if any((records[i].get("meta") or {}).get("out_of_catalog") for i in in_scope):
        ooc = grouped(
            (str(bool((records[i].get("meta") or {}).get("out_of_catalog"))), accepted[i])
            for i in in_scope
        )
        out += table("In-scope accepted, out-of-catalog city", ooc, "out of catalog")
    out += table(
        "Off-topic rejected per language",
        grouped((records[i]["lang"], not accepted[i]) for i in off_topic),
    )
    out += table(
        "Off-topic rejected per kind",
        grouped((off_kind(records[i]), not accepted[i]) for i in off_topic),
        "kind",
    )
    for title, idx in (
        ("In-scope rejected (lowest scores)", [i for i in in_scope if not accepted[i]]),
        ("Off-topic accepted (highest scores)", [i for i in off_topic if accepted[i]]),
    ):
        idx.sort(key=lambda i: decisions[i].score, reverse=title.startswith("Off"))
        if idx:
            out += [f"{title}:", "", "| score | lang | text |", "|---|---|---|"]
            out += [
                f"| {decisions[i].score:.4f} | {records[i]['lang']} | {records[i]['text']} |"
                for i in idx[: args.max_examples]
            ]
            out.append("")

    out += [
        "## Latency (CPU, one message at a time)",
        "",
        "| stage | p50 ms | p95 ms |",
        "|---|---|---|",
        f"| language ID | {percentile(lid_ms, 50):.2f} | {percentile(lid_ms, 95):.2f} |",
        f"| scope gate (embedding + SVM) | {percentile(gate_ms, 50):.1f} | "
        f"{percentile(gate_ms, 95):.1f} |",
        "",
    ]
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"language_scope_{args.split}.md"
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("\n".join(out))
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.exit(main())
