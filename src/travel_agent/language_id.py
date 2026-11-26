"""Language identification (pipeline stage 1): fastText lid.176 restricted to fr / en / ar.

Decision rule, applied to fastText's top-10 predictions (`ar`, `arz`, `ary`, ... count as `ar`):

1. The top prediction is a supported language -> that language.
2. The top prediction is another language with probability >= `unsupported_min_prob` -> the
   message is in an unsupported language: reply in English with a one-line notice.
3. Otherwise the most probable supported language. Short or noisy messages often put a wrong
   language first with a low probability ("plats typiques djerba" -> Swedish 0.47, French 0.06),
   so a supported language that is present at all is preferred.
4. No prediction at all (fastText returns nothing for some short texts): the previous turn's
   language if known, else `ar` if the text contains Arabic letters, else `en`.

With `previous` set (multi-turn), a detection below `keep_previous_below` keeps the previous
language, so a short follow-up ("ok", "visa tokyo") does not switch the reply language.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from travel_agent.config import PROJECT_ROOT, get_settings

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODELS_DIR = PROJECT_ROOT / "models"
ARABIC_LABELS = frozenset({"ar", "arz", "ary", "acm", "apc", "acq", "aeb", "ajp", "apd", "arq"})
TOP_K = 10

Lang = Literal["fr", "en", "ar"]
SUPPORTED: tuple[Lang, ...] = ("fr", "en", "ar")
# text -> [(fastText label without "__label__", probability)], most probable first
Predictor = Callable[[str], list[tuple[str, float]]]

_ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓ]")


@dataclass(frozen=True)
class LanguageResult:
    lang: Lang  # language to reply in
    supported: bool  # False -> reply in English with a notice
    detected: str  # fastText's top label ("" when it gave no prediction)
    confidence: float  # probability of `lang` (or of `detected` when unsupported)
    reason: str  # which rule decided: top / unsupported / best_supported / previous / fallback


def model_path() -> Path:
    return MODELS_DIR / get_settings().language_id.model


def fasttext_predictor(path: Path | None = None, k: int = TOP_K) -> Predictor:
    import fasttext

    model = fasttext.load_model(str(path or model_path()))

    def predict(text: str) -> list[tuple[str, float]]:
        # model.predict() breaks under numpy 2 (np.array(copy=False)); the binding works.
        raw = model.f.predict(text, k, 0.0, "strict")
        return [(label.removeprefix("__label__"), float(prob)) for prob, label in raw]

    return predict


def _base_label(label: str) -> str:
    return "ar" if label in ARABIC_LABELS else label


class LanguageIdentifier:
    def __init__(
        self,
        predictor: Predictor,
        unsupported_min_prob: float = 0.85,
        keep_previous_below: float = 0.5,
    ) -> None:
        self._predict = predictor
        self.unsupported_min_prob = unsupported_min_prob
        self.keep_previous_below = keep_previous_below

    def identify(self, text: str, previous: Lang | None = None) -> LanguageResult:
        clean = " ".join(text.split())  # fastText reads one line
        merged: dict[str, float] = {}
        for label, prob in self._predict(clean) if clean else []:
            base = _base_label(label)
            merged[base] = merged.get(base, 0.0) + prob
        ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)

        if not ranked:
            if previous is not None:
                return LanguageResult(previous, True, "", 0.0, "previous")
            lang: Lang = "ar" if _ARABIC_LETTER.search(clean) else "en"
            return LanguageResult(lang, True, "", 0.0, "fallback")

        top_label, top_prob = ranked[0]
        supported = [(lbl, p) for lbl, p in ranked if lbl in SUPPORTED]
        if top_label not in SUPPORTED and top_prob >= self.unsupported_min_prob:
            return LanguageResult("en", False, top_label, top_prob, "unsupported")
        if not supported:
            if previous is not None:
                return LanguageResult(previous, True, top_label, 0.0, "previous")
            fallback: Lang = "ar" if _ARABIC_LETTER.search(clean) else "en"
            return LanguageResult(fallback, True, top_label, 0.0, "fallback")

        best, prob = supported[0]
        if previous is not None and best != previous and prob < self.keep_previous_below:
            return LanguageResult(previous, True, top_label, merged.get(previous, 0.0), "previous")
        reason = "top" if best == top_label else "best_supported"
        return LanguageResult(_as_lang(best), True, top_label, prob, reason)


def _as_lang(label: str) -> Lang:
    for lang in SUPPORTED:
        if lang == label:
            return lang
    raise ValueError(label)


@lru_cache(maxsize=1)
def get_language_identifier() -> LanguageIdentifier:
    cfg = get_settings().language_id
    return LanguageIdentifier(
        fasttext_predictor(), cfg.unsupported_min_prob, cfg.keep_previous_below
    )
