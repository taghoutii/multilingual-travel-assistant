"""Text folding shared by date parsing, lookups and validation.

`fold` makes matching robust to case, Latin accents, Arabic diacritics/hamza spelling, Arabic-Indic
digits and punctuation. It is lossy on purpose: use it for matching, never for display.
"""

from __future__ import annotations

import re
import unicodedata

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ARABIC_LETTERS = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
        "ـ": None,  # tatweel
    }
)
# Keep letters, digits, whitespace and the characters used in numeric dates and currency symbols.
_PUNCT = re.compile(r"[^\w\s/€$£¥]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def strip_marks(text: str) -> str:
    """Remove combining marks: Latin accents and Arabic harakat/shadda."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def fold(text: str) -> str:
    """Lowercase, strip accents and Arabic diacritics, unify hamza forms, ASCII digits, no punct."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = strip_marks(text).translate(_ARABIC_DIGITS).translate(_ARABIC_LETTERS)
    text = text.replace("_", " ")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def squash_spaces(text: str) -> str:
    """Lowercase and collapse whitespace only (used to compare raw spans)."""
    return _SPACES.sub(" ", text.casefold()).strip()
