"""Word tables that rewrite fr/en/ar date expressions into one canonical English form.

All keys are folded with `text_norm.fold` at import time, so they can be written naturally here.
Canonical vocabulary: digits, month and weekday names, today, tomorrow, in, next, this, weekend,
mid, early, end, and the units day/week/month/night.
"""

from __future__ import annotations

from travel_agent.text_norm import fold

MONTHS_EN = (
    "january february march april may june july august september october november december".split()
)
WEEKDAYS_EN = "monday tuesday wednesday thursday friday saturday sunday".split()
UNITS = {
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "night": "night",
    "nights": "night",
}

_MONTH_WORDS: dict[str, tuple[str, ...]] = {
    "january": ("janvier", "janv", "jan", "january", "يناير", "كانون الثاني", "جانفي"),
    "february": ("fevrier", "fevr", "fev", "feb", "february", "فبراير", "شباط", "فيفري"),
    "march": ("mars", "mar", "march", "مارس", "اذار"),
    "april": ("avril", "avr", "apr", "april", "ابريل", "نيسان", "افريل"),
    "may": ("mai", "may", "مايو", "ايار", "ماي"),
    "june": ("juin", "jun", "june", "يونيو", "يونيه", "حزيران", "جوان"),
    "july": ("juillet", "juil", "jul", "july", "يوليو", "يوليه", "تموز", "جويليه", "يوليوز"),
    "august": ("aout", "aug", "august", "اغسطس", "اب", "اوت", "غشت"),
    "september": ("septembre", "sep", "september", "سبتمبر", "ايلول", "شتنبر"),
    "october": ("octobre", "oct", "october", "اكتوبر", "تشرين الاول"),
    "november": ("novembre", "nov", "november", "نوفمبر", "تشرين الثاني", "نونبر"),
    "december": ("decembre", "dec", "december", "ديسمبر", "كانون الاول", "دجنبر"),
}

_WEEKDAY_WORDS: dict[str, tuple[str, ...]] = {
    "monday": ("lundi", "monday", "mon", "الاثنين", "الإثنين", "اثنين"),
    "tuesday": ("mardi", "tuesday", "tue", "tues", "الثلاثاء", "ثلاثاء"),
    "wednesday": ("mercredi", "wednesday", "wed", "الاربعاء", "اربعاء"),
    "thursday": ("jeudi", "thursday", "thu", "thurs", "الخميس", "خميس"),
    "friday": ("vendredi", "friday", "fri", "الجمعة", "جمعة"),
    "saturday": ("samedi", "saturday", "sat", "السبت", "سبت"),
    "sunday": ("dimanche", "sunday", "sun", "الاحد", "احد"),
}

_PHRASES: dict[str, str] = {
    # today / tomorrow / day after tomorrow (before the generic "in" rewrites)
    "aujourd hui": "today",
    "aujourdhui": "today",
    "auj": "today",
    "ajd": "today",
    "today": "today",
    "2day": "today",
    "tdy": "today",
    "اليوم": "today",
    "apres demain": "in 2 day",
    "apres dem": "in 2 day",
    "day after tomorrow": "in 2 day",
    "the day after tomorrow": "in 2 day",
    "بعد غد": "in 2 day",
    "بعد الغد": "in 2 day",
    "demain": "tomorrow",
    "dem": "tomorrow",
    "dmain": "tomorrow",
    "2main": "tomorrow",
    "tomorrow": "tomorrow",
    "tmrw": "tomorrow",
    "tmr": "tomorrow",
    "tmrow": "tomorrow",
    "tomorow": "tomorrow",
    "tommorow": "tomorrow",
    "tommorrow": "tomorrow",
    "2moro": "tomorrow",
    "غدا": "tomorrow",
    "الغد": "tomorrow",
    "غد": "tomorrow",
    # weekend (before "end" and "week")
    "week end": "weekend",
    "weekend": "weekend",
    "wknd": "weekend",
    "ce we": "this weekend",
    "عطلة نهاية الاسبوع": "weekend",
    "نهاية الاسبوع": "weekend",
    "الويكند": "weekend",
    "ويكند": "weekend",
    "ويك اند": "weekend",
    # relative markers
    "dans": "in",
    "d ici": "in",
    "in": "in",
    "within": "in",
    "بعد": "in",
    "خلال": "in",
    "في غضون": "in",
    "prochain": "next",
    "prochaine": "next",
    "next": "next",
    "coming": "next",
    "القادم": "next",
    "القادمة": "next",
    "المقبل": "next",
    "المقبلة": "next",
    "الجاي": "next",
    "ce": "this",
    "cette": "this",
    "this": "this",
    "هذا": "this",
    "هذه": "this",
    # parts of a month
    "mi": "mid",
    "mid": "mid",
    "middle": "mid",
    "milieu": "mid",
    "منتصف": "mid",
    "اواسط": "mid",
    "debut": "early",
    "early": "early",
    "beginning": "early",
    "start": "early",
    "بداية": "early",
    "اوائل": "early",
    "مطلع": "early",
    "fin": "end",
    "end": "end",
    "late": "end",
    "نهاية": "end",
    "اواخر": "end",
    "اخر": "end",
    # units
    "jour": "day",
    "jours": "day",
    "journee": "day",
    "journees": "day",
    "day": "day",
    "days": "day",
    "يوم": "day",
    "ايام": "day",
    "يومين": "2 day",
    "semaine": "week",
    "semaines": "week",
    "week": "week",
    "weeks": "week",
    "wk": "week",
    "wks": "week",
    "fortnight": "2 week",
    "اسبوع": "week",
    "الاسبوع": "week",
    "اسابيع": "week",
    "اسبوعين": "2 week",
    "mois": "month",
    "month": "month",
    "months": "month",
    "شهر": "month",
    "الشهر": "month",
    "اشهر": "month",
    "شهور": "month",
    "شهرين": "2 month",
    "nuit": "night",
    "nuits": "night",
    "nuitee": "night",
    "nuitees": "night",
    "night": "night",
    "nights": "night",
    "nite": "night",
    "nites": "night",
    "ليلة": "night",
    "ليال": "night",
    "ليالي": "night",
    "ليلتين": "2 night",
    "premier": "1",
    "1er": "1",
    "1ere": "1",
    "first": "1",
}

_UNITS_FR = "un deux trois quatre cinq six sept huit neuf".split()
_UNITS_EN = "one two three four five six seven eight nine".split()
_TEENS_EN = (
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
)
_ORD_EN = (
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
    "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth"
).split()
_UNITS_AR = (
    ("واحد", "واحدة"),
    ("اثنين", "اثنان", "اثنتين"),
    ("ثلاث", "ثلاثة"),
    ("اربع", "اربعة"),
    ("خمس", "خمسة"),
    ("ست", "ستة"),
    ("سبع", "سبعة"),
    ("ثمان", "ثماني", "ثمانية"),
    ("تسع", "تسعة"),
)
_ORD_AR = (
    "الاول",
    "الثاني",
    "الثالث",
    "الرابع",
    "الخامس",
    "السادس",
    "السابع",
    "الثامن",
    "التاسع",
    "العاشر",
)


def _fr_teens_fixed() -> list[str]:
    # "dix sept", "dix huit", "dix neuf" are two words each
    return [
        "dix",
        "onze",
        "douze",
        "treize",
        "quatorze",
        "quinze",
        "seize",
        "dix sept",
        "dix huit",
        "dix neuf",
    ]


def _number_words() -> dict[str, str]:
    """Cardinal (fr/en/ar) and ordinal (en/ar) words for 1..31."""
    out: dict[str, str] = {"une": "1"}  # "a"/"an" are handled in dates.py (only before a unit)
    for i, word in enumerate(_UNITS_FR, 1):
        out[word] = str(i)
    for i, word in enumerate(_fr_teens_fixed(), 10):
        out[word] = str(i)
    out.update({"vingt": "20", "vingt et un": "21", "trente": "30", "trente et un": "31"})
    for i, word in enumerate(_UNITS_FR[1:], 2):
        out[f"vingt {word}"] = str(20 + i)
    for i, word in enumerate(_UNITS_EN, 1):
        out[word] = str(i)
    for i, word in enumerate(_TEENS_EN, 10):
        out[word] = str(i)
    out.update(
        {
            "twenty": "20",
            "thirty": "30",
            "thirty one": "31",
            "thirty first": "31",
            "thirtieth": "30",
            "twenty first": "21",
        }
    )
    for i, word in enumerate(_UNITS_EN, 1):
        out[f"twenty {word}"] = str(20 + i)
    for i, word in enumerate(_ORD_EN, 1):
        out[word] = str(i)
    for i, word in enumerate(
        ("second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"), 2
    ):
        out[f"twenty {word}"] = str(20 + i)
    for i, forms in enumerate(_UNITS_AR, 1):
        for form in forms:
            out[form] = str(i)
    for form in ("عشر", "عشرة"):
        out[form] = "10"
    for i, forms in enumerate(_UNITS_AR[2:], 3):  # 13..19: "ثلاثة عشر"
        for form in forms:
            out[f"{form} عشر"] = str(10 + i)
            out[f"{form} عشرة"] = str(10 + i)
    out.update(
        {
            "احد عشر": "11",
            "احدى عشرة": "11",
            "اثنا عشر": "12",
            "اثني عشر": "12",
            "اثنتا عشرة": "12",
            "عشرين": "20",
            "عشرون": "20",
            "ثلاثين": "30",
            "ثلاثون": "30",
        }
    )
    for i, forms in enumerate(_UNITS_AR, 1):
        for form in forms:
            for tens, base in (("عشرين", 20), ("عشرون", 20)):
                out[f"{form} و{tens}"] = str(base + i)
    for form in ("واحد", "واحدة"):
        out[f"{form} وثلاثين"] = "31"
    # Arabic ordinals
    for i, word in enumerate(_ORD_AR, 1):
        out[word] = str(i)
    out.update(
        {
            "اول": "1",
            "الحادي عشر": "11",
            "الثاني عشر": "12",
            "العشرين": "20",
            "العشرون": "20",
            "الثلاثين": "30",
            "الثلاثون": "30",
            "الحادي والعشرين": "21",
            "الحادي والثلاثين": "31",
        }
    )
    for i, word in enumerate(_ORD_AR[2:9], 3):  # الثالث عشر .. التاسع عشر
        out[f"{word} عشر"] = str(10 + i)
    for i, word in enumerate(_ORD_AR[1:9], 2):  # الثاني والعشرين .. التاسع والعشرين
        out[f"{word} والعشرين"] = str(20 + i)
    return out


def build_phrase_table() -> dict[str, str]:
    """Folded phrase -> canonical replacement. Longer phrases must be matched first."""
    table: dict[str, str] = {}
    for canonical, words in _MONTH_WORDS.items():
        for word in words:
            table[fold(word)] = canonical
    for canonical, words in _WEEKDAY_WORDS.items():
        for word in words:
            table[fold(word)] = canonical
    for word, replacement in _number_words().items():
        table[fold(word)] = replacement
    for word, replacement in _PHRASES.items():
        table[fold(word)] = replacement
    # "sept" is 7 in French but also an English abbreviation of September: resolved in dates.py.
    table["sept"] = "sept"
    return table


# Words carrying no date information once the phrase table has been applied.
STOPWORDS = frozenset(
    fold(w)
    for w in (
        "le la les l du de d des au aux a en pour vers jusqu jusqu au a partir partir from "
        "the of on at for to until till til during by around about approximately about "
        "يوم في من الى إلى حتى لمدة مدة لـ ل حوالي تقريبا to"
    ).split()
)
