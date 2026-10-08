"""Small, deterministic helpers for normalising everyday request wording."""

from __future__ import annotations

import re


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
NUMBER_PATTERN = r"\d+|" + "|".join(NUMBER_WORDS)

MONTH_ALIASES = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "sept": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}

WORD_ALIASES = {
    "tmr": "tomorrow",
    "tmrw": "tomorrow",
    "yday": "yesterday",
    "yestday": "yesterday",
    "ytd": "yesterday",
    "til": "until",
    "till": "until",
    "mon": "monday",
    "tue": "tuesday",
    "tues": "tuesday",
    "wed": "wednesday",
    "thu": "thursday",
    "thur": "thursday",
    "thurs": "thursday",
    "fri": "friday",
    "sat": "saturday",
    "sun": "sunday",
}


def number_value(value: str) -> int:
    """Convert a supported digit or number word to an integer."""
    cleaned = value.strip().lower()
    return int(cleaned) if cleaned.isdigit() else NUMBER_WORDS[cleaned]


def normalize_request(text: str) -> str:
    """Reduce harmless wording variations before semantic parsing.

    This deliberately performs only unambiguous rewrites. It does not infer a
    camera, date, or query by itself; the typed planner remains responsible for
    those decisions.
    """
    normalized = text.lower().replace("’", "'")
    normalized = normalized.replace("@", " at ")
    normalized = re.sub(
        r"\bthe day before yesterday\b|\bday before yesterday\b",
        "2 days ago",
        normalized,
    )
    normalized = re.sub(r"\ba couple of days ago\b", "2 days ago", normalized)
    normalized = re.sub(
        rf"\b({NUMBER_PATTERN})\s+(days?|weeks?)\s+back\b",
        r"past \1 \2",
        normalized,
    )
    for short, full in MONTH_ALIASES.items():
        normalized = re.sub(rf"\b{short}(?:\.(?=\s|$)|\b)", full, normalized)
    for short, full in WORD_ALIASES.items():
        normalized = re.sub(rf"\b{short}(?:\.(?=\s|$)|\b)", full, normalized)
    for word, value in NUMBER_WORDS.items():
        normalized = re.sub(rf"\b{word}\b", str(value), normalized)
    normalized = re.sub(r"\bgimme\b|\bdrop me\b", "give me", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()
