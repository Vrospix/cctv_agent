"""Camera registry, aliases, multi-camera extraction, and suggestions."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


CAMERAS = {
    "PIE": "Pan Island Expressway",
    "AYE": "Ayer Rajah Expressway",
    "ECP": "East Coast Parkway",
    "CTE": "Central Expressway",
    "TPE": "Tampines Expressway",
    "KPE": "Kallang-Paya Lebar Expressway",
    "SLE": "Seletar Expressway",
    "BKE": "Bukit Timah Expressway",
    "KJE": "Kranji Expressway",
    "MCE": "Marina Coastal Expressway",
}

ALIASES = {
    "kranji highway": "KJE",
    "pan island highway": "PIE",
    "pi expressway": "PIE",
    "east coast expressway": "ECP",
    "central highway": "CTE",
    "tampines highway": "TPE",
    "seletar highway": "SLE",
    "bukit timah highway": "BKE",
    "marina expressway": "MCE",
    "kallang paya lebar expressway": "KPE",
}

GENERIC_WORDS = {
    "expressway",
    "highway",
    "parkway",
    "road",
    "camera",
    "cctv",
    "frame",
    "frames",
    "show",
    "from",
    "use",
    "instead",
    "today",
    "yesterday",
}


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _candidate_aliases() -> dict[str, str]:
    candidates: dict[str, str] = {}
    for acronym, name in CAMERAS.items():
        candidates[acronym.lower()] = acronym
        candidates[_normalise(name)] = acronym
    candidates.update(ALIASES)
    return candidates


def resolve_cameras(text: str) -> tuple[str, ...]:
    """Return every confidently identified canonical camera in mention order."""
    normalised = _normalise(text)
    matches: list[tuple[int, str]] = []
    for alias, acronym in _candidate_aliases().items():
        match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalised)
        if match:
            matches.append((match.start(), acronym))

    if not matches:
        input_words = set(normalised.split()) - GENERIC_WORDS
        for acronym, name in CAMERAS.items():
            significant = set(_normalise(name).split()) - GENERIC_WORDS
            if significant and significant.issubset(input_words):
                matches.append((0, acronym))

    if not matches:
        words = normalised.split()
        scored: list[tuple[float, str]] = []
        for alias, acronym in _candidate_aliases().items():
            alias_words = alias.split()
            for size in {max(1, len(alias_words) - 1), len(alias_words), len(alias_words) + 1}:
                for index in range(max(0, len(words) - size + 1)):
                    window = " ".join(words[index : index + size])
                    scored.append((SequenceMatcher(None, window, alias).ratio(), acronym))
        scored.sort(reverse=True)
        if scored:
            best_score, best = scored[0]
            competing = max((score for score, item in scored if item != best), default=0.0)
            if best_score >= 0.82 and best_score - competing >= 0.08:
                matches.append((0, best))

    ordered: list[str] = []
    for _, acronym in sorted(matches):
        canonical = CAMERAS[acronym]
        if canonical not in ordered:
            ordered.append(canonical)
    return tuple(ordered)


def suggest_cameras(text: str, *, limit: int = 3) -> tuple[str, ...]:
    """Return likely canonical cameras for a clarification prompt."""
    content_words = [word for word in _normalise(text).split() if word not in GENERIC_WORDS]
    if not content_words:
        return ()
    content = " ".join(content_words)
    scores: dict[str, float] = {}
    for acronym, name in CAMERAS.items():
        variants = [acronym.lower(), _normalise(name)]
        variants.extend(alias for alias, target in ALIASES.items() if target == acronym)
        best = 0.0
        for variant in variants:
            useful = " ".join(word for word in variant.split() if word not in GENERIC_WORDS)
            if useful:
                best = max(best, SequenceMatcher(None, content, useful).ratio())
                for word in content_words:
                    best = max(best, SequenceMatcher(None, word, useful).ratio())
        scores[name] = best
    ranked = sorted(scores, key=scores.get, reverse=True)
    return tuple(name for name in ranked[:limit] if scores[name] >= 0.65)
