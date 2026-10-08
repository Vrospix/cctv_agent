"""Deterministic extraction of clock times and parts of day."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta


def _parse_clock(hour: str, minute: str | None, meridiem: str | None) -> time:
    numeric_hour = int(hour)
    numeric_minute = int(minute or 0)
    if not 0 <= numeric_minute <= 59:
        raise ValueError("Minutes must be between 00 and 59.")
    if meridiem:
        if not 1 <= numeric_hour <= 12:
            raise ValueError("12-hour times must use an hour from 1 to 12.")
        numeric_hour %= 12
        if meridiem.lower() == "pm":
            numeric_hour += 12
    elif not 0 <= numeric_hour <= 23:
        raise ValueError("24-hour times must use an hour from 0 to 23.")
    return time(numeric_hour, numeric_minute)


def _clock_text(value: str, *, default_meridiem: str | None = None) -> time:
    cleaned = re.sub(r"\s+", "", value.lower())
    if cleaned == "noon":
        return time(12)
    if cleaned == "midnight":
        return time.min
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", cleaned)
    if not match:
        raise ValueError(f"Unsupported time: {value}")
    hour, minute, meridiem = match.groups()
    return _parse_clock(hour, minute, meridiem or default_meridiem)


def _explicit_meridiem(value: str) -> str | None:
    match = re.search(r"\b(am|pm)\b", value)
    return match.group(1) if match else None


def _part_of_day(text: str) -> tuple[time, time] | None:
    if "morning" in text:
        return time(6), time(12)
    if "afternoon" in text:
        return time(12), time(17)
    if "evening" in text:
        return time(17), time(21)
    if re.search(r"\b(?:lunch|lunchtime)\b", text):
        return time(12), time(14)
    return None


def parse_time_range(text: str) -> tuple[time | None, time | None]:
    """Extract one validated daily time window from normalized text."""
    clock = r"(?:noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)"

    time_range = re.search(
        rf"(?:between|from)\s+({clock})\s+(?:and|to|until)\s+({clock})"
        rf"(?:\s+in\s+the\s+(morning|afternoon|evening))?",
        text,
    ) or re.search(rf"\b({clock})\s*(?:to|until|-)\s*({clock})\b", text)
    if time_range:
        first_text, last_text = time_range.group(1), time_range.group(2)
        day_part = time_range.group(3) if len(time_range.groups()) == 3 else None
        contextual_meridiem = {
            "morning": "am",
            "afternoon": "pm",
            "evening": "pm",
        }.get(day_part)
        first_meridiem = _explicit_meridiem(first_text)
        last_meridiem = _explicit_meridiem(last_text)
        start = _clock_text(
            first_text,
            default_meridiem=first_meridiem or last_meridiem or contextual_meridiem,
        )
        end = _clock_text(
            last_text,
            default_meridiem=last_meridiem or first_meridiem or contextual_meridiem,
        )
        if end <= start:
            raise ValueError("The end time must be later than the start time.")
        return start, end

    day_part_bounds = _part_of_day(text)
    before = re.search(rf"\bbefore\s+({clock})\b", text)
    after = re.search(rf"\bafter\s+({clock})\b", text)
    if before and after:
        raise ValueError(
            "Overnight or combined before/after ranges are ambiguous; "
            "use one explicit time range."
        )
    if before:
        end = _clock_text(before.group(1))
        if day_part_bounds:
            if end <= day_part_bounds[0]:
                raise ValueError("That time falls outside the requested part of day.")
            return day_part_bounds[0], min(end, day_part_bounds[1])
        return None, end
    if after:
        start = _clock_text(after.group(1))
        if day_part_bounds:
            if start >= day_part_bounds[1]:
                raise ValueError("That time falls outside the requested part of day.")
            return max(start, day_part_bounds[0]), day_part_bounds[1]
        return start, None

    exact = re.search(rf"\bat\s+({clock})\b", text)
    if exact:
        start = _clock_text(exact.group(1))
        anchor = datetime.combine(date(2000, 1, 1), start)
        return start, (anchor + timedelta(minutes=5)).time()
    if day_part_bounds:
        return day_part_bounds
    return None, None
