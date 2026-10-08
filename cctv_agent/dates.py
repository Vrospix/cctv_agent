"""Deterministic date and time extraction for supported query language."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, time, timedelta

from .language import NUMBER_PATTERN, normalize_request, number_value
from .models import QueryConstraints
from .times import parse_time_range


WEEKDAYS = {name.lower(): index for index, name in enumerate(calendar.day_name)}
MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
MONTH_PATTERN = "|".join(MONTHS)
WEEKDAY_PATTERN = "|".join(WEEKDAYS)


def _day_start(value: date) -> datetime:
    return datetime.combine(value, time.min)


def _next_day(value: date) -> datetime:
    return _day_start(value + timedelta(days=1))


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = date(year, month, 1)
    following = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return _day_start(start), _day_start(following)


def _number(value: str) -> int:
    return number_value(value)


def _named_date(match: re.Match[str], *, today: date, day_first: bool = False) -> date:
    if day_first:
        day_text, month_name, year_text = match.groups()
    else:
        month_name, day_text, year_text = match.groups()
    return date(int(year_text or today.year), MONTHS[month_name], int(day_text))


def _find_single_date(text: str, *, today: date) -> date | None:
    iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if iso:
        return date.fromisoformat(iso.group(1))
    month_first = re.search(
        rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b",
        text,
    )
    if month_first:
        return _named_date(month_first, today=today)
    day_first = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?"
        rf"({MONTH_PATTERN})(?:\s+(20\d{{2}}))?\b",
        text,
    )
    if day_first:
        return _named_date(day_first, today=today, day_first=True)
    return None


def _parse_date_range(
    text: str, *, today: date
) -> tuple[datetime | None, datetime | None, int | None]:
    simple_days = list(re.finditer(r"\b(today|yesterday)\b", text))
    if simple_days:
        target = today if simple_days[-1].group(1) == "today" else today - timedelta(days=1)
        return _day_start(target), _next_day(target), None

    days_ago = re.search(rf"\b({NUMBER_PATTERN}|a|an)\s+days?\s+ago\b", text)
    if days_ago:
        amount = 1 if days_ago.group(1) in {"a", "an"} else _number(days_ago.group(1))
        target = today - timedelta(days=amount)
        return _day_start(target), _next_day(target), None

    rolling = re.search(
        rf"\b(?:last|past)\s+({NUMBER_PATTERN})\s+(days?|weeks?)\b",
        text,
    )
    if rolling:
        amount = _number(rolling.group(1))
        days = amount * (7 if rolling.group(2).startswith("week") else 1)
        return _day_start(today - timedelta(days=days - 1)), _next_day(today), None

    if re.search(r"\blast week\b", text):
        monday = today - timedelta(days=today.weekday())
        return _day_start(monday - timedelta(days=7)), _day_start(monday), None
    if re.search(r"\bthis week\b", text):
        monday = today - timedelta(days=today.weekday())
        return _day_start(monday), _next_day(today), None
    relative_month_range = re.search(
        r"(?:from\s+)?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+to\s+"
        r"(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?",
        text,
    )
    if relative_month_range and "last month" in text:
        year, month = today.year, today.month - 1
        if month == 0:
            year, month = year - 1, 12
        first_day, last_day = map(int, relative_month_range.groups())
        first, last = date(year, month, first_day), date(year, month, last_day)
        return _day_start(first), _next_day(last), None
    last_month_day = re.search(
        r"\b(?:on\s+)?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+of\s+last month\b",
        text,
    )
    if last_month_day:
        year, month = today.year, today.month - 1
        if month == 0:
            year, month = year - 1, 12
        target = date(year, month, int(last_month_day.group(1)))
        return _day_start(target), _next_day(target), None
    if "last month" in text:
        year, month = today.year, today.month - 1
        if month == 0:
            year, month = year - 1, 12
        start, end = _month_bounds(year, month)
        return start, end, None
    if "month to date" in text or "this month" in text:
        start, _ = _month_bounds(today.year, today.month)
        return start, _next_day(today), None

    if "last weekend" in text or "this weekend" in text:
        monday = today - timedelta(days=today.weekday())
        saturday = monday + timedelta(days=5)
        if "last weekend" in text:
            saturday -= timedelta(days=7)
        if saturday > today:
            raise ValueError("This weekend has not occurred in the available data yet.")
        sunday = saturday + timedelta(days=1)
        end = min(_next_day(sunday), _next_day(today))
        return _day_start(saturday), end, None

    first_week = re.search(
        rf"\bfirst week of\s+({MONTH_PATTERN})(?:\s+(20\d{{2}}))?\b", text
    )
    if first_week:
        month = MONTHS[first_week.group(1)]
        year = int(first_week.group(2) or today.year)
        start = date(year, month, 1)
        return _day_start(start), _day_start(start + timedelta(days=7)), None

    weekday_range = re.search(
        rf"\b(?:from\s+)?({WEEKDAY_PATTERN})\s+(?:to|through|until)\s+({WEEKDAY_PATTERN})\b",
        text,
    )
    if weekday_range:
        monday = today - timedelta(days=today.weekday())
        first = monday + timedelta(days=WEEKDAYS[weekday_range.group(1)])
        last = monday + timedelta(days=WEEKDAYS[weekday_range.group(2)])
        if last < first:
            last += timedelta(days=7)
        if first > today:
            raise ValueError("That weekday range begins after the available current date.")
        return _day_start(first), min(_next_day(last), _next_day(today)), None

    recurring = re.search(rf"\bevery\s+({WEEKDAY_PATTERN})\b", text)
    if recurring:
        return datetime(today.year, 1, 1), _next_day(today), WEEKDAYS[recurring.group(1)]

    relative_weekday = re.search(
        rf"\b(next|last|previous|this)\s+({WEEKDAY_PATTERN})\b", text
    )
    if relative_weekday:
        direction, weekday_name = relative_weekday.groups()
        target_weekday = WEEKDAYS[weekday_name]
        if direction == "next":
            target = today + timedelta(days=(target_weekday - today.weekday()) % 7 or 7)
            if target > today:
                raise ValueError("That weekday is later than the available current date.")
        elif direction in {"last", "previous"}:
            target = today - timedelta(days=(today.weekday() - target_weekday) % 7 or 7)
        else:
            monday = today - timedelta(days=today.weekday())
            target = monday + timedelta(days=target_weekday)
            if target > today:
                raise ValueError("That weekday is later than the available current date.")
        return _day_start(target), _next_day(target), None

    single_weekday = re.search(rf"\b(?:on\s+)?({WEEKDAY_PATTERN})\b", text)
    if single_weekday:
        target_weekday = WEEKDAYS[single_weekday.group(1)]
        target = today - timedelta(days=(today.weekday() - target_weekday) % 7)
        return _day_start(target), _next_day(target), None

    iso_range = re.search(
        r"\b(?:from\s+)?(20\d{2}-\d{2}-\d{2})\s+(?:to|through|until)\s+"
        r"(20\d{2}-\d{2}-\d{2})\b",
        text,
    )
    if iso_range:
        first, last = map(date.fromisoformat, iso_range.groups())
        if last < first:
            raise ValueError("The end date must not be before the start date.")
        return _day_start(first), _next_day(last), None

    same_month_range = re.search(
        rf"\b(?:from\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:to|through|until)\s+"
        rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH_PATTERN})(?:\s+(20\d{{2}}))?\b",
        text,
    )
    if same_month_range:
        first_day, last_day, month_name, year_text = same_month_range.groups()
        year, month = int(year_text or today.year), MONTHS[month_name]
        first, last = date(year, month, int(first_day)), date(year, month, int(last_day))
        if last < first:
            raise ValueError("The end date must not be before the start date.")
        return _day_start(first), _next_day(last), None

    since_match = re.search(r"\bsince\s+(.+)$", text)
    if since_match:
        target = _find_single_date(since_match.group(1), today=today)
        if target:
            return _day_start(target), _next_day(today), None

    date_before = re.search(r"\bbefore\s+(.+)$", text)
    if date_before and not re.search(r"\b(?:am|pm|\d{1,2}:\d{2})\b", date_before.group(1)):
        target = _find_single_date(date_before.group(1), today=today)
        if target:
            return datetime(today.year, 1, 1), _day_start(target), None

    date_after = re.search(r"\bafter\s+(.+)$", text)
    if date_after and not re.search(r"\b(?:am|pm|\d{1,2}:\d{2})\b", date_after.group(1)):
        target = _find_single_date(date_after.group(1), today=today)
        if target:
            return _next_day(target), _next_day(today), None

    target = _find_single_date(text, today=today)
    if target:
        return _day_start(target), _next_day(target), None

    month_match = re.search(rf"(?:whole of\s+)?({MONTH_PATTERN})(?:\s+(20\d{{2}}))?", text)
    if month_match:
        year = int(month_match.group(2) or today.year)
        start, end = _month_bounds(year, MONTHS[month_match.group(1)])
        return start, end, None
    return None, None, None


def parse_temporal(text: str, *, today: date) -> QueryConstraints:
    lowered = normalize_request(text)
    start, end, weekday = _parse_date_range(lowered, today=today)
    time_start, time_end = parse_time_range(lowered)
    return QueryConstraints(
        start=start,
        end=end,
        weekday=weekday,
        time_start=time_start,
        time_end=time_end,
    )
