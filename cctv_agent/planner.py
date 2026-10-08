"""Convert natural-language messages into deterministic :class:`QueryPlan` objects."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

from .cameras import resolve_cameras, suggest_cameras
from .dates import parse_temporal
from .language import NUMBER_PATTERN, normalize_request, number_value
from .models import QueryConstraints, QueryPlan


COUNT_PATTERN = re.compile(
    r"\bhow many\b|\bcount\b|"
    r"\bnumber of\s+(?:matching\s+)?frames?\b|\btotal(?:\s+number)?\s+of\s+frames?\b",
    re.I,
)
CLEAR_DATE_PATTERN = re.compile(
    r"\b(?:remove|clear|without)\s+(?:the\s+)?date(?:\s+(?:filter|restriction))?\b|"
    r"\bany date\b",
    re.I,
)
CLEAR_TIME_PATTERN = re.compile(
    r"\b(?:remove|clear|without)\s+(?:the\s+)?(?:time|hours?)(?:\s+(?:filter|restriction))?\b|"
    r"\ball day\b",
    re.I,
)
CAMERA_WORDS = re.compile(r"\b(?:camera|expressway|highway|parkway)\b", re.I)
UNSUPPORTED_ANALYSIS_PATTERN = re.compile(
    r"\b(?:traffic|congestion|traffic jam|accident|weather)\b|"
    r"\b(?:is|was|looks?)\s+(?:the\s+)?(?:road|traffic)\s+(?:bad|busy|clear|heavy)\b",
    re.I,
)


def _control_plan(text: str) -> QueryPlan | None:
    """Recognize commands that do not change search filters."""
    if re.fullmatch(r"/?help", text):
        return QueryPlan(intent="help")
    if re.fullmatch(
        r"(?:start over|reset(?: the)? search|new search|clear (?:everything|all filters))\.?",
        text,
    ):
        return QueryPlan(intent="reset")
    if re.search(r"\b(?:undo(?: that| the change)?|previous (?:search|query))\b", text):
        return QueryPlan(intent="undo")
    if re.fullmatch(r"go back[.!]?", text):
        return QueryPlan(intent="undo")
    if re.fullmatch(
        r"(?:show\s+)?(?:current|active) filters?|what (?:filters|am i searching).*",
        text,
    ):
        return QueryPlan(intent="show_filters")
    if re.fullmatch(
        r"(?:show\s+)?(?:the\s+)?previous(?:\s+(?:page|results?|frames?))?\.?",
        text,
    ):
        return QueryPlan(intent="previous_page")

    next_page = re.fullmatch(
        r"(?:show\s+)?(?:the\s+)?next(?:\s+(\d+))?(?:\s+(?:page|results?|frames?))?\.?|"
        r"(?:show\s+)?more(?:\s+(?:results?|frames?))?\.?|"
        r"(?:give|show) me (?:the )?(?:ones|results|frames) after (?:those|these)[.!]?",
        text,
    )
    if next_page and "next day" not in text:
        size = next((group for group in next_page.groups() if group), None)
        return QueryPlan(intent="next_page", limit=int(size) if size else None)

    limit_match = re.fullmatch(
        r"(?:/limit\s+(\d+)|set\s+(?:the\s+)?limit(?:\s+to)?\s+(\d+))\.?",
        text,
    )
    if limit_match:
        size = next(group for group in limit_match.groups() if group)
        return QueryPlan(intent="set_limit", limit=int(size))
    if UNSUPPORTED_ANALYSIS_PATTERN.search(text):
        return QueryPlan(intent="unsupported_analysis")
    return None


def _camera_correction(text: str) -> tuple[str, ...]:
    """Prefer the final camera after an explicit correction phrase."""
    matches = list(
        re.finditer(r"\b(?:use|try|switch to|sorry)\s+(.+)$", text, re.I)
    )
    if not matches:
        return ()
    return resolve_cameras(matches[-1].group(1))


def _camera_change(text: str) -> tuple[str, tuple[str, ...], bool]:
    """Return camera action, resolved cameras, and mixed-negation ambiguity."""
    cameras = resolve_cameras(text)
    corrected = _camera_correction(text)
    if corrected:
        return "set", corrected, False

    mixed_negation = bool(
        len(cameras) > 1 and re.search(r"\bbut\s+(?:not|without)\b", text)
    )
    negative_camera = re.search(
        r"\b(?:not|no)\s+(?:the\s+)?(?:pie|aye|ecp|cte|tpe|kpe|sle|bke|kje|mce)\b|"
        r"\b(?:not|no)\s+(?:[a-z-]+\s+){0,3}(?:expressway|highway|parkway)\b",
        text,
    )
    if cameras and re.search(r"\b(?:anything|everything|all)\s+but\b", text):
        return "all_except", cameras, mixed_negation
    if cameras and (
        re.search(r"\b(?:except|exclude|excluding|without)\b", text)
        or negative_camera
    ):
        action = (
            "all_except"
            if re.search(r"\b(?:all|every|any) cameras?\b", text)
            else "exclude"
        )
        return action, cameras, mixed_negation
    if re.search(
        r"\b(?:all|every|any) cameras?\b|\b(?:remove|clear) (?:the )?camera",
        text,
    ):
        return "all", (), mixed_negation
    if cameras and re.search(r"\b(?:add|also|along with|as well as)\b", text):
        return "add", cameras, mixed_negation
    if cameras:
        return "set", cameras, mixed_negation
    return "keep", (), mixed_negation


def _needs_clarification(text: str) -> str | None:
    if re.search(
        r"\b(today|yesterday)\s+(?:or|/)\s+(today|yesterday)\b", text
    ):
        return "Did you mean today or yesterday?"
    if re.search(
        r"\b(?:around|roughly|approximately)\s+"
        r"(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight|lunch(?:time)?)\b|"
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?ish\b",
        text,
    ):
        return "What exact time or time range should I use?"
    if re.search(r"\b(?:recently|sometime|whenever)\b|(?<!most )\brecent\b", text):
        return "What date or date range should I use?"
    return None


def _clarification_plan(
    message: str,
    *,
    camera_action: str,
    cameras: tuple[str, ...],
    date_action: str,
    temporal: QueryConstraints,
    time_action: str,
) -> QueryPlan:
    return QueryPlan(
        intent="clarify",
        clarification=message,
        camera_action=camera_action,
        cameras=cameras,
        date_action=date_action,
        start=temporal.start,
        end=temporal.end,
        weekday=temporal.weekday,
        time_action=time_action,
        time_start=temporal.time_start,
        time_end=temporal.time_end,
    )


def _relative_day(
    text: str, *, today: date, context: QueryConstraints | None
) -> QueryConstraints | None:
    forward = re.search(r"\b(?:tomorrow|next day)\b", text, re.I)
    backward = re.search(r"\b(?:previous day|day before|go back one day)\b", text, re.I)
    if not forward and not backward:
        return None
    if context is None or context.start is None or context.end is None:
        raise ValueError("Ask for a specific day before moving to another day.")
    current_day = context.start.date()
    if context.end != datetime.combine(current_day + timedelta(days=1), time.min):
        raise ValueError("Day navigation needs a previous single-day query.")
    target = current_day + timedelta(days=1 if forward else -1)
    if target > today:
        raise ValueError("That day would be later than the available current date.")
    if target < date(2026, 1, 1):
        raise ValueError("The simulated dataset begins on 1 January 2026.")
    return QueryConstraints(
        start=datetime.combine(target, time.min),
        end=datetime.combine(target + timedelta(days=1), time.min),
    )


def _sort_order(text: str) -> str | None:
    if re.search(r"\b(?:latest|newest|most recent|newest first)\b", text):
        return "newest"
    if re.search(r"\b(?:earliest|oldest|oldest first)\b", text):
        return "oldest"
    if re.search(r"\b(?:random|randomly)\b", text):
        return "random"
    return None


def _sampling(text: str) -> tuple[str, int | None]:
    if re.search(r"\b(?:remove|clear) (?:the )?sampl|\bshow every frame\b", text):
        return "clear", None
    if re.search(r"\b(?:one|1)\b(?:\s+[a-z0-9-]+){0,4}\s+frames?\s+per\s+hour\b", text):
        return "set", 60
    sample_match = re.search(r"\b(?:1 frame )?every\s+(\d+)\s+minutes?\b", text)
    if sample_match:
        return "set", int(sample_match.group(1))
    return "keep", None


def _result_limit(text: str) -> int | None:
    """Extract a page size from common result-selection word orders."""
    patterns = (
        rf"\b({NUMBER_PATTERN})\s+of\s+(?:the\s+)?"
        rf"(?:latest|newest|most recent|earliest|oldest|random)\b",
        rf"\b(?:show|give me|return|fetch|find|list|i (?:only )?need)\s+"
        rf"(?:me\s+)?(?:the\s+)?"
        rf"(?:latest|newest|most recent|earliest|oldest|first|random)?\s*"
        rf"({NUMBER_PATTERN})\s+(?:random\s+)?"
        rf"(?:[a-z0-9-]+\s+){{0,3}}(?:frames?|results?)\b",
        rf"\b(?:latest|newest|most recent|earliest|oldest|random)\s+"
        rf"({NUMBER_PATTERN})(?:\s+(?:frames?|results?))?\s*[?.!]*$",
        rf"\bonly\s+({NUMBER_PATTERN})"
        rf"(?:\s+(?:frames?|results?|please|pls))*[.!]?\s*$",
        rf"\b(?:latest|newest|most recent|earliest|oldest|random)\s+"
        rf"({NUMBER_PATTERN})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return number_value(match.group(1))
    if re.search(r"\b(?:earliest|oldest) frame\b", text):
        return 1
    return None


def build_query_plan(
    text: str, *, today: date, context: QueryConstraints | None
) -> QueryPlan:
    """Interpret one user message without executing or constructing SQL."""
    lowered = normalize_request(text)

    control = _control_plan(lowered)
    if control:
        return control

    clarification = _needs_clarification(lowered)
    relative = _relative_day(lowered, today=today, context=context)
    temporal = relative or parse_temporal(lowered, today=today)
    camera_action, cameras, mixed_negation = _camera_change(lowered)

    clear_everything_else = bool(
        re.search(r"\bkeep (?:the )?camera(?:s)?(?: and|,) clear everything else\b", lowered)
    )
    clear_date = bool(CLEAR_DATE_PATTERN.search(lowered) or clear_everything_else)
    clear_time = bool(CLEAR_TIME_PATTERN.search(lowered) or clear_everything_else)

    date_changed = any((temporal.start, temporal.end, temporal.weekday is not None))
    time_changed = temporal.time_start is not None or temporal.time_end is not None
    date_action = "clear" if clear_date else ("set" if date_changed else "keep")
    time_action = "clear" if clear_time else ("set" if time_changed else "keep")

    if mixed_negation:
        return _clarification_plan(
            "Please state the cameras to include and exclude separately.",
            camera_action="keep",
            cameras=(),
            date_action=date_action,
            temporal=temporal,
            time_action=time_action,
        )
    if clarification:
        if "today or yesterday" in clarification.lower():
            temporal = QueryConstraints(
                time_start=temporal.time_start,
                time_end=temporal.time_end,
            )
            date_action = "keep"
        elif "exact time" in clarification.lower():
            temporal = QueryConstraints(
                start=temporal.start,
                end=temporal.end,
                weekday=temporal.weekday,
            )
            time_action = "keep"
        return _clarification_plan(
            clarification,
            camera_action=camera_action,
            cameras=cameras,
            date_action=date_action,
            temporal=temporal,
            time_action=time_action,
        )

    sort_order = _sort_order(lowered)
    sample_action, sample_interval = _sampling(lowered)
    result_limit = _result_limit(lowered)

    intent = "count" if COUNT_PATTERN.search(lowered) else "search"
    meaningful = any(
        (
            camera_action != "keep",
            date_action != "keep",
            time_action != "keep",
            sort_order is not None,
            sample_action != "keep",
            result_limit is not None,
            intent == "count",
            re.search(r"\b(?:repeat|same search|show (?:them|results) again)\b", lowered),
        )
    )

    if not cameras and camera_action == "keep" and CAMERA_WORDS.search(lowered):
        suggestions = suggest_cameras(lowered)
        if suggestions:
            return QueryPlan(
                intent="clarify",
                clarification="I couldn't identify that camera confidently. Which did you mean: "
                + ", ".join(suggestions)
                + "?",
                date_action=date_action,
                start=temporal.start,
                end=temporal.end,
                weekday=temporal.weekday,
                time_action=time_action,
                time_start=temporal.time_start,
                time_end=temporal.time_end,
            )

    if not meaningful:
        return QueryPlan(intent="unknown")
    return QueryPlan(
        intent=intent,
        camera_action=camera_action,
        cameras=cameras,
        date_action=date_action,
        start=temporal.start,
        end=temporal.end,
        weekday=temporal.weekday,
        time_action=time_action,
        time_start=temporal.time_start,
        time_end=temporal.time_end,
        sort_order=sort_order,
        sample_action=sample_action,
        sample_interval_minutes=sample_interval,
        limit=result_limit,
    )
