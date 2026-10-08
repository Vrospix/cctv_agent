"""Conversation orchestration, validation, guardrails, and query execution."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime, time, timedelta

from .cameras import CAMERAS, resolve_cameras, suggest_cameras
from .database import FrameDatabase
from .llm_intent import IntentParser, IntentParserError
from .language import normalize_request
from .models import AgentResponse, ConversationState, QueryConstraints, QueryPlan
from .planner import build_query_plan


BLOCKED_PATTERNS = re.compile(
    r"\b(?:drop|delete|alter|truncate)\s+(?:all\s+)?(?:table|database|frames?|records?)\b|"
    r"\binsert\s+into\b|\bupdate\s+(?:frames?|records?|table)\s+set\b|"
    r"\b(?:attach|detach)\s+database\b|\bpragma\b|\bexecute\s+(?:sql|query)\b|"
    r"--|/\*|\*/",
    re.IGNORECASE,
)
WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
CONTROL_INTENTS = {
    "help",
    "reset",
    "next_page",
    "previous_page",
    "undo",
    "show_filters",
    "set_limit",
}
CONTROL_GROUNDING = {
    "help": re.compile(r"\bhelp\b", re.I),
    "reset": re.compile(r"\b(?:reset|restart|start over|new search|clear all)\b", re.I),
    "next_page": re.compile(r"\b(?:next|more|forward|after)\b", re.I),
    "previous_page": re.compile(
        r"\b(?:previous|prior|back|backward|earlier|before|batch)\b", re.I
    ),
    "undo": re.compile(r"\b(?:undo|revert|mistake|change)\b", re.I),
    "show_filters": re.compile(r"\b(?:filter|searching|current|active)\b", re.I),
    "set_limit": re.compile(r"\b(?:limit|per page)\b", re.I),
}
NEGATION_CUE = re.compile(r"\b(?:not|no|except|exclude|excluding|without|dont|don't)\b", re.I)
TEMPORAL_CUE = re.compile(
    r"\b(?:today|tomorrow|yesterday|day|week|month|year|ago|morning|afternoon|"
    r"evening|night|noon|midnight|lunch|before|after|between|from|until|am|pm|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|jan(?:uary)?|"
    r"feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b|\d{1,2}:\d{2}",
    re.I,
)
SORT_CUE = re.compile(r"\b(?:latest|newest|recent|earliest|oldest|random)\b", re.I)
SAMPLE_CUE = re.compile(r"\b(?:every|per hour|sample|each hour)\b", re.I)


class CCTVQueryAgent:
    def __init__(
        self,
        database: FrameDatabase,
        *,
        today: date | None = None,
        intent_parser: IntentParser | None = None,
    ) -> None:
        self.database = database
        self.today = today or date.today()
        self.intent_parser = intent_parser
        self.state = ConversationState()
        self.last_intent_source = "deterministic"
        self.last_llm_error: str | None = None

    @property
    def context(self) -> QueryConstraints | None:
        return self.state.constraints

    def describe_context(self) -> str:
        """Return a user-facing summary of the active filters."""
        if self.context is None:
            return "No active filters."
        return self._describe(self.context, prefix="Active filters")

    def ask(self, request: str, *, limit: int | None = None) -> AgentResponse:
        text = request.strip()
        if not text:
            return AgentResponse("Please enter a CCTV frame request.")
        if BLOCKED_PATTERNS.search(text):
            return AgentResponse("I can only perform read-only CCTV frame searches.")

        try:
            plan = self._plan_request(text)
        except (ValueError, OverflowError) as error:
            return AgentResponse(f"I couldn't safely interpret that request: {error}")

        if plan.intent == "help":
            return AgentResponse(
                "Try a camera/date search, refine a filter, ask for counts or active filters, "
                "use 'next'/'previous', 'undo', '/limit 20', or 'start over'."
            )
        if plan.intent == "unsupported_analysis":
            return AgentResponse(
                "I can find CCTV frame records, but I cannot determine traffic, accidents, "
                "or road conditions from the stored metadata."
            )
        if plan.intent == "reset":
            page_size = self.state.page_size
            self.state = ConversationState(page_size=page_size)
            return AgentResponse("Search cleared. What would you like to find?")
        if plan.intent == "set_limit":
            return self._set_limit(plan.limit)
        if plan.intent == "show_filters":
            if self.context is None:
                return AgentResponse("There are no active filters yet.")
            return AgentResponse(
                self._describe(self.context, prefix="Active filters"),
                constraints=self.context,
            )
        if plan.intent == "undo":
            return self._undo()
        if plan.intent == "next_page":
            return self._next_page(plan.limit)
        if plan.intent == "previous_page":
            return self._previous_page()
        if plan.intent == "unknown":
            return self._unknown_response()
        if plan.intent == "clarify":
            partial = plan.apply(self.context)
            if not partial.is_empty:
                self.state.constraints = partial
            self.state.pending_clarification = "request"
            return AgentResponse(
                plan.clarification or "Which camera did you mean?",
                constraints=partial,
            )

        previous = self.context
        current = plan.apply(previous)
        try:
            current = self._validate_constraints(current)
        except ValueError as error:
            return AgentResponse(
                f"I couldn't safely apply that request: {error}",
                constraints=previous,
            )
        selection_only = plan.limit is not None and plan.sort_order is not None
        if current.is_empty and not selection_only and plan.intent != "count":
            return AgentResponse("Tell me at least a camera, date, or time period to search.")

        page_size = self.state.page_size
        try:
            if limit is not None:
                page_size = self._validated_limit(limit)
            if plan.limit is not None:
                page_size = self._validated_limit(plan.limit)
        except ValueError as error:
            return AgentResponse(str(error), constraints=previous)

        was_pending = self.state.pending_clarification is not None
        if previous is not None and current != previous and not was_pending:
            self.state.history.append(previous)
        self.state.constraints = current
        self.state.pending_clarification = None
        self._reset_pagination()
        self.state.page_size = page_size

        if plan.intent == "count":
            total = self.database.count(current)
            self.state.last_total = total
            message = self._describe(current) + f"\nThere are {total:,} matching frame(s)."
            return AgentResponse(message, constraints=current, total=total)
        return self._run_search()

    def _plan_request(self, text: str) -> QueryPlan:
        deterministic: QueryPlan | None = None
        deterministic_error: ValueError | OverflowError | None = None
        self.last_intent_source = "deterministic"
        self.last_llm_error = None

        try:
            deterministic = build_query_plan(
                text, today=self.today, context=self.context
            )
        except (ValueError, OverflowError) as error:
            deterministic_error = error

        if self.intent_parser is None or (
            deterministic is not None
            and deterministic.intent not in {"unknown", "clarify"}
        ):
            if deterministic_error:
                raise deterministic_error
            return deterministic or QueryPlan(intent="unknown")

        try:
            interpretation = self.intent_parser.interpret(
                text,
                today=self.today,
                context=self.context,
            )
            canonical = interpretation.canonical_request()
            if not canonical:
                raise IntentParserError("The local model did not identify an intent.")
            candidate = build_query_plan(
                canonical,
                today=self.today,
                context=self.context,
            )
            if candidate.intent == "unknown":
                raise IntentParserError("The structured intent could not be validated.")
            self._validate_ai_candidate(text, interpretation.intent, canonical, candidate)
            self.last_intent_source = "local_ai"
            return candidate
        except (IntentParserError, ValueError, OverflowError) as error:
            self.last_llm_error = str(error)
            if deterministic_error:
                raise deterministic_error
            return deterministic or QueryPlan(intent="unknown")

    def _validate_ai_candidate(
        self,
        original: str,
        category: str,
        meaning: str,
        candidate: QueryPlan,
    ) -> None:
        if category == "control":
            grounding = CONTROL_GROUNDING.get(candidate.intent)
            if grounding is None or not grounding.search(original):
                raise IntentParserError("The model's control intent was not grounded in the request.")
        elif candidate.intent in CONTROL_INTENTS:
            raise IntentParserError("The model's intent label did not match its meaning.")

        original_cameras = set(resolve_cameras(original)) | set(suggest_cameras(original))
        if not set(candidate.cameras).issubset(original_cameras):
            raise IntentParserError("The model added a camera that was not in the request.")

        original_text = normalize_request(original)
        meaning_text = normalize_request(meaning)
        original_numbers = set(re.findall(r"\b\d+\b", original_text))
        meaning_numbers = set(re.findall(r"\b\d+\b", meaning_text))
        if not meaning_numbers.issubset(original_numbers):
            raise IntentParserError("The model added a number that was not in the request.")

        excludes_camera = candidate.camera_action in {"exclude", "all_except"}
        if candidate.cameras and bool(NEGATION_CUE.search(original)) != excludes_camera:
            raise IntentParserError("The model changed the camera negation.")

        changed_time = candidate.date_action != "keep" or candidate.time_action != "keep"
        if changed_time and not TEMPORAL_CUE.search(original):
            raise IntentParserError("The model added a date or time that was not in the request.")
        if candidate.sort_order and not SORT_CUE.search(original):
            raise IntentParserError("The model added sorting that was not in the request.")
        if candidate.sample_action != "keep" and not SAMPLE_CUE.search(original):
            raise IntentParserError("The model added sampling that was not in the request.")

    def _validated_limit(self, value: int) -> int:
        if not 1 <= value <= 100:
            raise ValueError("The result limit must be between 1 and 100.")
        return value

    def _set_limit(self, value: int | None) -> AgentResponse:
        try:
            self.state.page_size = self._validated_limit(value or 0)
        except ValueError as error:
            return AgentResponse(str(error), constraints=self.context)
        self._reset_pagination()
        return AgentResponse(
            f"Result limit changed to {self.state.page_size}.", constraints=self.context
        )

    def _validate_constraints(self, constraints: QueryConstraints) -> QueryConstraints:
        unknown = set(constraints.camera_names + constraints.excluded_camera_names) - set(
            CAMERAS.values()
        )
        if unknown:
            raise ValueError(f"Unsupported camera: {sorted(unknown)[0]}")
        if constraints.start and constraints.start.date() > self.today:
            raise ValueError("The requested period begins after the available current date.")
        available_end = datetime.combine(self.today + timedelta(days=1), time.min)
        if constraints.end and constraints.end > available_end:
            constraints = replace(constraints, end=available_end)
        if constraints.start and constraints.end and constraints.end <= constraints.start:
            raise ValueError("The requested date range is empty.")
        if constraints.sample_interval_minutes is not None:
            if not 5 <= constraints.sample_interval_minutes <= 1_440:
                raise ValueError("Sampling intervals must be between 5 and 1,440 minutes.")
            if constraints.sample_interval_minutes % 5:
                raise ValueError(
                    "Sampling intervals must be a multiple of 5 minutes because "
                    "frames are stored every 5 minutes."
                )
        return constraints

    def _run_search(self) -> AgentResponse:
        constraints = self.context
        if constraints is None:
            return AgentResponse("Start with a camera, date, or time request first.")
        total = self.database.count(constraints)
        self.state.last_total = total
        if total == 0:
            self.state.offset = 0
            return AgentResponse(
                self._describe(constraints) + "\nNo frames matched those filters.",
                constraints=constraints,
                total=0,
            )
        if self.state.offset >= total:
            self.state.offset = max(0, ((total - 1) // self.state.page_size) * self.state.page_size)
        frames = self.database.query(
            constraints,
            limit=self.state.page_size,
            offset=self.state.offset,
        )
        self.state.last_page_count = len(frames)
        first, last = self.state.offset + 1, self.state.offset + len(frames)
        message = (
            self._describe(constraints)
            + f"\nFound {total:,} frame(s). Showing {first}-{last}."
        )
        return AgentResponse(
            message,
            frames,
            constraints,
            total=total,
            offset=self.state.offset,
        )

    def _next_page(self, requested_size: int | None) -> AgentResponse:
        if self.context is None or self.state.pending_clarification:
            return AgentResponse("Run a complete search before asking for the next page.")
        if self.context.sort_order == "random":
            return AgentResponse(
                "Random results do not have stable pages. Repeat the random search "
                "for a new sample.",
                constraints=self.context,
            )
        total = self.database.count(self.context)
        next_offset = self.state.offset + (self.state.last_page_count or self.state.page_size)
        if next_offset >= total:
            return AgentResponse(
                "You are already at the end of the matching results.",
                constraints=self.context,
                total=total,
                offset=self.state.offset,
            )
        if requested_size is not None:
            try:
                self.state.page_size = self._validated_limit(requested_size)
            except ValueError as error:
                return AgentResponse(str(error), constraints=self.context)
        self.state.page_history.append(self.state.offset)
        self.state.offset = next_offset
        return self._run_search()

    def _previous_page(self) -> AgentResponse:
        if self.context is None or self.state.pending_clarification:
            return AgentResponse("Run a complete search before asking for previous results.")
        if self.context.sort_order == "random":
            return AgentResponse(
                "Random results do not have stable previous pages.", constraints=self.context
            )
        if self.state.offset == 0 and not self.state.page_history:
            return AgentResponse(
                "You are already at the first page.",
                constraints=self.context,
                total=self.state.last_total,
            )
        self.state.offset = (
            self.state.page_history.pop()
            if self.state.page_history
            else max(0, self.state.offset - self.state.page_size)
        )
        return self._run_search()

    def _undo(self) -> AgentResponse:
        if not self.state.history:
            return AgentResponse(
                "There is no previous search to restore.", constraints=self.context
            )
        self.state.constraints = self.state.history.pop()
        self.state.pending_clarification = None
        self._reset_pagination()
        return self._run_search()

    def _reset_pagination(self) -> None:
        self.state.offset = 0
        self.state.last_page_count = 0
        self.state.page_history.clear()

    def _unknown_response(self) -> AgentResponse:
        if self.context is None:
            cameras = ", ".join(CAMERAS)
            return AgentResponse(
                f"I can only help search CCTV frames. Supported cameras: {cameras}."
            )
        return AgentResponse(
            "I couldn't tell what you want to change. Try a camera, date, time, sorting, "
            "'previous', 'undo', 'active filters', or 'start over'.",
            constraints=self.context,
        )

    def _describe(self, constraints: QueryConstraints, *, prefix: str = "Searching") -> str:
        parts: list[str] = []
        if constraints.camera_names:
            parts.append("cameras=" + ", ".join(constraints.camera_names))
        elif constraints.excluded_camera_names:
            parts.append("cameras=all except " + ", ".join(constraints.excluded_camera_names))
        else:
            parts.append("cameras=all")

        if constraints.start and constraints.end:
            last_inclusive = (constraints.end - timedelta(days=1)).date()
            if constraints.start.date() == last_inclusive:
                parts.append(f"date={constraints.start.date().isoformat()}")
            else:
                parts.append(
                    f"dates={constraints.start.date().isoformat()} to {last_inclusive.isoformat()}"
                )
        elif constraints.start:
            parts.append(f"from={constraints.start.date().isoformat()}")
        elif constraints.end:
            parts.append(f"before={constraints.end.date().isoformat()}")
        if constraints.weekday is not None:
            parts.append("weekday=" + WEEKDAY_NAMES[constraints.weekday])
        if constraints.time_start or constraints.time_end:
            start = (
                constraints.time_start.strftime("%H:%M")
                if constraints.time_start
                else "start of day"
            )
            end = constraints.time_end.strftime("%H:%M") if constraints.time_end else "end of day"
            parts.append(f"time={start}-{end}")
        if constraints.sample_interval_minutes:
            parts.append(f"sample=every {constraints.sample_interval_minutes} minutes")
        parts.append(f"order={constraints.sort_order}")
        return prefix + ": " + "; ".join(parts)
