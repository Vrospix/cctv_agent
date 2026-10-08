"""Shared domain models for planning, conversation state, and query results."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, time


@dataclass(frozen=True)
class QueryConstraints:
    camera_names: tuple[str, ...] = ()
    excluded_camera_names: tuple[str, ...] = ()
    start: datetime | None = None
    end: datetime | None = None
    weekday: int | None = None
    time_start: time | None = None
    time_end: time | None = None
    sort_order: str = "oldest"
    sample_interval_minutes: int | None = None

    @property
    def camera_name(self) -> str | None:
        """Convenience value retained for single-camera callers."""
        return self.camera_names[0] if len(self.camera_names) == 1 else None

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.camera_names,
                self.excluded_camera_names,
                self.start,
                self.end,
                self.weekday is not None,
                self.time_start,
                self.time_end,
                self.sample_interval_minutes,
            )
        )


@dataclass(frozen=True)
class QueryPlan:
    """A deterministic, validated description of what a message means."""

    intent: str = "search"
    camera_action: str = "keep"
    cameras: tuple[str, ...] = ()
    date_action: str = "keep"
    start: datetime | None = None
    end: datetime | None = None
    weekday: int | None = None
    time_action: str = "keep"
    time_start: time | None = None
    time_end: time | None = None
    sort_order: str | None = None
    sample_action: str = "keep"
    sample_interval_minutes: int | None = None
    limit: int | None = None
    clarification: str | None = None

    def apply(self, previous: QueryConstraints | None) -> QueryConstraints:
        current = previous or QueryConstraints()
        included = current.camera_names
        excluded = current.excluded_camera_names
        if self.camera_action == "set":
            included, excluded = self.cameras, ()
        elif self.camera_action == "add":
            included = tuple(dict.fromkeys((*included, *self.cameras)))
            excluded = tuple(name for name in excluded if name not in self.cameras)
        elif self.camera_action == "exclude":
            included = tuple(name for name in included if name not in self.cameras)
            excluded = tuple(dict.fromkeys((*excluded, *self.cameras)))
        elif self.camera_action == "all_except":
            included, excluded = (), self.cameras
        elif self.camera_action == "all":
            included, excluded = (), ()

        start, end, weekday = current.start, current.end, current.weekday
        if self.date_action == "set":
            start, end, weekday = self.start, self.end, self.weekday
        elif self.date_action == "clear":
            start = end = None
            weekday = None

        time_start, time_end = current.time_start, current.time_end
        if self.time_action == "set":
            time_start, time_end = self.time_start, self.time_end
        elif self.time_action == "clear":
            time_start = time_end = None

        sample = current.sample_interval_minutes
        if self.sample_action == "set":
            sample = self.sample_interval_minutes
        elif self.sample_action == "clear":
            sample = None

        return replace(
            current,
            camera_names=included,
            excluded_camera_names=excluded,
            start=start,
            end=end,
            weekday=weekday,
            time_start=time_start,
            time_end=time_end,
            sort_order=self.sort_order or current.sort_order,
            sample_interval_minutes=sample,
        )


@dataclass
class ConversationState:
    constraints: QueryConstraints | None = None
    history: list[QueryConstraints] = field(default_factory=list)
    page_size: int = 20
    offset: int = 0
    last_total: int = 0
    last_page_count: int = 0
    page_history: list[int] = field(default_factory=list)
    pending_clarification: str | None = None


@dataclass(frozen=True)
class Frame:
    frame_id: int
    captured_at: datetime
    camera_name: str


@dataclass(frozen=True)
class AgentResponse:
    message: str
    frames: tuple[Frame, ...] = ()
    constraints: QueryConstraints | None = None
    total: int | None = None
    offset: int = 0
