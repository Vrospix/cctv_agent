"""SQLite persistence with parameterized, read-only frame queries."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .cameras import CAMERAS
from .models import Frame, QueryConstraints


class FrameDatabase:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS frames (
                frame_id INTEGER PRIMARY KEY,
                captured_at TEXT NOT NULL,
                camera_name TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_frames_camera_time "
            "ON frames(camera_name, captured_at)"
        )

    def close(self) -> None:
        self.connection.close()

    def latest_capture_date(self) -> date | None:
        """Return the newest simulated day currently stored."""
        latest = self.connection.execute("SELECT MAX(captured_at) FROM frames").fetchone()[0]
        return datetime.fromisoformat(latest).date() if latest else None

    def ensure_current(self, *, today: date | None = None) -> int:
        """Incrementally extend the simulation through the day the app runs."""
        return self.seed(until=today or date.today())

    def seed(self, *, until: date, start: date = date(2026, 1, 1)) -> int:
        """Generate missing five-minute frames through ``until``."""
        if until < start:
            raise ValueError("The seed end date cannot be before the start date.")
        rows: list[tuple[str, str]] = []
        self.connection.execute("BEGIN IMMEDIATE")
        latest = self.connection.execute("SELECT MAX(captured_at) FROM frames").fetchone()[0]
        if latest:
            moment = datetime.fromisoformat(latest) + timedelta(minutes=5)
        else:
            moment = datetime.combine(start, time.min)
        final = datetime.combine(until + timedelta(days=1), time.min)
        names = tuple(CAMERAS.values())
        while moment < final:
            timestamp = moment.isoformat(timespec="seconds")
            rows.extend((timestamp, name) for name in names)
            if len(rows) >= 10_000:
                self.connection.executemany(
                    "INSERT INTO frames(captured_at, camera_name) VALUES (?, ?)", rows
                )
                rows.clear()
            moment += timedelta(minutes=5)
        if rows:
            self.connection.executemany(
                "INSERT INTO frames(captured_at, camera_name) VALUES (?, ?)", rows
            )
        self.connection.commit()
        return int(self.connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])

    @staticmethod
    def _where(constraints: QueryConstraints) -> tuple[str, list[object]]:
        clauses: list[str] = []
        parameters: list[object] = []
        if constraints.camera_names:
            placeholders = ", ".join("?" for _ in constraints.camera_names)
            clauses.append(f"camera_name IN ({placeholders})")
            parameters.extend(constraints.camera_names)
        if constraints.excluded_camera_names:
            placeholders = ", ".join("?" for _ in constraints.excluded_camera_names)
            clauses.append(f"camera_name NOT IN ({placeholders})")
            parameters.extend(constraints.excluded_camera_names)
        if constraints.start:
            clauses.append("captured_at >= ?")
            parameters.append(constraints.start.isoformat(timespec="seconds"))
        if constraints.end:
            clauses.append("captured_at < ?")
            parameters.append(constraints.end.isoformat(timespec="seconds"))
        if constraints.weekday is not None:
            clauses.append("CAST(strftime('%w', captured_at) AS INTEGER) = ?")
            parameters.append((constraints.weekday + 1) % 7)
        if constraints.time_start:
            clauses.append("time(captured_at) >= ?")
            parameters.append(constraints.time_start.isoformat(timespec="seconds"))
        if constraints.time_end:
            clauses.append("time(captured_at) < ?")
            parameters.append(constraints.time_end.isoformat(timespec="seconds"))
        if constraints.sample_interval_minutes:
            clauses.append("CAST(strftime('%s', captured_at) AS INTEGER) % ? = 0")
            parameters.append(constraints.sample_interval_minutes * 60)
        return (" AND ".join(clauses) if clauses else "1 = 1"), parameters

    def count(self, constraints: QueryConstraints) -> int:
        where, parameters = self._where(constraints)
        row = self.connection.execute(
            f"SELECT COUNT(*) FROM frames WHERE {where}", parameters
        ).fetchone()
        return int(row[0])

    def query(
        self, constraints: QueryConstraints, *, limit: int = 20, offset: int = 0
    ) -> tuple[Frame, ...]:
        where, parameters = self._where(constraints)
        parameters.extend((max(1, min(limit, 100)), max(0, offset)))
        order_by = {
            "oldest": "captured_at ASC, frame_id ASC",
            "newest": "captured_at DESC, frame_id DESC",
            "random": "RANDOM()",
        }.get(constraints.sort_order, "captured_at ASC, frame_id ASC")
        cursor = self.connection.execute(
            f"SELECT frame_id, captured_at, camera_name FROM frames "
            f"WHERE {where} ORDER BY {order_by} LIMIT ? OFFSET ?",
            parameters,
        )
        return tuple(
            Frame(frame_id=row[0], captured_at=datetime.fromisoformat(row[1]), camera_name=row[2])
            for row in cursor.fetchall()
        )
