from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from task_agent.models import TaskState


def _default_db_path() -> Path:
    override = os.getenv("TASK_AGENT_DB_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "cs_agent" / "knowledge" / "task_agent.db"


class TaskStateStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_states (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    task_status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save(self, state: TaskState) -> None:
        state.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        payload = state.model_dump_json(ensure_ascii=False)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO task_states (task_id, session_id, task_type, task_status, state_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    task_type=excluded.task_type,
                    task_status=excluded.task_status,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (
                    state.task_id,
                    state.session_id,
                    state.task_type,
                    state.task_status,
                    payload,
                    state.updated_at,
                ),
            )
            conn.commit()

    def load(self, task_id: str) -> TaskState:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT state_json FROM task_states WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return TaskState.model_validate(json.loads(row[0]))
