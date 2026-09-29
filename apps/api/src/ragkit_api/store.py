"""노트북·대화·노트 저장소 (SQLite 파일 하나, 기본 data/app/notebooks.sqlite).

요청마다 연결을 열고 닫아 웹 서버 스레드풀에서도 안전하게 쓴다.
인용(citations)은 답변 당시 조문을 그대로 저장해, 인덱스를 다시 만들어도 기록이 깨지지 않는다.
"""

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS notebooks (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, laws TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    role TEXT NOT NULL, content TEXT NOT NULL, citations TEXT NOT NULL, error TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title TEXT NOT NULL, content TEXT NOT NULL, citations TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_notebook ON messages(notebook_id, created_at);
CREATE INDEX IF NOT EXISTS notes_notebook ON notes(notebook_id, updated_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


def _notebook(row: sqlite3.Row) -> dict:
    return {**dict(row), "laws": json.loads(row["laws"])}


def _with_citations(row: sqlite3.Row) -> dict:
    record = dict(row)
    record.pop("notebook_id", None)
    record["citations"] = json.loads(record["citations"])
    return record


class NotebookStore:
    """노트북 CRUD. 반환값은 schemas의 Notebook / Message / Note와 같은 모양의 dict다."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:  # 블록이 끝나면 commit, 예외면 rollback
                yield conn
        finally:
            conn.close()

    def _touch(self, conn: sqlite3.Connection, notebook_id: str) -> None:
        conn.execute("UPDATE notebooks SET updated_at = ? WHERE id = ?", (_now(), notebook_id))

    # 노트북
    def list_notebooks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM notebooks ORDER BY updated_at DESC").fetchall()
        return [_notebook(r) for r in rows]

    def create_notebook(self, title: str, laws: list[str]) -> dict:
        now = _now()
        record = {"id": _new_id(), "title": title, "laws": laws, "created_at": now, "updated_at": now}
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO notebooks VALUES (:id, :title, :laws, :created_at, :updated_at)",
                {**record, "laws": json.dumps(laws, ensure_ascii=False)},
            )
        return record

    def get_notebook(self, notebook_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM notebooks WHERE id = ?", (notebook_id,)).fetchone()
        return _notebook(row) if row else None

    def update_notebook(self, notebook_id: str, title: str | None = None, laws: list[str] | None = None) -> dict | None:
        with self._connect() as conn:
            if title is not None:
                conn.execute("UPDATE notebooks SET title = ? WHERE id = ?", (title, notebook_id))
            if laws is not None:
                conn.execute(
                    "UPDATE notebooks SET laws = ? WHERE id = ?",
                    (json.dumps(laws, ensure_ascii=False), notebook_id),
                )
            self._touch(conn, notebook_id)
        return self.get_notebook(notebook_id)

    def delete_notebook(self, notebook_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,)).rowcount > 0

    # 대화
    def list_messages(self, notebook_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE notebook_id = ? ORDER BY created_at, rowid", (notebook_id,)
            ).fetchall()
        return [_with_citations(r) for r in rows]

    def add_message(
        self,
        notebook_id: str,
        role: str,
        content: str,
        citations: list[dict] | None = None,
        error: str | None = None,
    ) -> dict:
        record = {"id": _new_id(), "role": role, "content": content, "citations": citations or [],
                  "error": error, "created_at": _now()}
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages VALUES (:id, :notebook_id, :role, :content, :citations, :error, :created_at)",
                {**record, "notebook_id": notebook_id,
                 "citations": json.dumps(record["citations"], ensure_ascii=False)},
            )
            self._touch(conn, notebook_id)
        return record

    def clear_messages(self, notebook_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE notebook_id = ?", (notebook_id,))

    # 노트
    def list_notes(self, notebook_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM notes WHERE notebook_id = ? ORDER BY updated_at DESC", (notebook_id,)
            ).fetchall()
        return [_with_citations(r) for r in rows]

    def create_note(self, notebook_id: str, title: str, content: str, citations: list[dict]) -> dict:
        now = _now()
        record = {"id": _new_id(), "title": title, "content": content, "citations": citations,
                  "created_at": now, "updated_at": now}
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO notes VALUES (:id, :notebook_id, :title, :content, :citations, :created_at, :updated_at)",
                {**record, "notebook_id": notebook_id, "citations": json.dumps(citations, ensure_ascii=False)},
            )
            self._touch(conn, notebook_id)
        return record

    def update_note(self, notebook_id: str, note_id: str, title: str | None, content: str | None) -> dict | None:
        with self._connect() as conn:
            changed = conn.execute(
                """
                UPDATE notes SET title = coalesce(?, title), content = coalesce(?, content), updated_at = ?
                WHERE id = ? AND notebook_id = ?
                """,
                (title, content, _now(), note_id, notebook_id),
            ).rowcount
            if not changed:
                return None
            self._touch(conn, notebook_id)
            row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return _with_citations(row)

    def delete_note(self, notebook_id: str, note_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "DELETE FROM notes WHERE id = ? AND notebook_id = ?", (note_id, notebook_id)
            ).rowcount > 0
