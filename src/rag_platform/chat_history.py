"""
Persistent chat history — a single SQLite database (data/chat_history.sqlite3)
shared across all companies, with every row tagged by company_id.

Design decision: one DB, not one-per-company. Chat history isn't the kind of
data that needs the hard isolation vector stores need (it's the user's own
question log, not another tenant's source documents), and a single DB makes
"show me everything I've asked across all companies" and the cross-company
comparison feature trivial to implement — one query instead of a fan-out
across N per-company databases.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    starred INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_company ON conversations(company_id, created_at DESC);
"""


@dataclass
class ChatEntry:
    id: int
    company_id: str
    question: str
    answer: str
    citations: list[dict]
    starred: bool
    created_at: str


class ChatHistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add(self, company_id: str, question: str, answer: str, citations: Optional[list[dict]] = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (company_id, question, answer, citations_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (company_id, question, answer, json.dumps(citations or []), datetime.now(timezone.utc).isoformat()),
            )
            return cur.lastrowid

    def list_for_company(self, company_id: str, limit: int = 50, search: Optional[str] = None) -> list[ChatEntry]:
        query = "SELECT id, company_id, question, answer, citations_json, starred, created_at FROM conversations WHERE company_id = ?"
        params: list = [company_id]
        if search:
            query += " AND (question LIKE ? OR answer LIKE ?)"
            like = f"%{search}%"
            params += [like, like]
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def toggle_star(self, entry_id: int, company_id: str) -> bool:
        """Toggle star, scoped to the owning tenant (H7 fix).

        IDs are sequential SQLite integers — trivially enumerable — so the
        company_id guard is what stops one tenant's UI from starring or
        deleting another tenant's rows.

        Returns True if a row was actually toggled.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE conversations SET starred = 1 - starred WHERE id = ? AND company_id = ?",
                (entry_id, company_id),
            )
            return cur.rowcount > 0

    def delete(self, entry_id: int, company_id: str) -> bool:
        """Delete scoped to the owning tenant (H7 fix). Returns True if deleted."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM conversations WHERE id = ? AND company_id = ?", (entry_id, company_id)
            )
            return cur.rowcount > 0

    @staticmethod
    def _row_to_entry(row) -> ChatEntry:
        return ChatEntry(
            id=row[0],
            company_id=row[1],
            question=row[2],
            answer=row[3],
            citations=json.loads(row[4]),
            starred=bool(row[5]),
            created_at=row[6],
        )
