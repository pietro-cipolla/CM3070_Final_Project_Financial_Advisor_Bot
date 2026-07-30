"""
database.py
Persistence layer — Iteration 3.

Adds the project's first persistent state: conversation memory across
browser sessions. Uses the Python standard library's sqlite3 module.

Design notes:
  - Every function takes an explicit db_path (defaulting to DEFAULT_DB_PATH)
    so tests can point at a temporary file, instead of the
    real application database, without needing a module-level
    connection.
  - The actual .db file is NOT committed to the repository (see .gitignore);
    only this schema-creation code is. init_db() is idempotent (CREATE TABLE
    IF NOT EXISTS) and safe to call on every app startup.
  - KNOWN LIMITATION: Streamlit
    Community Cloud has an ephemeral filesystem, so this file, and
    everything stored in it, does not persist across app restarts once
    deployed there for user testing (Iteration 5). It behaves normally in
    local development (`streamlit run app.py`).
  - clear_conversation() always filters by session_id, so one session can
    never delete or read another session's data even if it somehow guessed
    a numeric id.
"""

import sqlite3
from contextlib import contextmanager

DEFAULT_DB_PATH = "financial_advisor.db"


@contextmanager
def _connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Create the conversations table if it does not already exist. Safe to
    call on every app startup.
    """
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)"
        )


# Conversation memory

def save_message(session_id: str, role: str, content: str, db_path: str = DEFAULT_DB_PATH) -> None:
    """Persist a single chat message for a session."""
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def load_conversation(session_id: str, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Return the full message history for a session, oldest first, in the
    same {"role": ..., "content": ...} shape used by st.session_state.messages.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def clear_conversation(session_id: str, db_path: str = DEFAULT_DB_PATH) -> None:
    """Delete all stored messages for a session."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
