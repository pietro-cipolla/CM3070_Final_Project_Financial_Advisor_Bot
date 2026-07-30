"""
database.py
Persistence layer — Iteration 3.

Adds the project's first persistent state: conversation memory across
browser sessions, and a simple portfolio tracker. Uses the Python standard
library's sqlite3 module.

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
    Create the conversations and portfolio tables if they do not already
    exist. Safe to call on every app startup.
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
            """
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                shares REAL NOT NULL,
                purchase_price REAL NOT NULL,
                purchase_date TEXT NOT NULL,
                added_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_portfolio_session ON portfolio(session_id)"
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


# Portfolio tracker

def add_holding(
    session_id: str,
    ticker: str,
    shares: float,
    purchase_price: float,
    purchase_date: str,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Insert a new portfolio holding and return its id.

    Raises ValueError for non-positive shares or purchase_price rather than
    silently storing a nonsensical holding (e.g. 0 shares, or a negative
    price from a typo) — callers (the UI layer) are expected to catch this
    and show a friendly message rather than letting a bad row into the DB.
    """
    if shares <= 0:
        raise ValueError("shares must be a positive number")
    if purchase_price <= 0:
        raise ValueError("purchase_price must be a positive number")

    with _connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO portfolio (session_id, ticker, shares, purchase_price, purchase_date) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, ticker.upper(), shares, purchase_price, purchase_date),
        )
        return cursor.lastrowid


def get_portfolio(session_id: str, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Return all holdings for a session as a list of dicts, oldest first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, ticker, shares, purchase_price, purchase_date "
            "FROM portfolio WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def remove_holding(session_id: str, holding_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Delete a single holding by id, scoped to session_id so a session can
    never delete another session's holding. Returns True if a row was
    actually deleted, False if no matching row was found.
    """
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM portfolio WHERE id = ? AND session_id = ?",
            (holding_id, session_id),
        )
        return cursor.rowcount > 0
