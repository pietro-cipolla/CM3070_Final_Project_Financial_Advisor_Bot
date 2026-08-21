"""
Persistence layer — Iteration 3.

Adds the project's first persistent state: conversation memory across
browser sessions, and a simple portfolio tracker. Uses the Python standard
library's sqlite3 module.

Design notes:
  - Every function takes an explicit db_path (defaulting to DEFAULT_DB_PATH)
    so tests can point at a temporary file, instead of the
    real application database, without needing a module-level
    connection.
  - The actual .db file is NOT committed to the repository;
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

import json
import sqlite3
from contextlib import contextmanager
from datetime import date

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


def _json_default(obj):
    """
    json.dumps() fallback for objects that aren't natively JSON-serializable.

    Problema 34 (Iteration 4, Sezione 4): message attachments can contain
    numpy scalar types surfaced from the genetic-algorithm backtester (e.g.
    np.int64, np.float64 inside a backtest result dict) — these are not
    handled by the stdlib json encoder. numpy scalars all expose a zero-arg
    .item() method that returns the equivalent native Python type, so that
    is tried first; anything else falls back to str() rather than raising,
    since a slightly-lossy persisted attachment is preferable to losing the
    whole message's rich content because one nested field couldn't encode.
    """
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


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
                attachments TEXT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        # Problema 34: pre-existing DBs created before this fix won't have
        # the attachments column (CREATE TABLE IF NOT EXISTS is a no-op on
        # them), so migrate it in explicitly rather than only handling the
        # fresh-DB case — same idempotent-migration approach as the rest of
        # this module's "safe to call on every app startup" contract.
        existing_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(conversations)")
        }
        if "attachments" not in existing_columns:
            conn.execute("ALTER TABLE conversations ADD COLUMN attachments TEXT")
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

def save_message(
    session_id: str,
    role: str,
    content: str,
    db_path: str = DEFAULT_DB_PATH,
    attachments: dict | None = None,
) -> None:
    """
    Persist a single chat message for a session.

    Problema 34 (Iteration 4, Sezione 4): attachments is the structured data
    behind a message's rich content (retrieved stock data, backtest result,
    news items, or portfolio summary — see app.py's _render_message_attachments
    dispatcher), JSON-serialized here so it survives both an app restart and
    any same-session Streamlit rerun, not just kept as a local variable that
    vanishes once the script re-executes. None (the default) covers plain
    messages with no rich content, and is stored as SQL NULL, not the string
    "null".

    attachments is deliberately added AFTER db_path (not before it) so that
    every pre-existing call site that passes db_path positionally — e.g.
    save_message(session_id, role, content, db_path), all over
    test_iteration3.py — keeps landing db_path in the db_path parameter
    rather than silently shifting it into attachments. app.py's own new
    call site passes attachments=... by keyword, so it is unaffected by
    this ordering either way.
    """
    serialized = (
        json.dumps(attachments, default=_json_default) if attachments is not None else None
    )
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, attachments) VALUES (?, ?, ?, ?)",
            (session_id, role, content, serialized),
        )


def load_conversation(session_id: str, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Return the full message history for a session, oldest first, in the
    same {"role": ..., "content": ...} shape used by st.session_state.messages.

    Problema 34 (Iteration 4, Sezione 4): a message saved with attachments
    also gets an "attachments" key holding the parsed dict. That key is
    deliberately OMITTED (not set to None) for messages saved without
    attachments — both plain messages saved after this fix and every
    message saved before it (their attachments column is NULL) — so the
    returned shape for those rows is byte-for-byte the same
    {"role": ..., "content": ...} dict as before this fix, and every
    pre-existing equality assertion against that exact shape (see
    test_iteration3.py) keeps passing unmodified. Callers that want rich
    content use msg.get("attachments"), which is None either way whether
    the key is absent or explicitly None.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content, attachments FROM conversations WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    messages = []
    for row in rows:
        message = {"role": row["role"], "content": row["content"]}
        if row["attachments"]:
            message["attachments"] = json.loads(row["attachments"])
        messages.append(message)
    return messages


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

    Raises ValueError for non-positive shares or purchase_price, or for a
    purchase_date in the future, rather than silently storing a nonsensical
    holding (e.g. 0 shares, a negative price from a typo, or a trade dated
    tomorrow) — callers (the UI layer) are expected to catch this and show
    a friendly message rather than letting a bad row into the DB.

    The future-date check is a defence-in-depth measure, not the only
    safeguard: app.py's date picker already constrains max_value to today,
    same pattern as the existing shares/price checks pairing a UI
    constraint with a code-level one (see Chapter 4 of the Draft Report).
    purchase_date is expected as an ISO "YYYY-MM-DD" string; a value that
    isn't a valid ISO date is also rejected here rather than stored as an
    unparseable string that would only fail later when read back.
    """
    if shares <= 0:
        raise ValueError("shares must be a positive number")
    if purchase_price <= 0:
        raise ValueError("purchase_price must be a positive number")
    try:
        parsed_date = date.fromisoformat(purchase_date)
    except ValueError:
        raise ValueError(f"purchase_date must be an ISO date (YYYY-MM-DD), got {purchase_date!r}")
    if parsed_date > date.today():
        raise ValueError("purchase_date cannot be in the future")

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
