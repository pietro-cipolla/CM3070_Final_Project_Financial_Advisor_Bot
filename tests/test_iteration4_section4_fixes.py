"""
tests/test_iteration4_section4_fixes.py
Automated tests for the fix block after Iteration 4
"""

import sqlite3
from unittest.mock import patch, MagicMock

import pytest

from src.rag_pipeline import (
    extract_tickers_from_query,
    classify_query_intent,
    build_prompt,
    _history_messages,
    MAX_HISTORY_MESSAGES,
)
from src.financial_data import get_stock_summary
from src.database import init_db, save_message, load_conversation, _json_default


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test_financial_advisor.db")
    init_db(path)
    return path


def _mock_completion(content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def test_history_messages_empty_for_none_and_empty_list():
    assert _history_messages(None) == []
    assert _history_messages([]) == []


def test_history_messages_caps_at_max_and_strips_extra_keys():
    history = [
        {"role": "user", "content": f"turn {i}", "extra_field": "ignored"}
        for i in range(10)
    ]
    result = _history_messages(history)
    assert len(result) == MAX_HISTORY_MESSAGES
    # Only the most recent MAX_HISTORY_MESSAGES turns are kept.
    assert result[0]["content"] == f"turn {10 - MAX_HISTORY_MESSAGES}"
    assert result[-1]["content"] == "turn 9"
    # Only role/content survive — no stray keys forwarded to the OpenAI call.
    assert set(result[0].keys()) == {"role", "content"}


def test_extract_tickers_forwards_history_to_the_api_call():
    """
    Problem 27: a follow-up query with no company of its own (e.g. "How
    does it compare to its main rival in GPUs?" right after a Nvidia
    question) must have the prior turns forwarded into the OpenAI call, so
    the model has a chance to resolve "it", this test asserts the history
    is actually sent, not that the model resolves it correctly,
    since that reasoning happens inside the real LLM.
    """
    history = [
        {"role": "user", "content": "Tell me about Nvidia"},
        {"role": "assistant", "content": "Nvidia (NVDA) is a leading GPU maker..."},
    ]
    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("AMD"),
    ) as mock_create:
        result = extract_tickers_from_query(
            "How does it compare to its main rival in GPUs?", history=history
        )
    assert result == ["AMD"]
    sent_messages = mock_create.call_args.kwargs["messages"]
    assert {"role": "user", "content": "Tell me about Nvidia"} in sent_messages
    assert {
        "role": "assistant",
        "content": "Nvidia (NVDA) is a leading GPU maker...",
    } in sent_messages
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[-1] == {
        "role": "user",
        "content": "How does it compare to its main rival in GPUs?",
    }


def test_extract_tickers_without_history_unchanged():
    """No history passed, unaffected."""
    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("AAPL"),
    ) as mock_create:
        assert extract_tickers_from_query("What is Apple's P/E ratio?") == ["AAPL"]
    sent_messages = mock_create.call_args.kwargs["messages"]
    assert len(sent_messages) == 2  # system + current user query only


def test_classify_intent_forwards_history_to_the_api_call():
    history = [{"role": "user", "content": "Tell me about Nvidia"}]
    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("stock_query"),
    ) as mock_create:
        result = classify_query_intent(
            "How does it compare to its main rival?", history=history
        )
    assert result == "stock_query"
    sent_messages = mock_create.call_args.kwargs["messages"]
    assert {"role": "user", "content": "Tell me about Nvidia"} in sent_messages


def test_build_prompt_inserts_history_between_system_and_current_query():
    stock_data = {
        "ticker": "AMD", "name": "Advanced Micro Devices", "price": 150, "change_pct": 0.8,
        "52_week_range": "90 – 180", "pe_ratio": 45, "eps": 3.3, "beta": 1.8,
        "dividend_yield": 0.0, "recommendation": "Hold", "target_price": 165,
        "sector": "Technology", "description": "Semiconductors.",
        "news_headlines": [], "timestamp": "2026-08-20 10:00:00",
    }
    history = [
        {"role": "user", "content": "Tell me about Nvidia"},
        {"role": "assistant", "content": "Nvidia (NVDA) is..."},
    ]
    messages = build_prompt(
        stock_data, "How does it compare to its main rival in GPUs?", history=history
    )
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Tell me about Nvidia"}
    assert messages[2] == {"role": "assistant", "content": "Nvidia (NVDA) is..."}
    assert messages[-1] == {
        "role": "user",
        "content": "How does it compare to its main rival in GPUs?",
    }


def test_build_prompt_without_history_unchanged():
    stock_data = {
        "ticker": "AAPL", "name": "Apple Inc.", "price": 200, "change_pct": 1.2,
        "52_week_range": "150 – 220", "pe_ratio": 30, "eps": 6.5, "beta": 1.1,
        "dividend_yield": 0.005, "recommendation": "Buy", "target_price": 230,
        "sector": "Technology", "description": "Consumer electronics.",
        "news_headlines": [], "timestamp": "2026-07-06 10:00:00",
    }
    messages = build_prompt(stock_data, "Should I buy Apple?")
    assert len(messages) == 2  # system + current user query only, as before


def _mock_ticker_with_info(info: dict) -> MagicMock:
    mock_ticker = MagicMock()
    mock_ticker.info = info
    mock_ticker.news = []
    return mock_ticker


def _wbd_style_info(**overrides) -> dict:
    info = {
        "currentPrice": 28.44,
        "previousClose": 28.00,
        "fiftyTwoWeekLow": 20.0,
        "fiftyTwoWeekHigh": 35.0,
        "trailingPE": 583.38,
        "trailingEps": -1.28,
        "beta": 1.5,
        "dividendYield": None,
        "marketCap": 1_000_000,
        "recommendationKey": "hold",
        "targetMeanPrice": 30.0,
        "sector": "Communication Services",
        "longBusinessSummary": "A media company.",
        "longName": "Warner Bros Discovery",
    }
    info.update(overrides)
    return info


def test_get_stock_summary_flags_pe_ratio_when_eps_negative():
    """
    yfinance's trailingPE (583.38) and trailingEps (-1.28) were
    observed mutually inconsistent for WBD, a positive P/E is mathematically
    impossible with negative earnings. Once EPS is negative, pe_ratio must
    not be passed through as if it were a number that can be trusted.
    """
    with patch("src.financial_data.yf.Ticker", return_value=_mock_ticker_with_info(_wbd_style_info())):
        result = get_stock_summary("WBD")
    assert result["eps"] == -1.28
    assert result["pe_ratio"] == "N/A (negative earnings)"


def test_get_stock_summary_leaves_pe_ratio_alone_when_eps_positive():
    """Same code path, positive EPS -> pe_ratio passed through unchanged."""
    with patch(
        "src.financial_data.yf.Ticker",
        return_value=_mock_ticker_with_info(_wbd_style_info(trailingEps=3.18, trailingPE=25.32)),
    ):
        result = get_stock_summary("NFLX")
    assert result["eps"] == 3.18
    assert result["pe_ratio"] == 25.32


def test_extract_tickers_preserves_exchange_suffix():
    """
    A bare non-US symbol (such as "ISP" for Intesa Sanpaolo) was
    observed resolving to a completely unrelated company on Yahoo Finance
    (ING Groep NV). Once the model returns a suffixed symbol (e.g.
    "ISP.MI"), the extraction pipeline must preserve it instead of
    stripping anything after the "." (the "." have to survive the existing
    comma-split/alpha-first-char filtering unchanged).
    """
    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("ISP.MI"),
    ):
        result = extract_tickers_from_query("Cosa ne pensi di Intesa Sanpaolo?")
    assert result == ["ISP.MI"]


def test_extract_tickers_prompt_instructs_exchange_suffixes():
    """The fix itself: the extraction prompt have to tell the model to include
    the correct Yahoo Finance exchange suffix for non-US companies, with a
    concrete example, instead of leaving suffix handling to the model's
    unguided judgement."""
    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("ISP.MI"),
    ) as mock_create:
        extract_tickers_from_query("Cosa ne pensi di Intesa Sanpaolo?")
    system_content = mock_create.call_args.kwargs["messages"][0]["content"]
    assert "exchange suffix" in system_content
    assert "ISP.MI" in system_content

def test_search_phrase_strips_foreign_legal_suffixes():
    """
    _SUFFIX_RE only recognized English legal suffixes, so a
    yfinance longName like "Pirelli & C. S.p.A." was passed to NewsAPI
    whole, returning zero news
    results even though the ticker/financial data were correct. Both
    trailing clauses must now be stripped, applied in a loop.
    """
    from src.news_data import _search_phrase

    assert _search_phrase("Pirelli & C. S.p.A.") == "Pirelli"
    assert _search_phrase("Intesa Sanpaolo S.p.A.") == "Intesa Sanpaolo"
    assert _search_phrase("Volkswagen AG") == "Volkswagen"
    assert _search_phrase("Siemens Healthineers AG") == "Siemens Healthineers"
    assert _search_phrase("L'Oreal SA") == "L'Oreal"
    assert _search_phrase("Ford Motor Company") == "Ford"
    assert _search_phrase("Apple Inc.") == "Apple"

def test_save_and_load_message_round_trips_attachments(db_path):
    """
	a message's rich content (stock data, backtest, news,
    portfolio summary) must survive being written to SQLite and read back,
    not just exist as a local Python variable during one Streamlit rerun.
    """
    attachments = {
        "kind": "single_ticker",
        "ticker": "AAPL",
        "stock_data": {"ticker": "AAPL", "price": 200.5},
        "backtest": {"total_return_pct": 12.3, "num_trades": 4},
        "news_items": [{"title": "Apple unveils..."}],
    }
    save_message("session-a", "assistant", "Here's Apple.", db_path, attachments=attachments)
    history = load_conversation("session-a", db_path)
    assert len(history) == 1
    assert history[0]["attachments"] == attachments


def test_load_conversation_omits_attachments_key_when_none(db_path):
    """
    A message saved without attachments must come back in exactly the old
    {"role", "content"} shape (no "attachments": None key) so that every
    pre-existing equality assertion in test_iteration3.py, written before
    this fix existed, keeps passing unmodified.
    """
    save_message("session-a", "user", "hi", db_path)
    history = load_conversation("session-a", db_path)
    assert history == [{"role": "user", "content": "hi"}]
    assert "attachments" not in history[0]


def test_save_message_still_accepts_db_path_positionally(db_path):
    """
    attachments was added AFTER db_path in the signature specifically so
    that pre-existing positional calls like
    save_message(session_id, role, content, db_path) keep working, this
    guards against a future refactor accidentally reordering the params.
    """
    save_message("session-a", "user", "positional db_path still works", db_path)
    history = load_conversation("session-a", db_path)
    assert history == [{"role": "user", "content": "positional db_path still works"}]


def test_json_default_uses_item_method_for_numpy_like_scalars():
    """
    _json_default is the json.dumps() fallback used when persisting
    attachments (the GA backtester can surface np.int64/np.float64 inside a
    result dict). Instead of depending on numpy in this test, a minimal
    stand-in with the same .item() contract numpy scalars expose is used.
    """
    class _NumpyLikeScalar:
        def item(self):
            return 42

    assert _json_default(_NumpyLikeScalar()) == 42


def test_json_default_falls_back_to_str_for_other_objects():
    class _NoItemMethod:
        def __str__(self):
            return "custom-object"

    assert _json_default(_NoItemMethod()) == "custom-object"


def test_init_db_migrates_attachments_column_into_pre_existing_db(tmp_path):
    """
    A DB created before a fix for schema without the attachments
    column  problem must still work once the app is upgraded, CREATE TABLE IF NOT
    EXISTS alone is a no-op on it, so init_db() must also migrate the
    column into the pre-existing table rather than assuming every DB it
    opens was created fresh with the new schema.
    """
    path = str(tmp_path / "pre_existing.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    conn.close()

    init_db(path)  # must not raise, and must add the missing column

    columns = {row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(conversations)")}
    assert "attachments" in columns

    # And the migrated DB must be immediately usable for save/load.
    save_message("session-a", "user", "hello after migration", path)
    assert load_conversation("session-a", path) == [
        {"role": "user", "content": "hello after migration"}
    ]
