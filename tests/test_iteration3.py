"""
tests/test_iteration3.py
Automated tests for Iteration 3: SQLite conversation memory, the portfolio
tracker, and the two prompt-level inclusive-design improvements (simplified
mode, language-matching instruction).
"""

import sqlite3

import pytest

from src.database import (
    init_db,
    save_message,
    load_conversation,
    clear_conversation,
    add_holding,
    get_portfolio,
    remove_holding,
)
from src.portfolio import compute_holding_pnl, compute_portfolio_summary
from src.rag_pipeline import build_prompt, LANGUAGE_MATCH_INSTRUCTION, SIMPLIFIED_MODE_INSTRUCTION


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test_financial_advisor.db")
    init_db(path)
    return path


# Conversation memory

def test_init_db_creates_both_tables(db_path):
    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "conversations" in tables
    assert "portfolio" in tables


def test_save_and_load_conversation_preserves_order(db_path):
    save_message("session-a", "user", "What is Tesla's P/E ratio?", db_path)
    save_message("session-a", "assistant", "Tesla's P/E ratio is...", db_path)
    save_message("session-a", "user", "And Ford's?", db_path)

    history = load_conversation("session-a", db_path)

    assert [m["content"] for m in history] == [
        "What is Tesla's P/E ratio?",
        "Tesla's P/E ratio is...",
        "And Ford's?",
    ]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]


def test_load_conversation_unknown_session_returns_empty_list(db_path):
    assert load_conversation("never-used-session", db_path) == []


def test_clear_conversation_removes_all_messages(db_path):
    save_message("session-b", "user", "Tell me about Apple", db_path)
    save_message("session-b", "assistant", "Apple is...", db_path)

    clear_conversation("session-b", db_path)

    assert load_conversation("session-b", db_path) == []


def test_conversations_are_isolated_by_session_id(db_path):
    save_message("session-x", "user", "Message in session X", db_path)
    save_message("session-y", "user", "Message in session Y", db_path)

    assert [m["content"] for m in load_conversation("session-x", db_path)] == ["Message in session X"]
    assert [m["content"] for m in load_conversation("session-y", db_path)] == ["Message in session Y"]


def test_conversation_handles_a_long_history_in_order(db_path):
    for i in range(60):
        role = "user" if i % 2 == 0 else "assistant"
        save_message("long-history-session", role, f"Message {i}", db_path)

    history = load_conversation("long-history-session", db_path)

    assert len(history) == 60
    assert [m["content"] for m in history] == [f"Message {i}" for i in range(60)]
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_session_id_with_sql_special_characters_is_stored_literally(db_path):
    malicious_session_id = "abc'); DROP TABLE conversations; --"

    save_message(malicious_session_id, "user", "hello", db_path)

    assert load_conversation(malicious_session_id, db_path) == [
        {"role": "user", "content": "hello"}
    ]
    save_message("unrelated-session", "user", "still works", db_path)
    assert load_conversation("unrelated-session", db_path) == [
        {"role": "user", "content": "still works"}
    ]


def test_ticker_with_sql_special_characters_is_stored_literally(db_path):
    weird_ticker = "aapl'); drop table portfolio; --"

    add_holding("session-a", weird_ticker, 1, 10.0, "2026-01-01", db_path)

    holdings = get_portfolio("session-a", db_path)
    assert len(holdings) == 1
    assert holdings[0]["ticker"] == weird_ticker.upper()
    # Table must still exist and accept further writes.
    add_holding("session-b", "AAPL", 1, 10.0, "2026-01-01", db_path)
    assert len(get_portfolio("session-b", db_path)) == 1


def test_session_id_with_unicode_characters(db_path):
    session_id = "sessione-àèìòù-日本語-🚀"

    save_message(session_id, "user", "ciao", db_path)

    assert load_conversation(session_id, db_path) == [{"role": "user", "content": "ciao"}]


def test_clear_conversation_does_not_affect_portfolio(db_path):
    save_message("session-a", "user", "hi", db_path)
    add_holding("session-a", "AAPL", 10, 150.0, "2026-01-01", db_path)

    clear_conversation("session-a", db_path)

    assert load_conversation("session-a", db_path) == []
    holdings = get_portfolio("session-a", db_path)
    assert len(holdings) == 1
    assert holdings[0]["ticker"] == "AAPL"


def test_remove_holding_does_not_affect_conversation(db_path):
    save_message("session-a", "user", "hi", db_path)
    holding_id = add_holding("session-a", "AAPL", 10, 150.0, "2026-01-01", db_path)

    remove_holding("session-a", holding_id, db_path)

    assert load_conversation("session-a", db_path) == [{"role": "user", "content": "hi"}]


# Portfolio persistence
def test_add_and_get_holding(db_path):
    add_holding("session-a", "aapl", 10, 150.0, "2026-01-15", db_path)

    holdings = get_portfolio("session-a", db_path)

    assert len(holdings) == 1
    # Ticker is normalised to uppercase.
    assert holdings[0]["ticker"] == "AAPL"
    assert holdings[0]["shares"] == 10
    assert holdings[0]["purchase_price"] == 150.0


def test_add_holding_rejects_non_positive_shares(db_path):
    with pytest.raises(ValueError):
        add_holding("session-a", "AAPL", 0, 150.0, "2026-01-15", db_path)
    with pytest.raises(ValueError):
        add_holding("session-a", "AAPL", -5, 150.0, "2026-01-15", db_path)


def test_add_holding_rejects_non_positive_price(db_path):
    with pytest.raises(ValueError):
        add_holding("session-a", "AAPL", 10, 0, "2026-01-15", db_path)
    with pytest.raises(ValueError):
        add_holding("session-a", "AAPL", 10, -1.0, "2026-01-15", db_path)


def test_add_holding_rejects_future_purchase_date(db_path):
    from datetime import date, timedelta

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError):
        add_holding("session-a", "AAPL", 10, 150.0, tomorrow, db_path)


def test_add_holding_accepts_todays_date(db_path):
    from datetime import date

    today = date.today().isoformat()
    # Today is the boundary.
    holding_id = add_holding("session-a", "AAPL", 10, 150.0, today, db_path)
    assert holding_id is not None


def test_add_holding_rejects_malformed_date_string(db_path):
    with pytest.raises(ValueError):
        add_holding("session-a", "AAPL", 10, 150.0, "not-a-date", db_path)


def test_portfolio_handles_many_distinct_holdings(db_path):
    tickers = [f"TICK{i}" for i in range(15)]
    for i, ticker in enumerate(tickers):
        add_holding("many-holdings-session", ticker, i + 1, 10.0 + i, "2026-01-01", db_path)

    holdings = get_portfolio("many-holdings-session", db_path)
    assert len(holdings) == 15
    assert {h["ticker"] for h in holdings} == set(tickers)

    prices = {ticker: 20.0 for ticker in tickers}
    summary = compute_portfolio_summary(holdings, lambda t: prices[t])

    assert len(summary["holdings"]) == 15
    assert summary["unpriced_tickers"] == []
    assert summary["total_cost_basis"] > 0
    assert summary["total_market_value"] > 0


def test_remove_holding_scoped_to_session(db_path):
    holding_id = add_holding("session-a", "AAPL", 10, 150.0, "2026-01-15", db_path)

    deleted = remove_holding("session-b", holding_id, db_path)

    assert deleted is False
    assert len(get_portfolio("session-a", db_path)) == 1


def test_remove_holding_returns_true_and_deletes(db_path):
    holding_id = add_holding("session-a", "AAPL", 10, 150.0, "2026-01-15", db_path)

    deleted = remove_holding("session-a", holding_id, db_path)

    assert deleted is True
    assert get_portfolio("session-a", db_path) == []


def test_remove_holding_returns_false_for_nonexistent_id(db_path):
    assert remove_holding("session-a", 9999, db_path) is False


# Portfolio profit/loss calculation

def test_compute_holding_pnl_gain():
    holding = {"id": 1, "ticker": "AAPL", "shares": 10, "purchase_price": 100.0}
    result = compute_holding_pnl(holding, current_price=120.0)

    assert result["market_value"] == 1200.0
    assert result["cost_basis"] == 1000.0
    assert result["pnl"] == 200.0
    assert result["pnl_pct"] == pytest.approx(20.0)


def test_compute_holding_pnl_loss():
    holding = {"id": 1, "ticker": "TSLA", "shares": 5, "purchase_price": 200.0}
    result = compute_holding_pnl(holding, current_price=150.0)

    assert result["pnl"] == -250.0
    assert result["pnl_pct"] == pytest.approx(-25.0)


def test_compute_holding_pnl_price_unavailable_is_none_not_zero():
    holding = {"id": 1, "ticker": "DELISTED", "shares": 5, "purchase_price": 10.0}
    result = compute_holding_pnl(holding, current_price=None)

    assert result["pnl"] is None
    assert result["pnl_pct"] is None
    assert result["market_value"] is None
    
    assert result["cost_basis"] == 50.0


def test_compute_portfolio_summary_totals_across_holdings():
    holdings = [
        {"id": 1, "ticker": "AAPL", "shares": 10, "purchase_price": 100.0},
        {"id": 2, "ticker": "MSFT", "shares": 4, "purchase_price": 300.0},
    ]
    prices = {"AAPL": 110.0, "MSFT": 280.0}

    summary = compute_portfolio_summary(holdings, lambda t: prices[t])

    assert summary["total_cost_basis"] == 2200.0
    assert summary["total_market_value"] == 2220.0
    assert summary["total_pnl"] == pytest.approx(20.0)
    assert summary["unpriced_tickers"] == []


def test_compute_portfolio_summary_excludes_unpriced_holding_from_totals():
    holdings = [
        {"id": 1, "ticker": "AAPL", "shares": 10, "purchase_price": 100.0},
        {"id": 2, "ticker": "BADTICKER", "shares": 4, "purchase_price": 300.0},
    ]

    def price_lookup(ticker):
        return None if ticker == "BADTICKER" else 110.0

    summary = compute_portfolio_summary(holdings, price_lookup)

    assert summary["total_cost_basis"] == 1000.0
    assert summary["total_market_value"] == 1100.0
    assert summary["unpriced_tickers"] == ["BADTICKER"]
    assert len(summary["holdings"]) == 2


def test_compute_portfolio_summary_looks_up_each_distinct_ticker_once():
    holdings = [
        {"id": 1, "ticker": "AAPL", "shares": 10, "purchase_price": 100.0},
        {"id": 2, "ticker": "AAPL", "shares": 5, "purchase_price": 110.0},
    ]
    calls = []

    def price_lookup(ticker):
        calls.append(ticker)
        return 120.0

    compute_portfolio_summary(holdings, price_lookup)

    assert calls == ["AAPL"]


# Inclusive design: language matching and simplified mode

def _system_message(messages):
    return next(m["content"] for m in messages if m["role"] == "system")


def test_build_prompt_always_includes_language_instruction():
    stock_data = {"ticker": "AAPL", "name": "Apple Inc.", "price": 200.0}
    messages = build_prompt(stock_data, "What is Apple's P/E ratio?")

    assert LANGUAGE_MATCH_INSTRUCTION.strip() in _system_message(messages)


def test_build_prompt_simplified_mode_off_by_default():
    stock_data = {"ticker": "AAPL", "name": "Apple Inc.", "price": 200.0}
    messages = build_prompt(stock_data, "What is Apple's P/E ratio?")

    assert SIMPLIFIED_MODE_INSTRUCTION.strip() not in _system_message(messages)


def test_build_prompt_simplified_mode_on_adds_instruction():
    stock_data = {"ticker": "AAPL", "name": "Apple Inc.", "price": 200.0}
    messages = build_prompt(stock_data, "What is Apple's P/E ratio?", simplified_mode=True)

    assert SIMPLIFIED_MODE_INSTRUCTION.strip() in _system_message(messages)


def test_build_prompt_multi_ticker_also_includes_language_instruction():
    stock_data_list = [
        {"ticker": "AAPL", "name": "Apple Inc.", "price": 200.0},
        {"ticker": "MSFT", "name": "Microsoft Corp.", "price": 300.0},
    ]
    messages = build_prompt(stock_data_list, "Compare Apple and Microsoft")

    assert LANGUAGE_MATCH_INSTRUCTION.strip() in _system_message(messages)
