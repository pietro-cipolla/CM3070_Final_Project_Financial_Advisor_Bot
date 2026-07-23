"""
Iteration 2 unit tests — NewsAPI integration and price-history charting.

NewsAPI HTTP calls and yfinance calls are mocked: these are unit tests of
our own parsing/formatting/error-handling logic, not of the external
services themselves, and must run without a real API key, real network
access, or a real OpenAI key (the marker will not run the code with API
keys provided).

Run with:  pytest tests/test_iteration2.py -v
"""

import os
from unittest.mock import patch, MagicMock

import pandas as pd

from src.news_data import get_news_for_company, build_news_context
from src.financial_data import get_price_history
from src.rag_pipeline import build_prompt


# ── get_news_for_company ───────────────────────────────────────────────────────

def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def test_get_news_returns_empty_list_without_api_key(monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    assert get_news_for_company("Apple Inc.", "AAPL") == []


def test_get_news_returns_empty_list_without_query(monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "dummy-key")
    assert get_news_for_company("", "") == []


def test_get_news_parses_successful_response(monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "dummy-key")
    fake_json = {
        "status": "ok",
        "articles": [
            {
                "title": "Apple unveils new iPhone",
                "source": {"name": "TechCrunch"},
                "url": "https://example.com/1",
                "publishedAt": "2026-07-14T10:00:00Z",
            },
            {
                "title": "Apple stock rises on earnings beat",
                "source": {"name": "Reuters"},
                "url": "https://example.com/2",
                "publishedAt": "2026-07-13T08:00:00Z",
            },
        ],
    }
    with patch("src.news_data.requests.get", return_value=_mock_response(200, fake_json)):
        result = get_news_for_company("Apple Inc.", "AAPL")

    assert len(result) == 2
    assert result[0]["title"] == "Apple unveils new iPhone"
    assert result[0]["source"] == "TechCrunch"
    assert result[0]["url"] == "https://example.com/1"


def test_get_news_filters_removed_articles(monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "dummy-key")
    fake_json = {
        "status": "ok",
        "articles": [
            {"title": "[Removed]", "source": {"name": "Unknown"}, "url": "", "publishedAt": ""},
            {"title": "Real headline", "source": {"name": "BBC"}, "url": "https://x.com", "publishedAt": "2026-07-14T00:00:00Z"},
        ],
    }
    with patch("src.news_data.requests.get", return_value=_mock_response(200, fake_json)):
        result = get_news_for_company("Ford Motor Company", "F")

    assert len(result) == 1
    assert result[0]["title"] == "Real headline"


def test_get_news_returns_empty_list_on_non_200(monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "dummy-key")
    with patch("src.news_data.requests.get", return_value=_mock_response(429, {"status": "error"})):
        assert get_news_for_company("Apple Inc.", "AAPL") == []


def test_get_news_returns_empty_list_on_request_exception(monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "dummy-key")
    import requests
    with patch("src.news_data.requests.get", side_effect=requests.RequestException("network error")):
        assert get_news_for_company("Apple Inc.", "AAPL") == []


def test_get_news_respects_max_articles(monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "dummy-key")
    fake_json = {
        "status": "ok",
        "articles": [
            {"title": f"Headline {i}", "source": {"name": "Src"}, "url": "https://x.com", "publishedAt": "2026-07-14T00:00:00Z"}
            for i in range(5)
        ],
    }
    with patch("src.news_data.requests.get", return_value=_mock_response(200, fake_json)):
        result = get_news_for_company("Apple Inc.", "AAPL", max_articles=2)
    assert len(result) == 2


# ── build_news_context ──────────────────────────────────────────────────────────

def test_build_news_context_empty_when_no_items():
    assert build_news_context([]) == ""


def test_build_news_context_formats_items():
    items = [{"title": "Test headline", "source": "TechCrunch", "url": "https://x.com", "published_at": "2026-07-14T10:00:00Z"}]
    context = build_news_context(items)
    assert "Test headline" in context
    assert "TechCrunch" in context
    assert "2026-07-14" in context


# ── get_price_history ────────────────────────────────────────────────────────

def _mock_ticker_with_history(history_df):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = history_df
    return mock_ticker


def test_get_price_history_adds_ma20_column():
    df = pd.DataFrame({"Close": [10, 11, 12, 13, 14]})
    with patch("src.financial_data.yf.Ticker", return_value=_mock_ticker_with_history(df)):
        result = get_price_history("AAPL")
    assert result is not None
    assert "MA20" in result.columns


def test_get_price_history_returns_none_on_empty_history():
    df = pd.DataFrame()
    with patch("src.financial_data.yf.Ticker", return_value=_mock_ticker_with_history(df)):
        assert get_price_history("INVALIDTICKER") is None


def test_get_price_history_returns_none_on_exception():
    with patch("src.financial_data.yf.Ticker", side_effect=Exception("network error")):
        assert get_price_history("AAPL") is None


# ── build_prompt with news_context ──────────────────────────────────────────────

def test_build_prompt_appends_news_context_when_provided():
    stock_data = {
        "ticker": "AAPL", "name": "Apple Inc.", "price": 200, "change_pct": 1.2,
        "52_week_range": "150 – 220", "pe_ratio": 30, "eps": 6.5, "beta": 1.1,
        "dividend_yield": 0.005, "recommendation": "Buy", "target_price": 230,
        "sector": "Technology", "description": "Consumer electronics.",
        "news_headlines": [], "timestamp": "2026-07-15 10:00:00",
    }
    news_context = "\nRecent news (via NewsAPI):\n  - Apple unveils new iPhone (TechCrunch, 2026-07-14)\n"
    messages = build_prompt(stock_data, "Should I buy Apple?", news_context=news_context)
    assert "Apple unveils new iPhone" in messages[0]["content"]


def test_build_prompt_backward_compatible_without_news_context():
    """Iteration 1 callers that don't pass news_context must still work unchanged."""
    stock_data = {
        "ticker": "AAPL", "name": "Apple Inc.", "price": 200, "change_pct": 1.2,
        "52_week_range": "150 – 220", "pe_ratio": 30, "eps": 6.5, "beta": 1.1,
        "dividend_yield": 0.005, "recommendation": "Buy", "target_price": 230,
        "sector": "Technology", "description": "Consumer electronics.",
        "news_headlines": [], "timestamp": "2026-07-15 10:00:00",
    }
    messages = build_prompt(stock_data, "Should I buy Apple?")
    assert len(messages) == 2
    assert "AAPL" in messages[0]["content"]
