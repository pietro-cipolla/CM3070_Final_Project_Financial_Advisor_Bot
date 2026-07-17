"""
Iteration 1 unit tests — multi-ticker extraction and query intent classification.

All OpenAI calls are mocked: these are unit tests of our own parsing/decision
logic, not of the LLM itself, and they must run without a real API key or
network access (the marker will not run the code with API keys provided).

Run with:  pytest tests/test_iteration1.py -v
"""

from unittest.mock import patch, MagicMock

from src.rag_pipeline import (
    extract_ticker_from_query,
    extract_tickers_from_query,
    classify_query_intent,
    build_prompt,
)


def _mock_completion(content: str) -> MagicMock:
    """Build a fake OpenAI ChatCompletion response with the given text content."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


# ── extract_tickers_from_query ────────────────────────────────────────────────

def test_extract_tickers_single():
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("AAPL")):
        assert extract_tickers_from_query("What is Apple's P/E ratio?") == ["AAPL"]


def test_extract_tickers_multi():
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("AAPL,MSFT,GOOGL")):
        result = extract_tickers_from_query("Compare Apple, Microsoft and Google")
        assert result == ["AAPL", "MSFT", "GOOGL"]


def test_extract_tickers_none_found():
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("NONE")):
        assert extract_tickers_from_query("What is the weather today?") == []


def test_extract_tickers_filters_stray_none_padding():
    """
    Regression test: observed in manual testing that the model sometimes
    pads a short result with a stray '.NONE' token instead of returning
    only the tickers it actually found (e.g. 'TSLA,F,.NONE' for a 2-company
    query). These artifacts must be filtered out defensively in code,
    since prompt wording alone did not fully prevent them.
    """
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("TSLA,F,.NONE")):
        assert extract_tickers_from_query("Compare Tesla and Ford") == ["TSLA", "F"]


def test_extract_tickers_filters_none_mixed_with_real_tickers():
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("TSLA,F,NONE")):
        assert extract_tickers_from_query("Compare Tesla and Ford") == ["TSLA", "F"]


def test_extract_tickers_corrects_company_name_instead_of_ticker():
    """
    Regression test: observed in manual testing that the model sometimes
    writes the company name in caps instead of the real ticker (e.g. 'FORD'
    instead of 'F'), which then fails at the yfinance lookup step. Known
    cases are corrected via COMMON_TICKER_FIXES.
    """
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("TSLA,FORD")):
        assert extract_tickers_from_query("Compare Tesla and Ford") == ["TSLA", "F"]


def test_extract_tickers_dedup_and_cap_at_three():
    # 5 raw tickers with a duplicate — must dedupe AND cap at MAX_TICKERS (3)
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("AAPL,AAPL,MSFT,GOOGL,TSLA")):
        result = extract_tickers_from_query("Compare Apple, Microsoft, Google and Tesla")
        assert result == ["AAPL", "MSFT", "GOOGL"]
        assert len(result) <= 3


def test_extract_tickers_handles_exception_gracefully():
    with patch("src.rag_pipeline.client.chat.completions.create", side_effect=Exception("network error")):
        assert extract_tickers_from_query("What is Apple's P/E ratio?") == []


def test_extract_ticker_from_query_backward_compat():
    """Single-ticker wrapper should still work and return the first match."""
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("TSLA,F")):
        assert extract_ticker_from_query("Compare Tesla and Ford") == "TSLA"


def test_extract_ticker_from_query_backward_compat_none():
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("NONE")):
        assert extract_ticker_from_query("hello there") is None


# ── classify_query_intent ──────────────────────────────────────────────────────

def test_classify_intent_stock_query():
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("stock_query")):
        assert classify_query_intent("What is Tesla's P/E ratio?") == "stock_query"


def test_classify_intent_open_ended():
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("open_ended")):
        assert classify_query_intent("What should I invest in right now?") == "open_ended"


def test_classify_intent_unclear():
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("unclear")):
        assert classify_query_intent("asdkjaskjd") == "unclear"


def test_classify_intent_invalid_label_defaults_to_stock_query():
    """If the LLM returns something outside the allowed label set, fail safe."""
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("banana")):
        assert classify_query_intent("What is Apple's P/E ratio?") == "stock_query"


def test_classify_intent_handles_exception_gracefully():
    with patch("src.rag_pipeline.client.chat.completions.create", side_effect=Exception("timeout")):
        assert classify_query_intent("What is Apple's P/E ratio?") == "stock_query"


def test_classify_intent_case_insensitive():
    with patch("src.rag_pipeline.client.chat.completions.create", return_value=_mock_completion("STOCK_QUERY")):
        assert classify_query_intent("Tell me about MSFT") == "stock_query"


# ── build_prompt dispatch (single vs. multi-ticker) ───────────────────────────

def test_build_prompt_single_ticker_dict():
    stock_data = {
        "ticker": "AAPL", "name": "Apple Inc.", "price": 200, "change_pct": 1.2,
        "52_week_range": "150 – 220", "pe_ratio": 30, "eps": 6.5, "beta": 1.1,
        "dividend_yield": 0.005, "recommendation": "Buy", "target_price": 230,
        "sector": "Technology", "description": "Consumer electronics.",
        "news_headlines": [], "timestamp": "2026-07-06 10:00:00",
    }
    messages = build_prompt(stock_data, "Should I buy Apple?")
    assert len(messages) == 2
    assert "AAPL" in messages[0]["content"]
    assert "RETRIEVED FINANCIAL DATA" in messages[0]["content"]


def test_build_prompt_multi_ticker_list_uses_comparative_context():
    stock_data_list = [
        {
            "ticker": "AAPL", "name": "Apple Inc.", "price": 200, "change_pct": 1.2,
            "52_week_range": "150 – 220", "pe_ratio": 30, "eps": 6.5, "beta": 1.1,
            "dividend_yield": 0.005, "recommendation": "Buy", "target_price": 230,
            "sector": "Technology", "description": "Consumer electronics.",
            "news_headlines": [], "timestamp": "2026-07-06 10:00:00",
        },
        {
            "ticker": "MSFT", "name": "Microsoft Corp.", "price": 420, "change_pct": -0.5,
            "52_week_range": "300 – 450", "pe_ratio": 35, "eps": 12.0, "beta": 0.9,
            "dividend_yield": 0.007, "recommendation": "Buy", "target_price": 460,
            "sector": "Technology", "description": "Software and cloud.",
            "news_headlines": [], "timestamp": "2026-07-06 10:00:00",
        },
    ]
    messages = build_prompt(stock_data_list, "Compare Apple and Microsoft")
    assert "AAPL" in messages[0]["content"]
    assert "MSFT" in messages[0]["content"]
    assert "COMPARATIVE" in messages[0]["content"]
    assert "compare" in messages[0]["content"].lower()
