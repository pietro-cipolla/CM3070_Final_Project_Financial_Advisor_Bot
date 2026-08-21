"""
tests/test_iteration4_section4_fixes.py
Automated tests for the fix block that followed Iteration 4's Sezione 4
end-to-end manual testing (20 agosto 2026, Problemi 26-34 — see the Diario
Tecnico in the planning document for the full write-up of each).

Kept as a separate file from test_iteration1.py/test_iteration2.py/etc.
because these fixes span multiple modules and were all found by the same
end-to-end testing pass, rather than belonging to one iteration's original
feature set — mirrors how test_optimizer.py was kept separate from
test_iteration4.py for the same kind of reason.

One or two tests per problem are added here as each fix is implemented and
packaged as its own dated commit (commit12 onward); this file grows
incrementally across that block of commits, the same way test_iteration1.py
grew across Iteration 1's own fix commits.
"""

from unittest.mock import patch, MagicMock

from src.rag_pipeline import (
    extract_tickers_from_query,
    classify_query_intent,
    build_prompt,
    _history_messages,
    MAX_HISTORY_MESSAGES,
)
from src.financial_data import get_stock_summary


def _mock_completion(content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


# Problema 27 — conversation history support (classify_query_intent,
# extract_tickers_from_query/_extract_all_tickers, build_prompt)

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
    Problema 27: a follow-up query with no company of its own (e.g. "How
    does it compare to its main rival in GPUs?" right after a Nvidia
    question) must have the prior turns forwarded into the OpenAI call, so
    the model has a chance to resolve "it" — this test asserts the history
    is actually sent, not that the (mocked) model resolves it correctly,
    since that reasoning happens inside the real LLM, not in this code.
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
    # History must come between the system message and the current query.
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[-1] == {
        "role": "user",
        "content": "How does it compare to its main rival in GPUs?",
    }


def test_extract_tickers_without_history_unchanged():
    """No history passed (the pre-Problema-27 call shape) -> unaffected."""
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


# Problema 28 — P/E ratio incoerente quando l'EPS è negativo

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
    Problema 28: yfinance's trailingPE (583.38) and trailingEps (-1.28) were
    observed mutually inconsistent for WBD — a positive P/E is mathematically
    impossible with negative earnings. Once EPS is negative, pe_ratio must
    not be passed through as if it were a trustworthy number.
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


# Problema 30 — ticker non-USA risolti senza suffisso di borsa

def test_extract_tickers_preserves_exchange_suffix():
    """
    Problema 30: a bare non-US symbol (e.g. "ISP" for Intesa Sanpaolo) was
    observed resolving to a completely unrelated company on Yahoo Finance
    (ING Groep NV). Once the model returns a suffixed symbol (e.g.
    "ISP.MI"), the extraction pipeline must preserve it rather than
    stripping anything after the "." (the "." must survive the existing
    comma-split/alpha-first-char filtering unchanged).
    """
    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("ISP.MI"),
    ):
        result = extract_tickers_from_query("Cosa ne pensi di Intesa Sanpaolo?")
    assert result == ["ISP.MI"]


def test_extract_tickers_prompt_instructs_exchange_suffixes():
    """The fix itself: the extraction prompt must tell the model to include
    the correct Yahoo Finance exchange suffix for non-US companies, with a
    concrete example, rather than leaving suffix handling to the model's
    unguided judgement (which worked for Pirelli but not Intesa Sanpaolo —
    see Diario Tecnico, Problema 30)."""
    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("ISP.MI"),
    ) as mock_create:
        extract_tickers_from_query("Cosa ne pensi di Intesa Sanpaolo?")
    system_content = mock_create.call_args.kwargs["messages"][0]["content"]
    assert "exchange suffix" in system_content
    assert "ISP.MI" in system_content


# Problema 32 — notizie assenti per suffissi legali esteri non riconosciuti

def test_search_phrase_strips_foreign_legal_suffixes():
    """
    Problema 32: _SUFFIX_RE only recognized English legal suffixes, so a
    yfinance longName like "Pirelli & C. S.p.A." was passed to NewsAPI
    whole (no real headline matches that verbatim), returning zero news
    results even though the ticker/financial data were correct. Both
    trailing clauses must now be stripped, applied in a loop.
    """
    from src.news_data import _search_phrase

    assert _search_phrase("Pirelli & C. S.p.A.") == "Pirelli"
    assert _search_phrase("Intesa Sanpaolo S.p.A.") == "Intesa Sanpaolo"
    assert _search_phrase("Volkswagen AG") == "Volkswagen"
    assert _search_phrase("Siemens Healthineers AG") == "Siemens Healthineers"
    assert _search_phrase("L'Oreal SA") == "L'Oreal"
    # English suffixes (pre-existing behavior) must still work unchanged.
    assert _search_phrase("Ford Motor Company") == "Ford"
    assert _search_phrase("Apple Inc.") == "Apple"
