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

import sqlite3
from unittest.mock import patch, MagicMock

import pytest

from src.rag_pipeline import (
    extract_tickers_from_query,
    classify_query_intent,
    build_prompt,
    _history_messages,
    _extract_all_tickers,
    _naive_ticker_candidates,
    MAX_HISTORY_MESSAGES,
)
from src.financial_data import get_stock_summary, resolve_ticker
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
        result = extract_tickers_from_query("What do you think about Intesa Sanpaolo?")
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
        extract_tickers_from_query("What do you think about Intesa Sanpaolo?")
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


# Problema 37 — riconoscimento ticker generalizzato via yfinance.Search()

def test_naive_ticker_candidates_matches_suffixed_symbol():
    """
    Problema 37: confirmed case — the user already typed a complete,
    correctly-suffixed ticker ("PST.MI" for Poste Italiane) verbatim, and
    the LLM extraction call still returned nothing. The local, no-network
    fallback must catch this exact shape.
    """
    assert _naive_ticker_candidates("What about PST.MI?") == ["PST.MI"]
    assert _naive_ticker_candidates("Tell me about pst.mi") == ["PST.MI"]


def test_naive_ticker_candidates_ignores_plain_words_and_bare_tickers():
    """
    The fallback must stay narrow: no dot-suffix present means it does not
    fire, so ordinary acronyms in a sentence (or a bare US ticker with no
    suffix, already handled correctly by the LLM) are not falsely matched.
    """
    assert _naive_ticker_candidates("What is the CEO's view on this?") == []
    assert _naive_ticker_candidates("Tell me about AAPL") == []
    assert _naive_ticker_candidates("Is now a good time to invest?") == []


def test_extract_all_tickers_falls_back_to_naive_candidates_when_llm_finds_nothing():
    """
    Problema 37: when the LLM extraction call returns NONE, a query
    containing an already-well-formed suffixed ticker must still resolve
    to that ticker via the local fallback, instead of surfacing the "could
    not identify a stock ticker" message the user actually hit.
    """
    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("NONE"),
    ):
        result = _extract_all_tickers("What about PST.MI?")
    assert result == ["PST.MI"]


def test_extract_all_tickers_does_not_use_naive_fallback_when_llm_succeeds():
    """The naive fallback must only be consulted when the LLM found
    nothing — it must never override or supplement a successful LLM
    extraction, even if the query also happens to contain a suffixed-
    looking token elsewhere."""
    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("AAPL"),
    ):
        result = _extract_all_tickers("Compare Apple to PST.MI")
    assert result == ["AAPL"]


def _mock_search_with_quotes(quotes: list[dict]) -> MagicMock:
    mock_search = MagicMock()
    mock_search.quotes = quotes
    return mock_search


def test_resolve_ticker_uses_first_search_result_symbol():
    """resolve_ticker() should return the symbol from yfinance's own search
    index rather than requiring a hardcoded name->ticker table."""
    with patch(
        "src.financial_data.yf.Search",
        return_value=_mock_search_with_quotes([{"symbol": "pst.mi"}, {"symbol": "OTHER"}]),
    ) as mock_search:
        result = resolve_ticker("Poste Italiane")
    mock_search.assert_called_once_with("Poste Italiane", max_results=5)
    assert result == "PST.MI"


def test_resolve_ticker_falls_back_to_candidate_when_search_finds_nothing():
    """No usable quote (empty result or search failure) must not raise —
    it must fall back to the uppercased original candidate so callers can
    still try it as-is."""
    with patch("src.financial_data.yf.Search", return_value=_mock_search_with_quotes([])):
        assert resolve_ticker("nonexistent") == "NONEXISTENT"
    with patch("src.financial_data.yf.Search", side_effect=Exception("network error")):
        assert resolve_ticker("pst.mi") == "PST.MI"


def test_get_stock_summary_retries_via_resolve_ticker_when_direct_lookup_fails():
    """
    Problema 37, end-to-end at the financial_data layer: if the symbol as
    given returns no price data, get_stock_summary() must retry once with
    the symbol resolved via resolve_ticker() before giving up, and the
    successful result must report the RESOLVED symbol (not the original
    unresolved guess) as its "ticker" field.
    """
    empty_info_ticker = _mock_ticker_with_info({})
    resolved_ticker_mock = _mock_ticker_with_info(_wbd_style_info(longName="Poste Italiane"))

    def _fake_ticker(symbol):
        return resolved_ticker_mock if symbol == "PST.MI" else empty_info_ticker

    with patch("src.financial_data.yf.Ticker", side_effect=_fake_ticker), patch(
        "src.financial_data.yf.Search",
        return_value=_mock_search_with_quotes([{"symbol": "PST.MI"}]),
    ):
        result = get_stock_summary("PST")

    assert result["ticker"] == "PST.MI"
    assert result["name"] == "Poste Italiane"
    assert "error" not in result


def test_get_stock_summary_still_errors_when_resolve_ticker_finds_nothing_either():
    """A genuinely invalid ticker (search also finds nothing) must still
    produce the original {'error': ...} shape, unchanged from before this
    fix."""
    empty_info_ticker = _mock_ticker_with_info({})
    with patch("src.financial_data.yf.Ticker", return_value=empty_info_ticker), patch(
        "src.financial_data.yf.Search", return_value=_mock_search_with_quotes([])
    ):
        result = get_stock_summary("NOTAREALTICKER")
    assert "error" in result


# Problema 38 — ticker sbagliato-E-inesistente in un confronto multi-azienda
# ("can you compare Poste Italiane and Nvidia?" ha estratto "PT.MI" invece di
# "PST.MI" — Problema 37's ticker-based retry cannot recover this, since it
# searches for the wrong symbol text itself, not the company name).

def test_extract_all_tickers_with_names_parses_ticker_name_pairs():
    """The extraction prompt now asks for TICKER:Company Name pairs so a
    wrong ticker guess can later be corrected by searching for the company
    name instead of the (wrong) symbol. This is the parsing half of that
    change."""
    from src.rag_pipeline import _extract_all_tickers_with_names

    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("PT.MI:Poste Italiane,NVDA:NVIDIA Corporation"),
    ):
        result = _extract_all_tickers_with_names("can you compare Poste Italiane and Nvidia?")
    assert result == [
        {"ticker": "PT.MI", "name": "Poste Italiane"},
        {"ticker": "NVDA", "name": "NVIDIA Corporation"},
    ]


def test_extract_all_tickers_with_names_tolerates_old_style_bare_tickers():
    """Backward compatibility: a reply with no ':' at all (the pre-Problema
    38 format, and what every pre-existing mocked test in this file and in
    test_iteration1.py still simulates) must still parse, as a ticker with
    no name attached, not as a parsing failure."""
    from src.rag_pipeline import _extract_all_tickers_with_names

    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("TSLA,F"),
    ):
        result = _extract_all_tickers_with_names("Compare Tesla and Ford")
    assert result == [{"ticker": "TSLA", "name": None}, {"ticker": "F", "name": None}]


def test_extract_all_tickers_unaffected_by_name_pairing():
    """_extract_all_tickers() (the pre-existing, still-used entry point)
    must keep returning plain ticker strings, unchanged, regardless of
    whether the mocked reply includes names — every pre-existing test in
    this file and in test_iteration1.py relies on this exact contract."""
    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("PT.MI:Poste Italiane,NVDA:NVIDIA Corporation"),
    ):
        result = _extract_all_tickers("can you compare Poste Italiane and Nvidia?")
    assert result == ["PT.MI", "NVDA"]


def test_extract_ticker_candidates_reports_truncation_with_names():
    """extract_ticker_candidates() must apply the same MAX_TICKERS
    truncation semantics as extract_tickers_with_truncation_info(), while
    also carrying the paired name for each surviving ticker."""
    from src.rag_pipeline import extract_ticker_candidates

    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion(
            "AAPL:Apple Inc,MSFT:Microsoft Corporation,GOOGL:Alphabet Inc,AMZN:Amazon.com Inc"
        ),
    ):
        pairs, truncated = extract_ticker_candidates("Compare Apple, Microsoft, Google and Amazon")
    assert truncated is True
    assert pairs == [
        {"ticker": "AAPL", "name": "Apple Inc"},
        {"ticker": "MSFT", "name": "Microsoft Corporation"},
        {"ticker": "GOOGL", "name": "Alphabet Inc"},
    ]


def test_get_stock_summary_recovers_via_name_when_ticker_guess_does_not_exist():
    """
    Problema 38, end-to-end at the financial_data layer: the confirmed
    failure — the LLM guesses "PT.MI" for Poste Italiane (wrong AND
    nonexistent), and Problema 37's ticker-based resolve_ticker("PT.MI")
    retry also finds nothing usable, because it searches for the wrong
    string. Passing expected_name="Poste Italiane" adds a THIRD attempt,
    resolve_ticker("Poste Italiane"), which must find the real "PST.MI"
    and succeed.
    """
    empty_info_ticker = _mock_ticker_with_info({})
    resolved_ticker_mock = _mock_ticker_with_info(_wbd_style_info(longName="Poste Italiane"))

    def _fake_ticker(symbol):
        return resolved_ticker_mock if symbol == "PST.MI" else empty_info_ticker

    def _fake_search(candidate, max_results=5):
        # resolve_ticker(ticker) is tried first with the wrong guess
        # "PT.MI" and must find nothing usable; only resolve_ticker
        # (expected_name) with "Poste Italiane" finds the real symbol.
        if candidate == "Poste Italiane":
            return _mock_search_with_quotes([{"symbol": "PST.MI"}])
        return _mock_search_with_quotes([])

    with patch("src.financial_data.yf.Ticker", side_effect=_fake_ticker), patch(
        "src.financial_data.yf.Search", side_effect=_fake_search
    ):
        result = get_stock_summary("PT.MI", expected_name="Poste Italiane")

    assert result["ticker"] == "PST.MI"
    assert result["name"] == "Poste Italiane"
    assert "error" not in result


def test_get_stock_summary_still_errors_when_name_fallback_also_finds_nothing():
    """If neither the ticker guess nor the company name resolve to any
    priced instrument, the original {'error': ...} shape is still
    returned, unchanged."""
    empty_info_ticker = _mock_ticker_with_info({})
    with patch("src.financial_data.yf.Ticker", return_value=empty_info_ticker), patch(
        "src.financial_data.yf.Search", return_value=_mock_search_with_quotes([])
    ):
        result = get_stock_summary("PT.MI", expected_name="Poste Italiane")
    assert "error" in result


def test_get_stock_summary_does_not_auto_correct_a_ticker_that_already_resolves():
    """
    UPDATED by Problema 39 (1 settembre 2026) — this test originally
    asserted that a ticker resolving on the first attempt skipped the name
    check AND the extra yf.Search() call entirely (zero added cost, zero
    revalidation, by deliberate design at the time). That trade-off was
    revisited after a live, confirmed case ("PST" used for Poste Italiane
    instead of "PST.MI", itself a real ETF, so the direct lookup "worked"
    and hid a wrong company with no warning at all) — see
    _ticker_name_mismatch_warning() in financial_data.py.

    What is STILL true, and still asserted here: the ticker and data
    returned are never silently changed just because expected_name doesn't
    textually match the real company name (e.g. "Google" vs. the actual
    longName "Alphabet Inc." — a legitimate brand/legal-name mismatch, see
    COMMON_TICKER_FIXES, Problema 12). Only the "no extra Search() call at
    all" half of the original guarantee changed — see
    test_get_stock_summary_no_warning_when_direct_success_ticker_matches_name_search
    for the dedicated regression test confirming this exact Google case
    still raises no warning either (the name search resolves back to the
    same "GOOGL" symbol).
    """
    google_style_info = _wbd_style_info(
        longName="Alphabet Inc.", trailingEps=6.5, trailingPE=25.0
    )
    with patch(
        "src.financial_data.yf.Ticker", return_value=_mock_ticker_with_info(google_style_info)
    ), patch(
        "src.financial_data.yf.Search",
        return_value=_mock_search_with_quotes([{"symbol": "GOOGL"}]),
    ):
        result = get_stock_summary("GOOGL", expected_name="Google")
    assert "error" not in result
    assert result["ticker"] == "GOOGL"
    assert result["name"] == "Alphabet Inc."


def test_get_multiple_stock_summaries_corrects_independently_per_ticker():
    """
    Problema 38, comparative path: get_multiple_stock_summaries() must
    forward each ticker's OWN name hint independently — a wrong guess for
    one company in a comparison must not affect, or borrow the name of,
    another company in the same request.
    """
    from src.financial_data import get_multiple_stock_summaries

    poste_mock = _mock_ticker_with_info(_wbd_style_info(longName="Poste Italiane"))
    nvda_mock = _mock_ticker_with_info(_wbd_style_info(longName="NVIDIA Corporation"))
    empty_info_ticker = _mock_ticker_with_info({})

    def _fake_ticker(symbol):
        if symbol == "PST.MI":
            return poste_mock
        if symbol == "NVDA":
            return nvda_mock
        return empty_info_ticker

    def _fake_search(candidate, max_results=5):
        if candidate == "Poste Italiane":
            return _mock_search_with_quotes([{"symbol": "PST.MI"}])
        return _mock_search_with_quotes([])

    with patch("src.financial_data.yf.Ticker", side_effect=_fake_ticker), patch(
        "src.financial_data.yf.Search", side_effect=_fake_search
    ):
        results = get_multiple_stock_summaries(
            ["PT.MI", "NVDA"],
            expected_names={"PT.MI": "Poste Italiane"},
        )

    assert results[0]["ticker"] == "PST.MI"
    assert results[0]["name"] == "Poste Italiane"
    assert results[1]["ticker"] == "NVDA"
    assert results[1]["name"] == "NVIDIA Corporation"


# Problema 38 continued — manual re-testing after the fix above surfaced two
# more extraction bugs on companies never exercised before: a purely
# numeric non-US ticker (Toyota, Tokyo exchange), and a company whose own
# legal name contains a comma (Block, Inc.).

def test_naive_ticker_candidates_matches_numeric_prefixed_ticker():
    """
    Toyota's real Yahoo Finance ticker on its home exchange is "7203.T" —
    a purely numeric local code, unlike most US/European tickers. The
    Problema 37 fallback regex originally required a letters-only prefix
    and would not recognize this even if the user typed it verbatim.
    """
    assert _naive_ticker_candidates("What about 7203.T?") == ["7203.T"]
    assert _naive_ticker_candidates("tell me about 7203.t") == ["7203.T"]
    # Still narrow: a bare number with no exchange suffix, or ordinary text
    # with numbers in it, must not match.
    assert _naive_ticker_candidates("It grew by 12.5% this year") == []
    assert _naive_ticker_candidates("Is 7 a lucky number?") == []


def test_extract_all_tickers_with_names_keeps_numeric_ticker():
    """
    Problema 38 continued: "compare ASML, Nestle and Toyota" extracted
    Toyota's ticker as "7203.T", which the original isalpha() filter
    silently discarded (first character is a digit) — Toyota never even
    reached financial_data, unlike a wrong-but-alphabetic guess. Widening
    the check to isalnum() must let it through, for both the new
    TICKER:Name format and the legacy bare-ticker format.
    """
    from src.rag_pipeline import _extract_all_tickers_with_names

    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion(
            "ASML:ASML Holding N.V.,NESN.SW:Nestle S.A.,7203.T:Toyota Motor Corporation"
        ),
    ):
        result = _extract_all_tickers_with_names("compare ASML, Nestle and Toyota")
    assert [p["ticker"] for p in result] == ["ASML", "NESN.SW", "7203.T"]
    assert result[2]["name"] == "Toyota Motor Corporation"

    # Legacy bare-ticker format (no ':' anywhere) must also keep it.
    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("ASML,NESN.SW,7203.T"),
    ):
        legacy_result = _extract_all_tickers_with_names("compare ASML, Nestle and Toyota")
    assert [p["ticker"] for p in legacy_result] == ["ASML", "NESN.SW", "7203.T"]


def test_extract_all_tickers_with_names_survives_comma_in_company_name():
    """
    Confirmed failure: "what can you tell me about Block?" extracted the
    pair "SQ:Block, Inc." — but Block's own legal name contains a comma
    before its "Inc." suffix, and the original naive comma-split cut the
    reply into ["SQ:Block", "Inc."], fabricating a second, bogus ticker
    ("INC.") out of half of the company's own name. The anchored-regex
    parser must keep the whole name intact as ONE entry.
    """
    from src.rag_pipeline import _extract_all_tickers_with_names

    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("SQ:Block, Inc."),
    ):
        result = _extract_all_tickers_with_names("what can you tell me about Block?")
    assert result == [{"ticker": "SQ", "name": "Block, Inc."}]


def test_extract_all_tickers_with_names_comma_in_name_does_not_swallow_next_entry():
    """Same comma-in-legal-name shape as above, but with a SECOND company
    right after it — the entry boundary must still be found correctly
    despite the comma inside the first company's own name."""
    from src.rag_pipeline import _extract_all_tickers_with_names

    with patch(
        "src.rag_pipeline.client.chat.completions.create",
        return_value=_mock_completion("SQ:Block, Inc.,NVDA:NVIDIA Corporation"),
    ):
        result = _extract_all_tickers_with_names("compare Block and Nvidia")
    assert result == [
        {"ticker": "SQ", "name": "Block, Inc."},
        {"ticker": "NVDA", "name": "NVIDIA Corporation"},
    ]


# Problema 38 continued (again) — the comma-in-name fix above correctly
# reduced "what can you tell me about Block?" to the single pair {"ticker": "SQ",
# "name": "Block, Inc."}, but Block's data still came back wrong: Bristol-
# Myers Squibb (BMY) instead of Block. Root cause confirmed against
# get_stock_summary(): Block changed ticker from "SQ" to "XYZ" in January
# 2025, so yf.Ticker("SQ") no longer has price data; the ticker-based
# fallback resolve_ticker("SQ") then fuzzy-matches "BMY" via yf.Search()'s
# free-text index (apparently because "Squibb" contains "SQ"), and — under
# the original ordering — that "success" was accepted before the
# name-based fallback (which would have found "XYZ") ever got a chance to
# run. Fix: try the name-based candidate BEFORE the ticker-based one once
# the direct lookup has failed.

def test_get_stock_summary_prefers_name_over_a_misleading_ticker_fallback_match():
    """
    Reproduces the confirmed live failure: "SQ" (Block's old ticker) has no
    price data any more, and resolve_ticker("SQ") itself resolves to an
    unrelated but real, priced instrument (BMY) via fuzzy text matching --
    a false positive from the ticker-based fallback, not a missing-data
    case. Because the name-based fallback is now tried first, it must find
    the real current symbol ("XYZ") and use IT, never touching BMY's data.
    """
    empty_info_ticker = _mock_ticker_with_info({})
    bmy_mock = _mock_ticker_with_info(_wbd_style_info(longName="Bristol-Myers Squibb Company"))
    xyz_mock = _mock_ticker_with_info(_wbd_style_info(longName="Block, Inc."))

    def _fake_ticker(symbol):
        if symbol == "BMY":
            return bmy_mock
        if symbol == "XYZ":
            return xyz_mock
        return empty_info_ticker  # "SQ" itself: no longer has price data

    def _fake_search(candidate, max_results=5):
        if candidate == "SQ":
            # The confirmed false-positive fuzzy match.
            return _mock_search_with_quotes([{"symbol": "BMY"}])
        if candidate == "Block, Inc.":
            return _mock_search_with_quotes([{"symbol": "XYZ"}])
        return _mock_search_with_quotes([])

    with patch("src.financial_data.yf.Ticker", side_effect=_fake_ticker), patch(
        "src.financial_data.yf.Search", side_effect=_fake_search
    ):
        result = get_stock_summary("SQ", expected_name="Block, Inc.")

    assert result["ticker"] == "XYZ"
    assert result["name"] == "Block, Inc."
    assert "error" not in result


def test_get_stock_summary_falls_back_to_ticker_match_when_name_search_finds_nothing():
    """
    If expected_name is given but genuinely resolves to nothing usable
    (no such company in yfinance's index at all), the ticker-based
    fallback must still be tried as a second-line rescue -- the reordering
    must not simply drop the old fallback, only move it after the name
    attempt.
    """
    empty_info_ticker = _mock_ticker_with_info({})
    pst_mock = _mock_ticker_with_info(_wbd_style_info(longName="Poste Italiane"))

    def _fake_ticker(symbol):
        return pst_mock if symbol == "PST.MI" else empty_info_ticker

    def _fake_search(candidate, max_results=5):
        if candidate == "PT.MI":
            return _mock_search_with_quotes([{"symbol": "PST.MI"}])
        return _mock_search_with_quotes([])  # the name hint finds nothing

    with patch("src.financial_data.yf.Ticker", side_effect=_fake_ticker), patch(
        "src.financial_data.yf.Search", side_effect=_fake_search
    ):
        result = get_stock_summary("PT.MI", expected_name="A Made Up Name")

    assert result["ticker"] == "PST.MI"
    assert "error" not in result


# Problema 39 (1 settembre 2026, confirmed live) — a ticker that succeeds
# on its own DIRECT lookup (no fallback involved at all) can still be the
# wrong company: confirmed live case, a differently-phrased query made the
# extractor guess bare "PST" for Poste Italiane, which is itself a real,
# priced US ETF (ProShares UltraShort 7-10 Year Treasury) -- so the direct
# lookup succeeded immediately and neither Problema 37/38 fallback above
# ever ran. Fix (Option 3, chosen explicitly by the user over auto-
# correcting or a name-string comparison): cross-check the ticker actually
# used against resolve_ticker(expected_name) and attach a WARNING only,
# never silently swap the ticker.

def test_get_stock_summary_warns_when_direct_success_ticker_disagrees_with_name_search():
    """
    Reproduces the confirmed live failure: "PST" resolves directly (it is a
    real ETF), so no fallback runs, but a search on the company name alone
    ("Poste Italiane") points to a different symbol ("PST.MI"). The wrong
    ticker's data must still be returned as before (never auto-corrected --
    deliberately not implemented, see _ticker_name_mismatch_warning
    docstring), but the result must now carry an explicit warning.
    """
    pst_etf_mock = _mock_ticker_with_info(_wbd_style_info(longName="ProShares UltraShort 7-10 Year Treasury"))

    def _fake_search(candidate, max_results=5):
        assert candidate == "Poste Italiane"
        return _mock_search_with_quotes([{"symbol": "PST.MI"}])

    with patch("src.financial_data.yf.Ticker", return_value=pst_etf_mock), patch(
        "src.financial_data.yf.Search", side_effect=_fake_search
    ):
        result = get_stock_summary("PST", expected_name="Poste Italiane")

    assert "error" not in result
    assert result["ticker"] == "PST"  # unchanged -- warning only, no auto-correction
    assert result["name"] == "ProShares UltraShort 7-10 Year Treasury"
    assert "ticker_mismatch_warning" in result
    assert "PST" in result["ticker_mismatch_warning"]
    assert "PST.MI" in result["ticker_mismatch_warning"]


def test_get_stock_summary_no_warning_when_direct_success_ticker_matches_name_search():
    """
    Regression guard for the exact false-positive risk this design was
    chosen to avoid: a legitimate brand/legal-name mismatch (Google's
    common name vs. "Alphabet Inc.", already relied upon via
    COMMON_TICKER_FIXES) must NOT raise a warning, because comparing
    tickers (not name strings) means the name-based search independently
    resolves back to the same symbol.
    """
    googl_mock = _mock_ticker_with_info(_wbd_style_info(longName="Alphabet Inc."))

    def _fake_search(candidate, max_results=5):
        assert candidate == "Google"
        return _mock_search_with_quotes([{"symbol": "GOOGL"}])

    with patch("src.financial_data.yf.Ticker", return_value=googl_mock), patch(
        "src.financial_data.yf.Search", side_effect=_fake_search
    ):
        result = get_stock_summary("GOOGL", expected_name="Google")

    assert "error" not in result
    assert "ticker_mismatch_warning" not in result


def test_get_stock_summary_skips_mismatch_check_without_an_expected_name():
    """
    No name hint at all (expected_name=None, the pre-Problema-38 default
    used by get_current_price()/get_closing_prices()) must skip the check
    entirely -- no extra yf.Search() call, no warning key -- so callers
    that never had a name hint pay zero added cost from this fix."""
    aapl_mock = _mock_ticker_with_info(_wbd_style_info(longName="Apple Inc."))

    with patch("src.financial_data.yf.Ticker", return_value=aapl_mock), patch(
        "src.financial_data.yf.Search"
    ) as mock_search:
        result = get_stock_summary("AAPL")

    assert "error" not in result
    assert "ticker_mismatch_warning" not in result
    mock_search.assert_not_called()


def test_get_stock_summary_skips_mismatch_check_when_result_came_from_a_fallback():
    """
    The cross-check must fire ONLY on a direct, first-try success -- a
    ticker that only succeeded via the Problema 37/38 fallback chain was
    already resolved by (or despite) the name hint, so re-running the same
    search again here would be redundant, not a genuine independent check.
    Reuses the confirmed SQ -> XYZ (Block) scenario, which succeeds via the
    name-based fallback, not directly.
    """
    empty_info_ticker = _mock_ticker_with_info({})
    xyz_mock = _mock_ticker_with_info(_wbd_style_info(longName="Block, Inc."))

    def _fake_ticker(symbol):
        return xyz_mock if symbol == "XYZ" else empty_info_ticker

    def _fake_search(candidate, max_results=5):
        if candidate == "Block, Inc.":
            return _mock_search_with_quotes([{"symbol": "XYZ"}])
        return _mock_search_with_quotes([])

    with patch("src.financial_data.yf.Ticker", side_effect=_fake_ticker), patch(
        "src.financial_data.yf.Search", side_effect=_fake_search
    ):
        result = get_stock_summary("SQ", expected_name="Block, Inc.")

    assert "error" not in result
    assert result["ticker"] == "XYZ"
    assert "ticker_mismatch_warning" not in result


# Problema 34 — cronologia perde grafico/backtest/notizie dopo riavvio (o
# dopo qualunque rerun nella stessa sessione)

def test_save_and_load_message_round_trips_attachments(db_path):
    """
    Problema 34: a message's rich content (stock data, backtest, news,
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
    A message saved without attachments must come back in EXACTLY the old
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
    save_message(session_id, role, content, db_path) keep working — this
    guards against a future refactor accidentally reordering the params.
    """
    save_message("session-a", "user", "positional db_path still works", db_path)
    history = load_conversation("session-a", db_path)
    assert history == [{"role": "user", "content": "positional db_path still works"}]


def test_json_default_uses_item_method_for_numpy_like_scalars():
    """
    _json_default is the json.dumps() fallback used when persisting
    attachments (the GA backtester can surface np.int64/np.float64 inside a
    result dict). Rather than depending on numpy in this test, a minimal
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
    A DB created before Problema 34's fix (schema without the attachments
    column) must still work once the app is upgraded — CREATE TABLE IF NOT
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
