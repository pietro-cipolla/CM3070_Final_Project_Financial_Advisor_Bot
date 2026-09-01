"""
financial_data.py
Data Retrieval layer — fetches stock data from Yahoo Finance via yfinance.

Iteration 1: adds multi-ticker retrieval and a comparative context block
so the RAG pipeline can answer questions that mention more than one company.
"""

import yfinance as yf
from datetime import datetime

MAX_TICKERS = 3


def resolve_ticker(candidate: str) -> str:
    """
    Iteration 4, Sezione 4 (Problema 37): resolve a company name or an
    imperfect/partial ticker guess to a real Yahoo Finance symbol using
    yfinance's own search index, instead of depending on the LLM extractor
    (rag_pipeline.py) to recall every non-US exchange suffix from memory —
    the previous approach (a short hardcoded example list in the extraction
    prompt, plus COMMON_TICKER_FIXES) only ever covered the handful of
    companies someone thought to add as examples.

    yf.Search() is the same fuzzy company/ticker search that powers the
    Yahoo Finance website's own search box, so it returns the correctly
    exchange-suffixed symbol for any global market without a manually
    maintained table. This is used here as a FALLBACK/RESOLUTION step, not
    a replacement for the LLM: the LLM is still what turns natural language
    into a candidate company name or ticker guess (a task it already does
    well); yf.Search() is only responsible for turning that candidate into
    a real, correctly-suffixed symbol.

    Returns the resolved uppercase symbol from the first search result, or
    the uppercased candidate UNCHANGED if the search fails or returns no
    usable quote (network error, no match, or a candidate that is already
    a correct symbol yf.Search does not happen to return first) — this
    function never raises and never returns an empty string, so callers
    can always fall back to trying the original candidate as-is.
    """
    try:
        search = yf.Search(candidate, max_results=5)
        for quote in search.quotes or []:
            symbol = quote.get("symbol")
            if symbol:
                return symbol.upper()
    except Exception:
        pass
    return candidate.upper()


def _has_price_data(info: dict | None) -> bool:
    """Shared check for whether a yfinance `.info` dict represents a real,
    priced instrument. Factored out so the Problema 38 name-based retry
    below can reuse the exact same condition as the original ticker-based
    checks, instead of a second, potentially drifting copy of it."""
    return bool(info) and (
        info.get("regularMarketPrice") is not None or info.get("currentPrice") is not None
    )


def get_stock_summary(ticker: str, expected_name: str | None = None) -> dict:
    """
    Fetch key financial data for a single ticker symbol.
    Returns a flat dictionary of data points, or {'error': '...'} on failure.

    Problema 37 (Iteration 4, Sezione 4): if the symbol as given does not
    resolve directly, this now retries once via resolve_ticker() before
    giving up — this is what lets an already-correct but less common
    symbol (e.g. "PST.MI"), or a name/guess the direct yf.Ticker lookup
    can't handle, still succeed. Callers see no change to the function's
    contract (same success shape, same {'error': ...} shape on failure);
    the only difference is what happens internally between the first failed
    lookup and the final error being returned.

    Problema 38 (Iteration 4, Sezione 4): if an `expected_name` hint is also
    given (the company name paired with this ticker during extraction —
    see rag_pipeline.extract_ticker_candidates), it is tried as a SECOND,
    independent resolution path, tried ONLY once the ticker as given has
    already failed to find any priced data at all. Searching yfinance by
    the actual company name rather than by a ticker guess already shown
    not to work is what lets a wrong-AND-nonexistent guess self-correct
    (confirmed case: "PT.MI" guessed for Poste Italiane, where "PST.MI"
    was the real symbol). The same path applies identically to any
    market, including a hypothetical wrong-and-nonexistent US guess, with
    no per-company table.

    Ordering (revised after a confirmed live failure — see below):
    the NAME-based candidate (resolve_ticker(expected_name)) is tried
    BEFORE the ticker-based one (resolve_ticker(ticker)), not after.
    Originally the ticker-based fallback was tried first and the name
    fallback was only a last resort if that also came up empty. That
    order silently broke on a real, confirmed case: Block, Inc. traded
    as "SQ" on NYSE until its symbol changed to "XYZ" in January 2025;
    the LLM extractor (general knowledge, not live-updated) still
    guesses "SQ". A direct yf.Ticker("SQ") lookup correctly fails (no
    price data under that symbol any more) — but resolve_ticker("SQ")
    then fuzzy-matches, via yf.Search()'s free-text index, an entirely
    unrelated company: Bristol-Myers Squibb ("BMY"), apparently because
    "Squibb" textually contains "SQ". That match DOES have real price
    data, so under the old ordering the code accepted it as a success
    and never even tried resolve_ticker("Block, Inc."), which would have
    found the real, current symbol ("XYZ"). A short, already-shown-not-
    to-work ticker string is weaker search evidence than the actual
    company name once the direct lookup has failed — trying the name
    first fixes this without adding any new risk: this whole block is
    still only reached after the DIRECT lookup has already failed, so a
    ticker that resolves immediately (any market, including Google/
    Facebook below) never reaches either fallback and is unaffected.

    Deliberately NOT extended to re-validate a ticker that already
    resolves successfully on the very first attempt: an earlier version
    of this fix considered comparing the returned company name against
    expected_name on every successful lookup too (which would also close
    the older, still-open Problema 30 residual — a wrong ticker that DOES
    have data on its OWN first try, e.g. "ISP" resolving to ING Groep NV
    instead of Intesa Sanpaolo, never revalidated by this fix). That was
    deliberately not implemented: a generic name-similarity check cannot
    distinguish "wrong company" from a legitimate brand/legal-name
    mismatch that already exists in this same codebase (a user asking
    about "Google" or "Facebook" gets back "Alphabet Inc."/"Meta
    Platforms, Inc." from yfinance — see COMMON_TICKER_FIXES, Problema
    12) — applying it universally would trade a rare, unconfirmed failure
    mode for a new, guaranteed one on two of the most common queries this
    app receives. So the Problema 30 "wrong-but-existing-on-first-try"
    case remains an explicitly documented residual limitation (see
    Diario Tecnico, Problema 38); what changed here is narrower — it is
    the SECOND-stage resolution (already past a failed direct lookup)
    that now prefers the name over a demonstrably-unreliable ticker
    string, not a new check on first-try successes.
    """
    try:
        resolved_ticker = ticker.upper()
        stock = yf.Ticker(resolved_ticker)
        info = stock.info

        # Validate that we received a real ticker
        if not _has_price_data(info):
            if expected_name:
                name_fallback_symbol = resolve_ticker(expected_name)
                if name_fallback_symbol and name_fallback_symbol != resolved_ticker:
                    candidate_stock = yf.Ticker(name_fallback_symbol)
                    candidate_info = candidate_stock.info
                    if _has_price_data(candidate_info):
                        resolved_ticker = name_fallback_symbol
                        stock = candidate_stock
                        info = candidate_info

            if not _has_price_data(info):
                fallback_symbol = resolve_ticker(ticker)
                if fallback_symbol and fallback_symbol != resolved_ticker:
                    resolved_ticker = fallback_symbol
                    stock = yf.Ticker(resolved_ticker)
                    info = stock.info

            if not _has_price_data(info):
                return {"error": f"No data found for ticker '{ticker}'. It may be delisted or invalid."}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        change_pct = round(((price - prev_close) / prev_close) * 100, 2) if price and prev_close else None

        week_low = info.get("fiftyTwoWeekLow")
        week_high = info.get("fiftyTwoWeekHigh")
        week_range = f"{week_low} – {week_high}" if week_low and week_high else "N/A"

        # Recent news headlines (up to 3)
        news_items = stock.news or []
        headlines = [item.get("title", "") for item in news_items[:3] if item.get("title")]

        eps = info.get("trailingEps")
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        # Problema 28 (Iteration 4, Sezione 4): yfinance's trailingPE can be
        # computed over a different trailing-earnings window than the
        # trailingEps figure returned alongside it, so the two can go
        # mutually inconsistent — most visibly once a company swings to a
        # net loss, where a mathematically correct P/E (price / negative
        # EPS) must itself be negative, but trailingPE was observed
        # returning a large POSITIVE number instead (found on WBD: P/E
        # 583.38 shown next to EPS -1.28). Rather than pass through a
        # figure that is numerically present but not a coherent P/E for
        # this EPS, treat it as untrustworthy whenever EPS is negative —
        # same "don't silently present a bad number as good" principle
        # already used for unpriced tickers (portfolio.py) and low-history
        # MPT estimates (optimizer.py, Problema 25).
        if eps is not None and eps < 0 and pe_ratio is not None:
            pe_ratio = "N/A (negative earnings)"

        return {
            "ticker": resolved_ticker,
            "name": info.get("longName") or info.get("shortName", resolved_ticker),
            "price": price,
            "change_pct": change_pct,
            "52_week_range": week_range,
            "pe_ratio": pe_ratio,
            "eps": eps,
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "market_cap": info.get("marketCap"),
            "recommendation": info.get("recommendationKey", "N/A").replace("_", " ").title(),
            "target_price": info.get("targetMeanPrice"),
            "sector": info.get("sector", "N/A"),
            "description": (info.get("longBusinessSummary", "") or "")[:400],
            "news_headlines": headlines,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        return {"error": str(e)}


def get_price_history(ticker: str, period: str = "3mo"):
    """
    Fetch historical daily price data for charting, with a 20-day moving
    average column (MA20) precomputed for the UI's Plotly chart.

    Returns a pandas DataFrame (yfinance's own format, indexed by date,
    with an added "MA20" column) on success, or None on any failure —
    invalid ticker, no data, network error — so a charting problem never
    blocks the rest of the response; the UI simply omits the chart.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        if hist is None or hist.empty:
            return None
        hist["MA20"] = hist["Close"].rolling(window=20, min_periods=1).mean()
        return hist
    except Exception:
        return None


def get_multiple_stock_summaries(
    tickers: list[str], expected_names: dict[str, str] | None = None
) -> list[dict]:
    """
    Fetch stock summaries for up to MAX_TICKERS tickers.
    Each entry in the returned list is the dict produced by get_stock_summary,
    tagged with its ticker even in the error case so the caller can report
    which specific ticker failed.

    Problema 38 (Iteration 4, Sezione 4): optional expected_names maps each
    ticker (uppercase) to the company name paired with it during extraction
    (see rag_pipeline.extract_ticker_candidates), forwarded to
    get_stock_summary()'s expected_name so a wrong-and-nonexistent guess can
    self-correct via the same name-search fallback as the single-ticker
    path — applied independently per company, so one wrong guess in a
    3-company comparison does not affect the other two, and a company with
    no entry in expected_names (or no dict at all) behaves exactly as
    before this parameter existed. Defaults to None, so every existing
    caller (get_multiple_stock_summaries(tickers), no second argument) is
    unaffected.
    """
    expected_names = expected_names or {}
    results = []
    for ticker in tickers[:MAX_TICKERS]:
        summary = get_stock_summary(ticker, expected_name=expected_names.get(ticker.upper()))
        if "error" in summary:
            summary = {"ticker": ticker.upper(), **summary}
        results.append(summary)
    return results


def build_data_context(stock_data: dict) -> str:
    """
    Convert a single stock data dictionary into a structured text block
    to be injected into the RAG prompt as context.
    """
    headlines_text = ""
    if stock_data.get("news_headlines"):
        headlines_text = "\nRecent news:\n" + "\n".join(
            f"  - {h}" for h in stock_data["news_headlines"]
        )

    dividend = (
        f"{round(stock_data['dividend_yield'], 2)}%"
        if stock_data.get("dividend_yield")
        else "None"
    )

    return f"""
=== RETRIEVED FINANCIAL DATA ===
Ticker: {stock_data['ticker']}
Company: {stock_data['name']}
Sector: {stock_data.get('sector', 'N/A')}
Current price: ${stock_data.get('price', 'N/A')}
Day change: {stock_data.get('change_pct', 'N/A')}%
52-week range: {stock_data.get('52_week_range', 'N/A')}
P/E ratio (trailing): {stock_data.get('pe_ratio', 'N/A')}
EPS (trailing): {stock_data.get('eps', 'N/A')}
Beta: {stock_data.get('beta', 'N/A')}
Dividend yield: {dividend}
Analyst consensus: {stock_data.get('recommendation', 'N/A')}
Analyst target price: ${stock_data.get('target_price', 'N/A')}
Company description: {stock_data.get('description', 'N/A')}
{headlines_text}
Data retrieved at: {stock_data.get('timestamp', 'N/A')}
=================================
"""


def build_comparative_context(stock_data_list: list[dict]) -> str:
    """
    Build a single context block covering multiple tickers so the LLM can
    compare them directly instead of receiving isolated single-stock blocks.

    Tickers that failed retrieval are listed separately so the model (and
    the transparency panel in the UI) can be explicit about what data is
    actually available, rather than silently ignoring the failure.
    """
    valid = [d for d in stock_data_list if "error" not in d]
    failed = [d for d in stock_data_list if "error" in d]

    if not valid:
        return "=== RETRIEVED FINANCIAL DATA ===\nNo valid data could be retrieved for any requested ticker.\n=================================\n"

    blocks = [build_data_context(d).strip() for d in valid]

    header = f"=== COMPARATIVE FINANCIAL DATA ({len(valid)} companies) ===\n"
    body = "\n\n".join(blocks)

    footer = ""
    if failed:
        failed_list = ", ".join(d["ticker"] for d in failed)
        footer = f"\n\nNote: data could not be retrieved for: {failed_list}. Do not fabricate figures for these tickers."

    return f"{header}\n{body}{footer}\n"


def get_current_price(ticker: str) -> float | None:
    """
    Iteration 3 (portfolio tracker). Lightweight current-price lookup used
    to compute portfolio profit/loss. Deliberately reuses get_stock_summary()
    rather than a separate yfinance call, so ticker validation and error
    handling stay in one place, the portfolio tracker should not need its
    own copy of the "is this a real ticker" logic.

    Returns the current price as a float, or None if the ticker could not
    be resolved (e.g. delisted, typo, or a temporary API failure). Callers
    must treat None as "price unavailable", not as zero, a holding with an
    unavailable price should be shown as such, never silently priced at $0.
    """
    summary = get_stock_summary(ticker)
    if "error" in summary:
        return None
    return summary.get("price")


def get_closing_prices(ticker: str, period: str = "1y"):
    """
    Iteration 4 (MPT optimizer and genetic-algorithm backtester). Historical
    closing-price series for a ticker, reusing get_price_history() rather
    than a separate yfinance call — same "don't duplicate the is-this-a-
    real-ticker / network logic" principle as get_current_price() reusing
    get_stock_summary().

    Returns a pandas Series of closing prices indexed by date, or None if
    history could not be retrieved (invalid ticker, no data, network
    error) — never an empty or fabricated series, so callers (optimizer.py,
    backtesting.py) can treat None as "exclude this ticker" unambiguously.
    """
    hist = get_price_history(ticker, period=period)
    if hist is None or hist.empty:
        return None
    return hist["Close"]
