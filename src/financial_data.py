"""
financial_data.py
Data Retrieval layer — fetches stock data from Yahoo Finance via yfinance.

Iteration 1: adds multi-ticker retrieval and a comparative context block
so the RAG pipeline can answer questions that mention more than one company
(e.g. "Compare Apple and Microsoft").
"""

import yfinance as yf
from datetime import datetime

MAX_TICKERS = 3


def get_stock_summary(ticker: str) -> dict:
    """
    Fetch key financial data for a single ticker symbol.
    Returns a flat dictionary of data points, or {'error': '...'} on failure.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        # Validate that we received a real ticker
        if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
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

        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName", ticker.upper()),
            "price": price,
            "change_pct": change_pct,
            "52_week_range": week_range,
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "eps": info.get("trailingEps"),
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


def get_multiple_stock_summaries(tickers: list[str]) -> list[dict]:
    """
    Fetch stock summaries for up to MAX_TICKERS tickers.
    Each entry in the returned list is the dict produced by get_stock_summary,
    tagged with its ticker even in the error case so the caller can report
    which specific ticker failed.
    """
    results = []
    for ticker in tickers[:MAX_TICKERS]:
        summary = get_stock_summary(ticker)
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
        f"{round(stock_data['dividend_yield'] * 100, 2)}%"
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
