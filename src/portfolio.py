"""
portfolio.py
Portfolio tracker business logic — Iteration 3.

Kept separate from database.py (pure persistence) and from app.py (pure
UI), so profit/loss calculation can be unit-tested with plain dicts and a
fake price lookup, without touching SQLite or the network. This mirrors the
existing separation between financial_data.py (retrieval) and
rag_pipeline.py (reasoning over retrieved data).
"""

from typing import Callable, Optional


def compute_holding_pnl(holding: dict, current_price: Optional[float]) -> dict:
    """
    Enrich a single holding dict (as returned by database.get_portfolio)
    with current price, market value, and profit/loss.

    If current_price is None (price could not be retrieved), the holding is
    returned with pnl fields set to None rather than 0 — a missing price
    must never be silently treated as a $0 loss.
    """
    enriched = dict(holding)
    shares = holding["shares"]
    purchase_price = holding["purchase_price"]
    cost_basis = shares * purchase_price

    if current_price is None:
        enriched.update(
            current_price=None,
            market_value=None,
            pnl=None,
            pnl_pct=None,
            cost_basis=cost_basis,
        )
        return enriched

    market_value = shares * current_price
    pnl = market_value - cost_basis
    pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else None

    enriched.update(
        current_price=current_price,
        market_value=market_value,
        pnl=pnl,
        pnl_pct=pnl_pct,
        cost_basis=cost_basis,
    )
    return enriched


def compute_portfolio_summary(
    holdings: list[dict],
    price_lookup: Callable[[str], Optional[float]],
) -> dict:
    """
    Compute per-holding and total profit/loss for a portfolio.

    price_lookup is injected (rather than calling financial_data directly)
    so this function can be unit-tested with a fake price map instead of
    hitting yfinance. app.py passes financial_data.get_current_price here.

    One live price lookup is performed per DISTINCT ticker, not per holding,
	a portfolio with three separate AAPL purchases only costs one API
    call, not three. Holdings whose price lookup fails are still included
    in the per-holding results (with pnl fields set to None) but excluded
    from the portfolio totals, so one bad ticker cannot silently corrupt
    the total P&L of an otherwise-healthy portfolio.
    """
    distinct_tickers = {h["ticker"] for h in holdings}
    price_cache = {ticker: price_lookup(ticker) for ticker in distinct_tickers}

    enriched_holdings = [
        compute_holding_pnl(h, price_cache[h["ticker"]]) for h in holdings
    ]

    valid = [h for h in enriched_holdings if h["pnl"] is not None]
    total_cost_basis = sum(h["cost_basis"] for h in valid)
    total_market_value = sum(h["market_value"] for h in valid)
    total_pnl = total_market_value - total_cost_basis
    total_pnl_pct = (total_pnl / total_cost_basis) * 100 if total_cost_basis > 0 else None

    return {
        "holdings": enriched_holdings,
        "total_cost_basis": total_cost_basis,
        "total_market_value": total_market_value,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "unpriced_tickers": sorted({h["ticker"] for h in enriched_holdings if h["pnl"] is None}),
    }
