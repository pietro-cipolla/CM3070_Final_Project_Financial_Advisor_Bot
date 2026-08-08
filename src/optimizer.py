"""
optimizer.py
Iteration 4: modern Portfolio Theory optimization.

Suggests a rebalancing of the user's tracked portfolio (Iteration 3) that
maximises the Sharpe ratio for the set of tickers currently held, given
their historical daily returns. This is the project's third algorithmic
component alongside sentiment analysis (VADER) and the genetic-algorithm
backtester, and move the "active portfolio
management" language from the template description in an actual
portfolio-level optimization, rather than only single-stock analysis.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MIN_HISTORY_POINTS = 30
TRADING_DAYS_PER_YEAR = 252


def compute_daily_returns(prices: pd.Series) -> pd.Series:
    """
    Simple (not log) daily percentage returns, matching the convention
    already used for portfolio profit/loss (compute_holding_pnl in
    src/portfolio.py uses simple, not log, returns).
    """
    return prices.pct_change().dropna()


def annualize_returns_and_covariance(
    price_histories: dict[str, pd.Series],
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Build an annualized mean-return vector and covariance matrix from a
    dict of {ticker: price series}.

    All series are aligned on their shared dates (inner join, via
    pd.DataFrame(...).dropna()) before computing returns, so tickers with
    slightly different trading calendars (e.g. a recent IPO with a shorter
    history) don't silently misalign into meaningless day-to-day pairings.
    """
    price_df = pd.DataFrame(price_histories).dropna(how="any")
    daily_returns = price_df.pct_change().dropna(how="any")
    mean_returns = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    cov_matrix = daily_returns.cov() * TRADING_DAYS_PER_YEAR
    return mean_returns, cov_matrix


def portfolio_performance(
    weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame
) -> tuple[float, float]:
    """Return (expected annual return, expected annual volatility) for a
    given weight vector, in the same ticker order as mean_returns.index."""
    expected_return = float(np.dot(weights, mean_returns))
    expected_volatility = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    return expected_return, expected_volatility


def _negative_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """Objective function for the minimizer: scipy only minimizes, so we
    minimize the negative Sharpe ratio to maximise the Sharpe ratio."""
    expected_return, expected_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    if expected_volatility == 0:
        return 0.0
    return -(expected_return - risk_free_rate) / expected_volatility


def optimize_weights(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Solve for the long-only, fully-invested weight vector that maximises
    the Sharpe ratio, using scipy's SLSQP solver.

    Returns a dict with "weights" ({ticker: weight}), "expected_return",
    "expected_volatility", "sharpe_ratio", "converged" (bool), and "note"
    (str or None) for the solution found.
    """
    tickers = list(mean_returns.index)
    n = len(tickers)

    if n == 1:
        weights = np.array([1.0])
        expected_return, expected_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (
            (expected_return - risk_free_rate) / expected_volatility
            if expected_volatility else None
        )
        return {
            "weights": {tickers[0]: 1.0},
            "expected_return": expected_return,
            "expected_volatility": expected_volatility,
            "sharpe_ratio": sharpe,
            "converged": True,
            "note": (
                "Only one ticker with usable history, mean-variance optimization "
                "has no diversification to exploit, so the only long-only, "
                "fully-invested allocation is 100% in this ticker."
            ),
        }

    initial_guess = np.repeat(1.0 / n, n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    result = minimize(
        _negative_sharpe_ratio,
        initial_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = result.x if result.success else initial_guess
    expected_return, expected_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = (
        (expected_return - risk_free_rate) / expected_volatility
        if expected_volatility else None
    )

    return {
        "weights": {ticker: float(w) for ticker, w in zip(tickers, weights)},
        "expected_return": expected_return,
        "expected_volatility": expected_volatility,
        "sharpe_ratio": sharpe,
        "converged": bool(result.success),
        "note": None if result.success else (
            "The optimizer did not converge; showing an equal-weight allocation "
            "instead of a possibly unreliable partial solution."
        ),
    }


def compute_current_weights(
    holdings: list[dict], price_lookup: Callable[[str], Optional[float]]
) -> dict:
    """
    Market-value weights of the portfolio as it stands today, for
    side-by-side comparison against the suggested weights. Reuses the same
    price_lookup convention as compute_portfolio_summary in src/portfolio.py.
    Holdings whose current price is
    unavailable are excluded from the weights entirely, never treated as
    zero value (which would silently understate that ticker's true
    allocation).

    Multiple lots of the same ticker (e.g. two separate purchases of AAPL)
    are summed into a single weight, matching how the portfolio tracker
    already displays holdings ticker-by-ticker rather than lot-by-lot for
    the purpose of an allocation percentage.
    """
    by_ticker: dict[str, float] = {}
    distinct_tickers = {h["ticker"] for h in holdings}
    price_cache = {t: price_lookup(t) for t in distinct_tickers}

    for h in holdings:
        price = price_cache[h["ticker"]]
        if price is None:
            continue
        by_ticker[h["ticker"]] = by_ticker.get(h["ticker"], 0.0) + h["shares"] * price

    total = sum(by_ticker.values())
    if total <= 0:
        return {}
    return {ticker: value / total for ticker, value in by_ticker.items()}


def compute_portfolio_optimization(
    holdings: list[dict],
    history_lookup: Callable[[str], Optional[pd.Series]],
    price_lookup: Optional[Callable[[str], Optional[float]]] = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    End-to-end Markowitz optimization for a tracked portfolio.
	Fetches one historical price series per distinct ticker via
    history_lookup, builds annualized return/covariance estimates, and
    returns suggested long-only weights maximising the Sharpe ratio.

    Tickers are excluded (and reported in `excluded_tickers`, with a
    reason) rather than silently dropped when: history could not be
    retrieved at all, or fewer than MIN_HISTORY_POINTS usable data points
    are available.

    If price_lookup is also supplied, `current_weights` gives the
    portfolio's present market-value allocation for comparison against
    'suggested_weights', this is optional because computing it requires a
    second, different kind of price data (current price, not historical
    series) that a caller who only wants a suggested allocation may not
    need to fetch.

    Returns a dict:
      {
        "suggested_weights": {ticker: weight, ...}  (empty if none usable),
        "current_weights": {ticker: weight, ...} or None,
        "expected_return": float or None,
        "expected_volatility": float or None,
        "sharpe_ratio": float or None,
        "excluded_tickers": {ticker: reason, ...},
        "note": str or None,
      }
    """
    distinct_tickers = sorted({h["ticker"] for h in holdings})
    excluded: dict[str, str] = {}
    price_histories: dict[str, pd.Series] = {}

    for ticker in distinct_tickers:
        series = history_lookup(ticker)
        if series is None or len(series) == 0:
            excluded[ticker] = "price history unavailable"
            continue
        if len(series) < MIN_HISTORY_POINTS:
            excluded[ticker] = f"only {len(series)} data points available (minimum {MIN_HISTORY_POINTS})"
            continue
        price_histories[ticker] = series

    current_weights = compute_current_weights(holdings, price_lookup) if price_lookup else None

    if len(price_histories) == 0:
        return {
            "suggested_weights": {},
            "current_weights": current_weights,
            "expected_return": None,
            "expected_volatility": None,
            "sharpe_ratio": None,
            "excluded_tickers": excluded,
            "note": "No holdings had enough usable price history to run the optimization.",
        }

    mean_returns, cov_matrix = annualize_returns_and_covariance(price_histories)
    result = optimize_weights(mean_returns, cov_matrix, risk_free_rate)

    return {
        "suggested_weights": result["weights"],
        "current_weights": current_weights,
        "expected_return": result["expected_return"],
        "expected_volatility": result["expected_volatility"],
        "sharpe_ratio": result["sharpe_ratio"],
        "excluded_tickers": excluded,
        "note": result["note"],
    }
