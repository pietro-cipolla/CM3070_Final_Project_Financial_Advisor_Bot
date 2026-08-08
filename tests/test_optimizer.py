"""
tests/test_optimizer.py
Automated tests for src/optimizer.py — Markowitz mean-variance portfolio
optimization, the third algorithmic component of Iteration 4.

Uses synthetic, seeded price series throughout rather than live yfinance
data, same principle as tests/test_iteration3.py's fake price_lookup:
deterministic, offline, and fast. No network access or API keys required
to run this file.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    compute_daily_returns,
    annualize_returns_and_covariance,
    portfolio_performance,
    optimize_weights,
    compute_current_weights,
    compute_portfolio_optimization,
)


def _price_series(start, daily_return, days=252, seed=None, noise=0.0):
    """
    Build a synthetic price series: deterministic daily growth of
    daily_return, optionally with small seeded Gaussian noise, so tests can
    construct assets with a known, controllable expected return/volatility
    instead of depending on real market data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="B")
    returns = np.full(days - 1, daily_return)
    if noise:
        returns = returns + rng.normal(0, noise, size=days - 1)
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return pd.Series(prices, index=dates)


# compute_daily_returns / annualize_returns_and_covariance

def test_compute_daily_returns_drops_first_nan():
    prices = pd.Series([100.0, 110.0, 121.0])
    returns = compute_daily_returns(prices)
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(0.10)


def test_annualize_returns_and_covariance_matches_known_growth_rate():
    series = _price_series(start=100, daily_return=0.001, days=252)
    mean_returns, cov_matrix = annualize_returns_and_covariance({"A": series})
    expected_annual = 0.001 * 252
    assert mean_returns["A"] == pytest.approx(expected_annual, rel=1e-6)
    assert cov_matrix.loc["A", "A"] == pytest.approx(0.0, abs=1e-10)


def test_annualize_returns_and_covariance_aligns_on_shared_dates():
    series_a = _price_series(start=100, daily_return=0.0005, days=100, seed=1, noise=0.01)
    series_b = _price_series(start=50, daily_return=0.0003, days=90, seed=2, noise=0.01)
    series_b.index = series_a.index[-90:]  # align B's shorter window to A's tail
    mean_returns, cov_matrix = annualize_returns_and_covariance({"A": series_a, "B": series_b})
    assert set(mean_returns.index) == {"A", "B"}
    assert cov_matrix.shape == (2, 2)


def test_portfolio_performance_matches_manual_calculation_for_two_assets():
    mean_returns = pd.Series({"A": 0.10, "B": 0.20})
    cov_matrix = pd.DataFrame({"A": [0.04, 0.0], "B": [0.0, 0.09]}, index=["A", "B"])
    weights = np.array([0.5, 0.5])
    expected_return, expected_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    assert expected_return == pytest.approx(0.15)
    assert expected_volatility == pytest.approx(np.sqrt(0.25 * 0.04 + 0.25 * 0.09))


# optimize_weights

def test_optimize_weights_single_ticker_is_fully_allocated():
    series = _price_series(start=100, daily_return=0.0004, days=60, seed=3, noise=0.01)
    mean_returns, cov_matrix = annualize_returns_and_covariance({"A": series})
    result = optimize_weights(mean_returns, cov_matrix)
    assert result["weights"] == {"A": 1.0}
    assert result["converged"] is True
    assert result["note"] is not None


def test_optimize_weights_prefers_higher_sharpe_asset():
    high = _price_series(start=100, daily_return=0.0020, days=252, seed=10, noise=0.01)
    low = _price_series(start=100, daily_return=0.0002, days=252, seed=20, noise=0.01)
    mean_returns, cov_matrix = annualize_returns_and_covariance({"HIGH": high, "LOW": low})
    result = optimize_weights(mean_returns, cov_matrix)
    assert result["weights"]["HIGH"] > result["weights"]["LOW"]
    assert result["converged"] is True


def test_optimize_weights_sums_to_one_and_is_long_only():
    a = _price_series(start=100, daily_return=0.0010, days=200, seed=11, noise=0.015)
    b = _price_series(start=80, daily_return=0.0006, days=200, seed=12, noise=0.02)
    c = _price_series(start=50, daily_return=0.0004, days=200, seed=13, noise=0.01)
    mean_returns, cov_matrix = annualize_returns_and_covariance({"A": a, "B": b, "C": c})
    result = optimize_weights(mean_returns, cov_matrix)
    total = sum(result["weights"].values())
    assert total == pytest.approx(1.0, abs=1e-4)
    assert all(w >= -1e-6 for w in result["weights"].values())  # no shorting


def test_optimize_weights_diversification_does_not_exceed_worse_single_asset_volatility():
    a = _price_series(start=100, daily_return=0.0008, days=252, seed=30, noise=0.02)
    b = _price_series(start=100, daily_return=0.0008, days=252, seed=99, noise=0.02)
    mean_returns, cov_matrix = annualize_returns_and_covariance({"A": a, "B": b})
    result = optimize_weights(mean_returns, cov_matrix)
    vol_a = np.sqrt(cov_matrix.loc["A", "A"])
    vol_b = np.sqrt(cov_matrix.loc["B", "B"])
    assert result["expected_volatility"] <= max(vol_a, vol_b) + 1e-6


# compute_current_weights

def test_compute_current_weights_uses_market_value():
    holdings = [
        {"ticker": "AAA", "shares": 10, "purchase_price": 5.0},
        {"ticker": "BBB", "shares": 5, "purchase_price": 20.0},
    ]
    prices = {"AAA": 10.0, "BBB": 20.0}  # AAA market value 100, BBB market value 100
    weights = compute_current_weights(holdings, lambda t: prices[t])
    assert weights["AAA"] == pytest.approx(0.5)
    assert weights["BBB"] == pytest.approx(0.5)


def test_compute_current_weights_excludes_unpriced_ticker():
    holdings = [
        {"ticker": "AAA", "shares": 10, "purchase_price": 5.0},
        {"ticker": "ZZZ", "shares": 5, "purchase_price": 20.0},
    ]
    prices = {"AAA": 10.0, "ZZZ": None}
    weights = compute_current_weights(holdings, lambda t: prices[t])
    assert "ZZZ" not in weights
    assert weights["AAA"] == pytest.approx(1.0)


def test_compute_current_weights_sums_multiple_lots_of_same_ticker():
    holdings = [
        {"ticker": "AAA", "shares": 10, "purchase_price": 5.0},
        {"ticker": "AAA", "shares": 10, "purchase_price": 6.0},
    ]
    weights = compute_current_weights(holdings, lambda t: 10.0)
    assert weights == {"AAA": 1.0}


def test_compute_current_weights_empty_holdings_returns_empty_dict():
    assert compute_current_weights([], lambda t: 10.0) == {}


# compute_portfolio_optimization (end-to-end)

def test_compute_portfolio_optimization_excludes_ticker_with_no_history():
    holdings = [
        {"ticker": "AAA", "shares": 10, "purchase_price": 5.0},
        {"ticker": "ZZZ", "shares": 5, "purchase_price": 20.0},
    ]
    good_series = _price_series(start=100, daily_return=0.0008, days=252, seed=40, noise=0.01)
    histories = {"AAA": good_series, "ZZZ": None}
    result = compute_portfolio_optimization(holdings, lambda t: histories[t])
    assert "ZZZ" in result["excluded_tickers"]
    assert result["excluded_tickers"]["ZZZ"] == "price history unavailable"
    assert result["suggested_weights"] == {"AAA": 1.0}


def test_compute_portfolio_optimization_excludes_ticker_with_too_little_history():
    holdings = [
        {"ticker": "AAA", "shares": 10, "purchase_price": 5.0},
        {"ticker": "NEW", "shares": 5, "purchase_price": 20.0},
    ]
    good_series = _price_series(start=100, daily_return=0.0008, days=252, seed=41, noise=0.01)
    short_series = _price_series(start=50, daily_return=0.0005, days=10, seed=42, noise=0.01)
    histories = {"AAA": good_series, "NEW": short_series}
    result = compute_portfolio_optimization(holdings, lambda t: histories[t])
    assert "NEW" in result["excluded_tickers"]
    assert "minimum" in result["excluded_tickers"]["NEW"]


def test_compute_portfolio_optimization_no_usable_history_returns_empty_weights():
    holdings = [{"ticker": "ZZZ", "shares": 5, "purchase_price": 20.0}]
    result = compute_portfolio_optimization(holdings, lambda t: None)
    assert result["suggested_weights"] == {}
    assert result["note"] is not None


def test_compute_portfolio_optimization_includes_current_weights_when_price_lookup_given():
    holdings = [{"ticker": "AAA", "shares": 10, "purchase_price": 5.0}]
    series = _price_series(start=100, daily_return=0.0008, days=252, seed=43, noise=0.01)
    result = compute_portfolio_optimization(
        holdings,
        history_lookup=lambda t: series,
        price_lookup=lambda t: 12.0,
    )
    assert result["current_weights"] == {"AAA": 1.0}


def test_compute_portfolio_optimization_current_weights_none_when_no_price_lookup():
    holdings = [{"ticker": "AAA", "shares": 10, "purchase_price": 5.0}]
    series = _price_series(start=100, daily_return=0.0008, days=252, seed=44, noise=0.01)
    result = compute_portfolio_optimization(holdings, history_lookup=lambda t: series)
    assert result["current_weights"] is None


def test_compute_portfolio_optimization_two_good_tickers_returns_full_result():
    holdings = [
        {"ticker": "AAA", "shares": 10, "purchase_price": 5.0},
        {"ticker": "BBB", "shares": 5, "purchase_price": 20.0},
    ]
    a = _price_series(start=100, daily_return=0.0010, days=252, seed=50, noise=0.015)
    b = _price_series(start=80, daily_return=0.0006, days=252, seed=51, noise=0.02)
    histories = {"AAA": a, "BBB": b}
    result = compute_portfolio_optimization(holdings, lambda t: histories[t])
    assert set(result["suggested_weights"].keys()) == {"AAA", "BBB"}
    assert sum(result["suggested_weights"].values()) == pytest.approx(1.0, abs=1e-4)
    assert result["expected_return"] is not None
    assert result["expected_volatility"] is not None
    assert result["sharpe_ratio"] is not None
    assert result["excluded_tickers"] == {}
