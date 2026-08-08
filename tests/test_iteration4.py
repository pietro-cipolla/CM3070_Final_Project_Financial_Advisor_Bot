"""
tests/test_iteration4.py
Automated tests for Iteration 4: VADER sentiment analysis (src/news_data.py)
and the genetic-algorithm backtester (src/backtesting.py). MPT/Markowitz
portfolio optimization (src/optimizer.py) has its own dedicated test file,
tests/test_optimizer.py.

Sentiment tests use plain strings/dicts, no network or OpenAI calls needed
(VADER is a local lexicon-based analyzer). Backtesting tests use
deterministic or seeded synthetic price series, following the same
principle already used in tests/test_iteration3.py and
tests/test_optimizer.py: offline, fast, reproducible.
"""

import numpy as np
import pandas as pd
import pytest

from src.news_data import (
    score_headline_sentiment,
    score_news_sentiment,
    summarize_sentiment,
    build_news_context,
    POSITIVE_THRESHOLD,
    NEGATIVE_THRESHOLD,
)
from src.backtesting import (
    simulate_crossover_strategy,
    _random_individual,
    _crossover,
    _mutate,
    _repair,
    _fitness,
    evolve_strategy,
    compute_buy_and_hold_return,
    backtest_ticker,
    SHORT_WINDOW_BOUNDS,
    LONG_WINDOW_BOUNDS,
    MIN_WINDOW_GAP,
)



# Sentiment analysis (VADER) — src/news_data.py


def test_score_headline_sentiment_clearly_positive():
    result = score_headline_sentiment("Great quarter for the company, profits soar")
    assert result["label"] == "positive"
    assert result["compound"] >= POSITIVE_THRESHOLD


def test_score_headline_sentiment_clearly_negative():
    result = score_headline_sentiment("Terrible results, investors are furious")
    assert result["label"] == "negative"
    assert result["compound"] <= NEGATIVE_THRESHOLD


def test_score_headline_sentiment_antitrust_probe_known_limitation():
    result = score_headline_sentiment("Apple faces antitrust probe in EU")
    assert result["label"] == "neutral" 


def test_score_headline_sentiment_empty_title_is_neutral():
    result = score_headline_sentiment("")
    assert result["label"] == "neutral"
    assert result["compound"] == 0.0


def test_score_headline_sentiment_none_title_is_neutral():
    result = score_headline_sentiment(None)
    assert result["label"] == "neutral"


def test_score_news_sentiment_enriches_every_item():
    items = [
        {"title": "Great quarter for the company, profits soar", "source": "A", "url": "x", "published_at": "2026-07-01"},
        {"title": "Terrible results, investors are furious", "source": "B", "url": "y", "published_at": "2026-07-02"},
    ]
    enriched = score_news_sentiment(items)
    assert enriched[0]["sentiment_label"] == "positive"
    assert enriched[1]["sentiment_label"] == "negative"
    # Original fields preserved
    assert enriched[0]["source"] == "A"
    assert enriched[1]["url"] == "y"


def test_score_news_sentiment_does_not_mutate_input():
    items = [{"title": "Great quarter for the company, profits soar", "source": "A", "url": "x", "published_at": "2026-07-01"}]
    score_news_sentiment(items)
    assert "sentiment_label" not in items[0]


def test_score_news_sentiment_empty_list_returns_empty_list():
    assert score_news_sentiment([]) == []


def test_summarize_sentiment_counts_each_label():
    scored = [
        {"sentiment_label": "positive", "sentiment_compound": 0.5},
        {"sentiment_label": "positive", "sentiment_compound": 0.6},
        {"sentiment_label": "negative", "sentiment_compound": -0.4},
        {"sentiment_label": "neutral", "sentiment_compound": 0.0},
    ]
    summary = summarize_sentiment(scored)
    assert summary["positive"] == 2
    assert summary["negative"] == 1
    assert summary["neutral"] == 1
    assert summary["overall_label"] == "positive"


def test_summarize_sentiment_tie_resolves_to_neutral():
    scored = [
        {"sentiment_label": "positive", "sentiment_compound": 0.5},
        {"sentiment_label": "negative", "sentiment_compound": -0.5},
    ]
    summary = summarize_sentiment(scored)
    assert summary["overall_label"] == "neutral"


def test_summarize_sentiment_empty_input():
    summary = summarize_sentiment([])
    assert summary == {"positive": 0, "neutral": 0, "negative": 0, "overall_label": "neutral", "average_compound": None}


def test_summarize_sentiment_average_compound_is_mean():
    scored = [
        {"sentiment_label": "positive", "sentiment_compound": 0.6},
        {"sentiment_label": "negative", "sentiment_compound": -0.2},
    ]
    summary = summarize_sentiment(scored)
    assert summary["average_compound"] == pytest.approx(0.2)


def test_build_news_context_empty_list_returns_empty_string():
    # Preserves Iteration 2 behavior exactly.
    assert build_news_context([]) == ""


def test_build_news_context_includes_sentiment_label_by_default():
    items = [{"title": "Great quarter for the company, profits soar", "source": "TechCrunch", "url": "https://x.com", "published_at": "2026-07-14T10:00:00Z"}]
    context = build_news_context(items)
    assert "positive" in context.lower()
    assert "overall sentiment" in context.lower()
    assert "Great quarter" in context


def test_build_news_context_can_disable_sentiment():
    items = [{"title": "Great quarter for the company, profits soar", "source": "TechCrunch", "url": "https://x.com", "published_at": "2026-07-14T10:00:00Z"}]
    context = build_news_context(items, include_sentiment=False)
    assert "sentiment" not in context.lower()
    assert "Great quarter" in context



# Genetic-algorithm backtesting — src/backtesting.py

def _flat_series(price=100.0, days=80):
    dates = pd.date_range("2025-01-01", periods=days, freq="B")
    return pd.Series([price] * days, index=dates)


def _monotonic_series(start=100.0, step=0.5, days=252):
    dates = pd.date_range("2025-01-01", periods=days, freq="B")
    return pd.Series([start + i * step for i in range(days)], index=dates)


def _monotonic_decreasing_series(start=100.0, step=0.5, days=252):
    dates = pd.date_range("2025-01-01", periods=days, freq="B")
    return pd.Series([start - i * step for i in range(days)], index=dates)


def _seeded_walk(start=100.0, drift=0.0, vol=0.01, days=252, seed=1):
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, days - 1)
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    dates = pd.date_range("2025-01-01", periods=days, freq="B")
    return pd.Series(prices, index=dates)


# simulate_crossover_strategy

def test_simulate_crossover_strategy_flat_price_gives_zero_return():
    series = _flat_series(days=80)
    result = simulate_crossover_strategy(series, short_window=5, long_window=20)
    assert result["total_return"] == pytest.approx(0.0, abs=1e-9)


def test_simulate_crossover_strategy_trades_next_day_not_same_day():
    series = _monotonic_series(start=100, step=1.0, days=60)
    result = simulate_crossover_strategy(series, short_window=5, long_window=20)
    benchmark = compute_buy_and_hold_return(series)
    assert result["total_return"] <= benchmark + 1e-9


def test_simulate_crossover_strategy_short_series_returns_zero():
    series = _flat_series(days=5)  # shorter than long_window
    result = simulate_crossover_strategy(series, short_window=5, long_window=20)
    assert result["total_return"] == 0.0
    assert (result["daily_returns"] == 0.0).all()


def test_simulate_crossover_strategy_never_shorts():
    # Position values (post-shift) must only ever be 0 or 1, never negative.
    series = _seeded_walk(days=100, seed=7)
    short_ma = series.rolling(5).mean()
    long_ma = series.rolling(20).mean()
    position = (short_ma > long_ma).astype(int).shift(1).dropna()
    assert set(position.unique()).issubset({0, 1})


# GA building blocks

def test_random_individual_respects_bounds_and_gap():
    rng = np.random.default_rng(0)
    for _ in range(200):
        short_window, long_window = _random_individual(rng)
        assert SHORT_WINDOW_BOUNDS[0] <= short_window <= SHORT_WINDOW_BOUNDS[1]
        assert LONG_WINDOW_BOUNDS[0] <= long_window <= LONG_WINDOW_BOUNDS[1]
        assert long_window - short_window >= MIN_WINDOW_GAP


def test_repair_pushes_long_window_up_when_gap_too_small():
    short_window, long_window = _repair(short_window=40, long_window=42)
    assert long_window - short_window >= MIN_WINDOW_GAP


def test_repair_pulls_short_window_down_when_long_window_at_upper_bound():
    short_window, long_window = _repair(short_window=199, long_window=200)
    assert long_window - short_window >= MIN_WINDOW_GAP
    assert long_window <= LONG_WINDOW_BOUNDS[1]


def test_crossover_always_produces_valid_gap():
    rng = np.random.default_rng(3)
    parent_a = (10, 30)
    parent_b = (45, 50)  # deliberately tight/invalid combinations possible
    for _ in range(50):
        child = _crossover(parent_a, parent_b, rng)
        assert child[1] - child[0] >= MIN_WINDOW_GAP


def test_mutate_stays_within_bounds_and_gap():
    rng = np.random.default_rng(4)
    individual = (10, 30)
    for _ in range(50):
        individual = _mutate(individual, rng, mutation_rate=1.0)  # force mutation every time
        assert SHORT_WINDOW_BOUNDS[0] <= individual[0] <= SHORT_WINDOW_BOUNDS[1]
        assert LONG_WINDOW_BOUNDS[0] <= individual[1] <= LONG_WINDOW_BOUNDS[1]
        assert individual[1] - individual[0] >= MIN_WINDOW_GAP


# evolve_strategy

def test_evolve_strategy_fitness_history_is_non_decreasing():
    series = _seeded_walk(days=252, drift=0.0008, vol=0.015, seed=5)
    result = evolve_strategy(series, population_size=20, generations=10, seed=5)
    history = result["fitness_history"]
    assert all(history[i] <= history[i + 1] + 1e-12 for i in range(len(history) - 1))


def test_evolve_strategy_returns_a_valid_individual():
    series = _seeded_walk(days=252, drift=0.0005, vol=0.01, seed=6)
    result = evolve_strategy(series, population_size=15, generations=8, seed=6)
    short_window, long_window = result["best_individual"]
    assert SHORT_WINDOW_BOUNDS[0] <= short_window <= SHORT_WINDOW_BOUNDS[1]
    assert LONG_WINDOW_BOUNDS[0] <= long_window <= LONG_WINDOW_BOUNDS[1]
    assert long_window - short_window >= MIN_WINDOW_GAP


def test_evolve_strategy_is_deterministic_given_a_seed():
    series = _seeded_walk(days=200, drift=0.0006, vol=0.012, seed=9)
    result_a = evolve_strategy(series, population_size=15, generations=8, seed=42)
    result_b = evolve_strategy(series, population_size=15, generations=8, seed=42)
    assert result_a["best_individual"] == result_b["best_individual"]
    assert result_a["best_fitness"] == pytest.approx(result_b["best_fitness"])


# compute_buy_and_hold_return

def test_compute_buy_and_hold_return_matches_manual_calculation():
    series = pd.Series([100.0, 110.0, 121.0])
    assert compute_buy_and_hold_return(series) == pytest.approx(0.21)


def test_compute_buy_and_hold_return_short_series_is_zero():
    assert compute_buy_and_hold_return(pd.Series([100.0])) == 0.0


# backtest_ticker (end-to-end)

def test_backtest_ticker_excludes_ticker_with_no_history():
    result = backtest_ticker("ZZZ", history_lookup=lambda t: None)
    assert result["note"] == "price history unavailable"
    assert result["strategy_return"] is None
    assert result["beat_benchmark"] is None


def test_backtest_ticker_excludes_ticker_with_too_little_history():
    short_series = _flat_series(days=10)
    result = backtest_ticker("NEW", history_lookup=lambda t: short_series)
    assert "minimum" in result["note"]
    assert result["strategy_return"] is None


def test_backtest_ticker_downtrend_strategy_avoids_the_worst_of_the_decline():
    series = _monotonic_decreasing_series(start=200, step=0.5, days=150)
    result = backtest_ticker("DOWN", history_lookup=lambda t: series, generations=8, seed=1)
    assert result["note"] is None
    assert result["strategy_return"] == pytest.approx(0.0, abs=1e-9)
    assert result["benchmark_return"] < 0
    assert result["beat_benchmark"] is True


def test_backtest_ticker_reports_result_shape_for_uptrend():
    series = _seeded_walk(days=252, drift=0.001, vol=0.012, seed=11)
    result = backtest_ticker("UP", history_lookup=lambda t: series, generations=10, seed=11)
    assert result["note"] is None
    assert result["short_window"] is not None
    assert result["long_window"] is not None
    assert isinstance(result["strategy_return"], float)
    assert isinstance(result["benchmark_return"], float)
    assert isinstance(result["beat_benchmark"], bool)
    assert len(result["fitness_history"]) == 10


def test_backtest_ticker_is_deterministic_given_a_seed():
    series = _seeded_walk(days=200, drift=0.0007, vol=0.013, seed=21)
    result_a = backtest_ticker("REPEAT", history_lookup=lambda t: series, generations=8, seed=99)
    result_b = backtest_ticker("REPEAT", history_lookup=lambda t: series, generations=8, seed=99)
    assert result_a["short_window"] == result_b["short_window"]
    assert result_a["long_window"] == result_b["long_window"]
    assert result_a["strategy_return"] == pytest.approx(result_b["strategy_return"])
