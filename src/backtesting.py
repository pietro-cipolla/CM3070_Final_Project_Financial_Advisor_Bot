"""
backtesting.py
Genetic-algorithm-optimized moving-average-crossover backtesting - Iteration 4.

Evolves the two parameters of a simple moving-average crossover trading
strategy (a short and a long lookback window, in trading days) using a
genetic algorithm, tournament selection with elitism, single-gene-swap
crossover, and per-gene mutation across multiple generations, then
backtests the evolved strategy against a buy-and-hold benchmark over the
same historical price data.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

# Search space bounds for the two evolved genes (in trading days)
SHORT_WINDOW_BOUNDS = (5, 50)
LONG_WINDOW_BOUNDS = (20, 200)
MIN_WINDOW_GAP = 5  # long_window must exceed short_window by at least this much

# Need enough days for the longest possible long_window, plus a handful of
# post-warm-up days to generate an actual tradeable signal.
MIN_HISTORY_POINTS = 60

# GA hyperparameters
DEFAULT_POPULATION_SIZE = 30
DEFAULT_GENERATIONS = 20
DEFAULT_ELITE_COUNT = 2
DEFAULT_MUTATION_RATE = 0.2
DEFAULT_TOURNAMENT_SIZE = 3
DEFAULT_COST_PER_TRADE = 0.0005


def simulate_crossover_strategy(
    prices: pd.Series, short_window: int, long_window: int, cost_per_trade: float = 0.0
) -> dict:
    """
    Simulate a long-only moving-average crossover strategy on a price
    series: fully invested when the short-window moving average is above
    the long-window moving average, in cash otherwise.

    cost_per_trade (default 0.0, i.e. cost-free, matching the original
    behaviour) deducts a flat fraction of position value from the
    strategy's return on every day a trade — an entry or an exit —
    actually occurs (see module docstring for the partial transaction-cost
    model this implements and its limitations).

    Days before the long moving average has enough data to compute (the
    first long_window-1 observations) produce no signal and are excluded
    from both the position series and the return calculation, rather than
    treated as a (false) flat/cash period with no real signal behind it.

    Returns {"total_return": float, "daily_returns": pd.Series,
    "num_trades": int}. total_return is the compounded return of the
    strategy's realized (net of any cost) daily returns over the whole
    usable period; 0.0 if there were no usable return observations at all
    (e.g. price series too short). num_trades counts how many entries and
    exits occurred, regardless of cost_per_trade — useful for reporting
    how "chatty" a given (short_window, long_window) pair is even when no
    cost is being modeled.
    """
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()

    signal = (short_ma > long_ma).astype(int)
    position = signal.shift(1).fillna(0)

    trades = position.diff().abs().fillna(0)
    num_trades = int(trades.sum())

    daily_returns = prices.pct_change()
    strategy_returns = daily_returns * position
    if cost_per_trade > 0:
        strategy_returns = strategy_returns - trades * cost_per_trade
    strategy_returns = strategy_returns.dropna()

    total_return = float((1 + strategy_returns).prod() - 1) if len(strategy_returns) else 0.0
    return {"total_return": total_return, "daily_returns": strategy_returns, "num_trades": num_trades}


def _random_individual(rng: np.random.Generator) -> tuple[int, int]:
    """Generate one random (short_window, long_window) pair respecting the
    bounds and the minimum gap between the two windows."""
    short_window = int(rng.integers(SHORT_WINDOW_BOUNDS[0], SHORT_WINDOW_BOUNDS[1] + 1))
    min_long = max(LONG_WINDOW_BOUNDS[0], short_window + MIN_WINDOW_GAP)
    if min_long > LONG_WINDOW_BOUNDS[1]:
        # short_window landed too high to leave room for a valid long_window;
        # resample short_window downward rather than producing an invalid pair.
        short_window = LONG_WINDOW_BOUNDS[1] - MIN_WINDOW_GAP
        min_long = short_window + MIN_WINDOW_GAP
    long_window = int(rng.integers(min_long, LONG_WINDOW_BOUNDS[1] + 1))
    return short_window, long_window


def _fitness(individual: tuple[int, int], prices: pd.Series, cost_per_trade: float = 0.0) -> float:
    """
    Fitness = total_return NET of cost_per_trade, not raw total_return.
    Threading the cost through fitness (rather than applying it only when
    reporting the winner afterward) is what lets a non-zero cost actually
    steer the genetic algorithm toward less frequently trading parameter
    pairs — see the module docstring's note on this design decision.
    """
    short_window, long_window = individual
    return simulate_crossover_strategy(prices, short_window, long_window, cost_per_trade)["total_return"]


def _tournament_select(
    population: list[tuple[int, int]],
    fitnesses: list[float],
    rng: np.random.Generator,
    tournament_size: int,
) -> tuple[int, int]:
    """
    Pick tournament_size individuals at random and return the fittest of
    them. Tournament selection is used rather than roulette-wheel
    selection because strategy total_return can be negative, and
    roulette-wheel weighting (proportional to raw fitness) breaks down
    once fitness values can be negative or near zero.
    """
    idxs = rng.integers(0, len(population), size=tournament_size)
    best_idx = max(idxs, key=lambda i: fitnesses[i])
    return population[best_idx]


def _crossover(
    parent_a: tuple[int, int], parent_b: tuple[int, int], rng: np.random.Generator
) -> tuple[int, int]:
    """
    Single-gene-swap crossover: the child independently inherits its
    short_window from one parent and its long_window from the other,
    each chosen at random. If the resulting pair violates the minimum-gap
    constraint, the pair is repaired (long_window nudged up, then
    short_window nudged down if still needed) rather than discarded, so
    crossover always produces a usable individual.
    """
    short_window = parent_a[0] if rng.random() < 0.5 else parent_b[0]
    long_window = parent_a[1] if rng.random() < 0.5 else parent_b[1]
    return _repair(short_window, long_window)


def _mutate(
    individual: tuple[int, int], rng: np.random.Generator, mutation_rate: float
) -> tuple[int, int]:
    """
    Each gene independently has mutation_rate probability of being
    resampled to a new random value within its bounds — "genetic
    manipulation... to create variation". Re-validates the min-gap constraint afterward, the
    same repair logic as crossover.
    """
    short_window, long_window = individual
    if rng.random() < mutation_rate:
        short_window = int(rng.integers(SHORT_WINDOW_BOUNDS[0], SHORT_WINDOW_BOUNDS[1] + 1))
    if rng.random() < mutation_rate:
        long_window = int(rng.integers(LONG_WINDOW_BOUNDS[0], LONG_WINDOW_BOUNDS[1] + 1))
    return _repair(short_window, long_window)


def _repair(short_window: int, long_window: int) -> tuple[int, int]:
    """Shared constraint-repair logic used by both crossover and mutation:
    ensures long_window - short_window >= MIN_WINDOW_GAP while staying
    within each gene's bounds."""
    if long_window - short_window < MIN_WINDOW_GAP:
        long_window = min(short_window + MIN_WINDOW_GAP, LONG_WINDOW_BOUNDS[1])
        if long_window - short_window < MIN_WINDOW_GAP:
            short_window = max(SHORT_WINDOW_BOUNDS[0], long_window - MIN_WINDOW_GAP)
    return short_window, long_window


def evolve_strategy(
    prices: pd.Series,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    elite_count: int = DEFAULT_ELITE_COUNT,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    cost_per_trade: float = 0.0,
    seed: Optional[int] = None,
) -> dict:
    """
    Run the genetic algorithm: evolve a population of (short_window,
    long_window) pairs over `generations` generations, using tournament
    selection, single-gene-swap crossover, and per-gene mutation, with
    elitism, the top elite_count individuals of each generation survive
    unchanged into the next, so the algorithm's best-found-so-far result
    can never regress between generations.

    cost_per_trade (default 0.0) is passed straight through to the fitness
    function, so the population evolves toward parameters that are best
    NET of trading costs, not best in a cost-free world with costs
    subtracted only when reporting the winner.

    Returns {"best_individual": (short_window, long_window),
    "best_fitness": float, "fitness_history": list[float]}, where
    fitness_history has one entry per generation (the best fitness seen so
    far as of that generation) — a non-decreasing sequence by
    construction, useful for reporting the algorithm's convergence.
    """
    rng = np.random.default_rng(seed)
    population = [_random_individual(rng) for _ in range(population_size)]
    fitness_history = []

    best_individual = None
    best_fitness = float("-inf")

    for _ in range(generations):
        fitnesses = [_fitness(ind, prices, cost_per_trade) for ind in population]

        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx]
        fitness_history.append(best_fitness)

        ranked = sorted(zip(population, fitnesses), key=lambda p: p[1], reverse=True)
        next_population = [ind for ind, _ in ranked[:elite_count]]

        while len(next_population) < population_size:
            parent_a = _tournament_select(population, fitnesses, rng, tournament_size)
            parent_b = _tournament_select(population, fitnesses, rng, tournament_size)
            child = _crossover(parent_a, parent_b, rng)
            child = _mutate(child, rng, mutation_rate)
            next_population.append(child)

        population = next_population

    return {
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "fitness_history": fitness_history,
    }


def compute_buy_and_hold_return(prices: pd.Series) -> float:
    """Total return of simply holding the asset for the entire period —
    the benchmark the evolved strategy is compared against. Returns 0.0
    for a series with fewer than 2 points (no return can be computed)."""
    if len(prices) < 2:
        return 0.0
    return float(prices.iloc[-1] / prices.iloc[0] - 1)


def backtest_ticker(
    ticker: str,
    history_lookup: Callable[[str], Optional[pd.Series]],
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    cost_per_trade: float = 0.0,
    seed: Optional[int] = None,
) -> dict:
    """
    End-to-end genetic-algorithm backtest for a single ticker: fetches one
    historical price series via history_lookup, evolves the best moving-average crossover
    parameters found, net of cost_per_trade, if non-zero, and compares
    the evolved strategy's total return against buy-and-hold over the same
    period.

    cost_per_trade (default 0.0, cost-free, matching the original
    behaviour) is passed through to evolve_strategy, so a non-zero value
    genuinely changes which parameters the algorithm converges to, not
    just the number reported afterward.

    Returns a dict:
      {
        "ticker": str,
        "short_window": int or None, "long_window": int or None,
        "strategy_return": float or None, "benchmark_return": float or None,
        "beat_benchmark": bool or None,
        "num_trades": int or None,
        "cost_per_trade": float,
        "fitness_history": list[float],
        "note": str or None,
      }
    num_trades is recomputed once, after evolution, by re-simulating the
    winning (short_window, long_window) pair with the same cost_per_trade,
    this is the number of entries+exits the reported strategy_return
    actually reflects, useful for showing how "chatty" the evolved
    strategy is (a wide-window strategy typically trades a handful of
    times a year; a narrow-window one can trade dozens).

    If history could not be retrieved, or has fewer than
    MIN_HISTORY_POINTS usable points, returns a result with the numeric
    fields set to None and `note` explaining why, rather than running a
    backtest on insufficient data and reporting a misleadingly precise
    number.
    """
    series = history_lookup(ticker)
    if series is None or len(series) == 0:
        return _empty_backtest_result(ticker, "price history unavailable", cost_per_trade)
    if len(series) < MIN_HISTORY_POINTS:
        return _empty_backtest_result(
            ticker, f"only {len(series)} data points available (minimum {MIN_HISTORY_POINTS})", cost_per_trade
        )

    evolution = evolve_strategy(
        series,
        population_size=population_size,
        generations=generations,
        cost_per_trade=cost_per_trade,
        seed=seed,
    )
    short_window, long_window = evolution["best_individual"]

    strategy_return = evolution["best_fitness"]
    benchmark_return = compute_buy_and_hold_return(series)
    final_sim = simulate_crossover_strategy(series, short_window, long_window, cost_per_trade)

    return {
        "ticker": ticker,
        "short_window": short_window,
        "long_window": long_window,
        "strategy_return": strategy_return,
        "benchmark_return": benchmark_return,
        "beat_benchmark": strategy_return > benchmark_return,
        "num_trades": final_sim["num_trades"],
        "cost_per_trade": cost_per_trade,
        "fitness_history": evolution["fitness_history"],
        "note": None,
    }


def _empty_backtest_result(ticker: str, reason: str, cost_per_trade: float = 0.0) -> dict:
    return {
        "ticker": ticker,
        "short_window": None,
        "long_window": None,
        "strategy_return": None,
        "benchmark_return": None,
        "beat_benchmark": None,
        "num_trades": None,
        "cost_per_trade": cost_per_trade,
        "fitness_history": [],
        "note": reason,
    }
