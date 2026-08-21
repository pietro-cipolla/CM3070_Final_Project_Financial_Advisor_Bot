"""
Financial Advisor Bot - Feature Prototype
Iteration 2: migrates the UI to the wide/sidebar "FULL" layout, adds a
Plotly price chart with a 20-day moving average (MA20) for single-ticker
queries, and adds real news headlines via NewsAPI (replacing the more
limited yfinance-bundled headlines used in Iteration 1).
Iteration 3: adds SQLite-backed conversation memory (resumable via a
session ID) and a portfolio tracker, plus three inclusive-design
improvements — colourblind-safe chart colours, an optional simplified
explanation mode, and always-on language-matching in responses.
Iteration 4: adds VADER-based sentiment icons on news headlines, a
genetic-algorithm-optimized moving-average-crossover backtest shown
alongside the price chart, and a Markowitz (MPT) mean-variance
optimization panel that suggests rebalancing weights for the tracked
portfolio, three algorithmic components (language-model reasoning,
a genetic algorithm, and mean-variance optimization) grounding the
template's "active portfolio management" language in real,
independently testable code.
"""

from dotenv import load_dotenv
load_dotenv()

import hashlib
import uuid
from datetime import date

import streamlit as st
import plotly.graph_objects as go

from src.financial_data import (
    get_stock_summary,
    get_multiple_stock_summaries,
    get_price_history,
    get_current_price,
    get_closing_prices,
)
from src.rag_pipeline import classify_query_intent, extract_tickers_with_truncation_info, build_prompt, MAX_TICKERS
from src.advisor import get_advice
from src.news_data import get_news_for_company, build_news_context, score_news_sentiment
from src.database import (
    init_db,
    save_message,
    load_conversation,
    clear_conversation,
    add_holding,
    get_portfolio,
    remove_holding,
)
from src.portfolio import compute_portfolio_summary
from src.backtesting import backtest_ticker, DEFAULT_COST_PER_TRADE
from src.optimizer import compute_portfolio_optimization


def escape_dollars(text: str) -> str:
    """
   Streamlit's markdown renderer treats a pair of '$' as LaTeX math
    delimiters. LLM responses routinely mention two or more dollar amounts
    in the same paragraph, which Streamlit then renders as a single garbled math
    block instead of plain text.

    A backslash escape ("\\$") is not enough, Streamlit's math-detection
    still pairs up escaped dollar signs and swallows everything between
    them. Replacing '$' with the HTML entity '&#36;' sidesteps this: the
    raw '$' character never appears in the text Streamlit scans for math
    delimiters, but the browser still renders the entity as a normal '$'.
    """
    return text.replace("$", "&#36;")


@st.cache_data(ttl=60, show_spinner=False)
def _cached_current_price(ticker: str):
    return get_current_price(ticker)


# Iteration 4: both the backtester and the MPT optimizer need historical
# closing prices for the same tickers within a single run,
# cached so the two features never double-fetch from yfinance
# for the same ticker. A longer TTL than the current-price cache is
# appropriate: a full year of historical closes does not meaningfully
# change within a 15 minute window.
@st.cache_data(ttl=900, show_spinner=False)
def _cached_closing_prices(ticker: str):
    return get_closing_prices(ticker)


def _stable_seed_for_ticker(ticker: str) -> int:
    """
    Deterministic seed derived from the ticker symbol, so the genetic
    algorithm's random population initialization is reproducible across
    reruns and sessions for a given ticker, without hardcoding one single
    seed for every ticker (which would make every ticker's GA start from
    an identical population before the price data even differs them).

    This exists purely for reproducibility, not to cherry-pick a seed
    that happens to make the strategy look good, the seed is a fixed,
    non-adjustable function of the ticker string alone, so there is no
    opportunity to search for a flattering outcome.
    """
    digest = hashlib.sha256(ticker.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**32)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_backtest(ticker: str):
    """
    Cache the full genetic-algorithm backtest per ticker (not just the
    underlying price history): the GA itself (30 individuals x 20
    generations) is the expensive part, not the yfinance call, so caching
    only _cached_closing_prices left the GA re-evolving from scratch on
    every call. Combined with _stable_seed_for_ticker, this also means the
    result shown to the user for a given ticker is stable within the TTL,
    not just fast.
    """
    return backtest_ticker(
        ticker,
        get_closing_prices,
        cost_per_trade=DEFAULT_COST_PER_TRADE,
        seed=_stable_seed_for_ticker(ticker),
    )


# Iteration 3, inclusive design improvement 
CHART_COLOR_CLOSE = "#0072B2"  
CHART_COLOR_MA20 = "#E69F00"   


MIN_CHART_POINTS = 5  # Problema 31: below this, a line chart is not meaningful


def render_price_chart(ticker: str) -> None:
    """
    Render a Plotly line chart of closing price + 20-day moving average
    (MA20) for the last 3 months. Silently renders nothing if history could
    not be retrieved, so a charting failure never blocks the rest of the
    response.

    Problema 31 (Iteration 4, Sezione 4): when the retrieved history is
    down to 1-2 points (e.g. a very recently listed or thinly-traded
    ticker), Plotly's auto-detected x-axis tick granularity could fall back
    to a sub-second format ("23:59:59.999") instead of a date, because the
    axis was never given an explicit tickformat and had to guess one from
    an almost-degenerate date range. Two independent fixes: (a) an explicit
    day-level tickformat, so the axis never guesses a sub-day granularity
    regardless of how narrow the date range is; (b) a minimum point count
    below which the chart isn't rendered at all — a 1-2 point "line" chart
    isn't informative even with a fixed axis, so a caption explaining why
    is more honest than a chart that looks like real data.
    """
    hist = get_price_history(ticker)
    if hist is None or hist.empty:
        return
    if len(hist) < MIN_CHART_POINTS:
        st.caption(
            f"Not enough price history to chart {ticker} "
            f"({len(hist)} data point{'s' if len(hist) != 1 else ''} available, "
            f"minimum {MIN_CHART_POINTS})."
        )
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close price", mode="lines",
                              line=dict(color=CHART_COLOR_CLOSE)))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA20"], name="20-day MA", mode="lines",
                              line=dict(dash="dash", color=CHART_COLOR_MA20)))
    fig.update_layout(
        title=f"{ticker} — last 3 months",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickformat="%b %d"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_backtest(ticker: str) -> None:
    """
    Render the genetic-algorithm moving-average-crossover backtest result
    for a single ticker (Iteration 4, src/backtesting.py). Reports the
    evolved strategy's total return against buy-and-hold honestly in both
    directions, never hiding an underperforming result. Silently renders
    nothing if history could not be retrieved or is too short, same
    fail-soft convention as render_price_chart, so a backtest failure
    never blocks the rest of the response.
    """
    result = _cached_backtest(ticker)
    if result["note"] is not None:
        st.caption(f"Backtest unavailable for {ticker}: {result['note']}")
        return

    indicator = "🟢" if result["beat_benchmark"] else "🔴"
    st.write(
        f"Evolved strategy: **{result['short_window']}/{result['long_window']}-day** "
        f"moving-average crossover ({result['num_trades']} trades)"
    )
    st.write(
        f"Strategy return: **{result['strategy_return']*100:.1f}%** vs. "
        f"buy-and-hold: **{result['benchmark_return']*100:.1f}%** "
        f"{indicator} {'beat' if result['beat_benchmark'] else 'underperformed'} the benchmark"
    )
    st.caption(
        "Genetic algorithm (tournament selection, crossover, mutation) evolved over "
        "20 generations on 1 year of historical data, optimizing for return net of an "
        f"estimated {result['cost_per_trade']*100:.2f}% cost per trade (a flat approximation, "
        "not a real broker's actual spread/commission/slippage). "
        "Past performance does not guarantee future results — this is not financial advice."
    )


# Iteration 4: sentiment icon shown next to each headline. Colour-coded
# rather than red/green-only, consistent with the colourblind-safe design
# principle already applied to the price chart in Iteration 3 (a shape/hue
# combination — traffic-light colours plus distinct positions in the UI —
# rather than relying on red/green alone to carry meaning).
SENTIMENT_ICONS = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}


def render_news(news_items: list[dict], ticker: str) -> None:
    """
    Render a small news headlines list with clickable source links and a
    VADER sentiment icon per headline (Iteration 4).

    Does not render its own "Recent news" title: callers wrap this in an
    st.expander(f"... — {ticker}") that already carries that heading, and
    printing it again here produced a visibly duplicated title in the UI
    (Iteration 2, Problema 14).
    """
    if not news_items:
        return
    scored_items = score_news_sentiment(news_items)
    for item in scored_items:
        date = item["published_at"][:10] if item.get("published_at") else ""
        icon = SENTIMENT_ICONS.get(item.get("sentiment_label"), "⚪")
        st.markdown(f"{icon} [{item['title']}]({item['url']}) — *{item['source']}, {date}*")


# Page config
st.set_page_config(page_title="Financial Advisor Bot", page_icon="📈", layout="wide")

# Database init
init_db()

# Session state: session ID and conversation memory
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "loaded_session_id" not in st.session_state:
    st.session_state.loaded_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("📈 Financial Advisor Bot")
    st.caption(
        "Ask a question about one or more publicly traded stocks (up to 3). "
        "Data from Yahoo Finance, news from NewsAPI, analysis from an LLM."
    )
    st.divider()

    st.subheader("🧠 Session memory")
    st.caption(
        "Save this ID to resume this conversation and portfolio later on the "
        "same device. On the hosted demo, memory may not survive an app restart."
    )
    session_input = st.text_input("Session ID", value=st.session_state.session_id).strip()
    
    if session_input and session_input != st.session_state.session_id:
        st.session_state.session_id = session_input
        st.session_state.loaded_session_id = None  # force a reload below

    if st.session_state.loaded_session_id != st.session_state.session_id:
        st.session_state.messages = load_conversation(st.session_state.session_id)
        st.session_state.loaded_session_id = st.session_state.session_id

    if st.button("🗑️ Clear conversation"):
        clear_conversation(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

    st.divider()
    simplified_mode = st.checkbox(
        "🔤 Simplified explanations",
        value=False,
        help="Reduce financial jargon and briefly define technical terms when they can't be avoided.",
    )

    st.divider()
    st.subheader("💼 Portfolio tracker")
    with st.form("add_holding_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            new_ticker = st.text_input("Ticker", key="new_ticker")
            new_shares = st.number_input("Shares", min_value=0.0001, value=1.0, step=1.0, format="%.4f")
        with col2:
            new_price = st.number_input("Purchase price ($)", min_value=0.01, value=1.0, step=1.0, format="%.2f")
            # max_value=today
            new_date = st.date_input("Purchase date", value=date.today(), max_value=date.today())
        submitted = st.form_submit_button("➕ Add holding")
        if submitted:
            ticker_clean = new_ticker.strip().upper()
            if not ticker_clean:
                st.warning("Enter a ticker symbol.")
            else:
                try:
                    add_holding(
                        st.session_state.session_id,
                        ticker_clean,
                        new_shares,
                        new_price,
                        str(new_date),
                    )
                    st.success(f"Added {new_shares:g} shares of {ticker_clean}.")
                except ValueError as e:
                    st.error(f"Could not add holding: {e}")

    holdings = get_portfolio(st.session_state.session_id)
    if holdings:
        with st.spinner("Updating portfolio value..."):
            summary = compute_portfolio_summary(holdings, _cached_current_price)

        for h in summary["holdings"]:
            st.write(f"**{h['ticker']}** — {h['shares']:g} sh @ ${h['purchase_price']:.2f}")
            if h["pnl"] is None:
                st.caption("⚠️ Current price unavailable — check the ticker symbol.")
            else:
                indicator = "🟢" if h["pnl"] >= 0 else "🔴"
                st.caption(
                    f"Current: ${h['current_price']:.2f} · "
                    f"P&L: {indicator} ${h['pnl']:.2f} ({h['pnl_pct']:.1f}%)"
                )
            if st.button("Remove", key=f"remove_holding_{h['id']}"):
                remove_holding(st.session_state.session_id, h["id"])
                st.rerun()

        st.divider()
        if summary["total_pnl_pct"] is not None:
            indicator = "🟢" if summary["total_pnl"] >= 0 else "🔴"
            st.write(
                f"**Total P&L: {indicator} ${summary['total_pnl']:.2f} "
                f"({summary['total_pnl_pct']:.1f}%)**"
            )
        if summary["unpriced_tickers"]:
            st.caption(f"Excluded from totals (price unavailable): {', '.join(summary['unpriced_tickers'])}")

        st.divider()
        with st.expander("🎯 Suggested rebalancing"):
            with st.spinner("Optimizing portfolio weights..."):
                opt = compute_portfolio_optimization(
                    holdings,
                    history_lookup=_cached_closing_prices,
                    price_lookup=_cached_current_price,
                )
            if opt["note"]:
                st.caption(opt["note"])
            if opt["suggested_weights"]:
                for ticker, w in sorted(opt["suggested_weights"].items(), key=lambda x: -x[1]):
                    current = (opt["current_weights"] or {}).get(ticker)
                    current_str = f" (currently {current*100:.1f}%)" if current is not None else ""
                    st.write(f"**{ticker}**: {w*100:.1f}%{current_str}")
                if opt["sharpe_ratio"] is not None:
                    st.caption(
                        f"Expected annual return: {opt['expected_return']*100:.1f}% · "
                        f"Expected annual volatility: {opt['expected_volatility']*100:.1f}% · "
                        f"Sharpe ratio: {opt['sharpe_ratio']:.2f}"
                    )
            if opt.get("low_confidence_tickers"):
                for t, reason in opt["low_confidence_tickers"].items():
                    st.caption(f"⚠️ Low-confidence estimate for {t}: {reason}")
            if opt["excluded_tickers"]:
                for t, reason in opt["excluded_tickers"].items():
                    st.caption(f"⚠️ {t} excluded: {reason}")
            st.caption(
                "Long-only allocation based on 1 year of historical returns. "
                "Past performance does not guarantee future results — this is "
                "not financial advice."
            )
    else:
        st.caption("No holdings yet. Add one above to start tracking your portfolio.")

    st.divider()
    st.caption(
        "⚠️ This tool is for educational purposes only and does not "
        "constitute financial advice. Always consult a qualified financial "
        "advisor before making investment decisions."
    )

st.title("📈 Financial Advisor Bot")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_query = st.chat_input(
    "e.g. Should I buy Apple stock? Compare Tesla and Ford."
)

if user_query:
    # Show user message
    with st.chat_message("user"):
        st.write(user_query)
    # Problema 27 (Iteration 4, Sezione 4): snapshot the conversation as it
    # stood BEFORE this turn's user message, so it can be passed as
    # "history" to the LLM calls below without the current query
    # duplicating itself inside its own history.
    recent_history = st.session_state.messages[-6:]
    st.session_state.messages.append({"role": "user", "content": user_query})
    save_message(st.session_state.session_id, "user", user_query)

    with st.chat_message("assistant"):
        with st.spinner("Understanding your question..."):
            intent = classify_query_intent(user_query, history=recent_history)

        if intent == "unclear":
            response = (
                "I'm not sure what you're asking. Could you rephrase your question "
                "and mention a company name or ticker symbol? "
                "For example: *'What is Tesla's P/E ratio?'* or *'Compare Apple and Microsoft.'*"
            )
            st.write(response)

        elif intent == "open_ended":
            response = (
                "That's a broad question — to give you a grounded answer I need to know "
                "which company or companies you're interested in. "
                "Could you name a specific stock (e.g. *'Should I buy Apple?'*), "
                "or up to three to compare (e.g. *'Compare Apple, Microsoft and Google'*)?"
            )
            st.write(response)

        elif intent == "portfolio_query":
            # Problema 26 (Iteration 4, Sezione 4): reuses the same
            # compute_portfolio_summary() already used by the sidebar
            # Portfolio Tracker widget, so a portfolio-level question asked
            # in chat and the sidebar numbers can never disagree — one
            # summary, two places it's shown.
            holdings = get_portfolio(st.session_state.session_id)
            if not holdings:
                response = (
                    "You don't have any holdings tracked yet. Add one from the "
                    "**💼 Portfolio tracker** panel in the sidebar, then ask me "
                    "again — for example *'How is my portfolio doing?'*"
                )
                st.write(response)
            else:
                with st.spinner("Checking your portfolio..."):
                    summary = compute_portfolio_summary(holdings, _cached_current_price)

                with st.expander("💼 Your portfolio", expanded=True):
                    for h in summary["holdings"]:
                        st.write(f"**{h['ticker']}** — {h['shares']:g} sh @ ${h['purchase_price']:.2f}")
                        if h["pnl"] is None:
                            st.caption("⚠️ Current price unavailable — check the ticker symbol.")
                        else:
                            indicator = "🟢" if h["pnl"] >= 0 else "🔴"
                            st.caption(
                                f"Current: ${h['current_price']:.2f} · "
                                f"P&L: {indicator} ${h['pnl']:.2f} ({h['pnl_pct']:.1f}%)"
                            )
                    if summary["unpriced_tickers"]:
                        st.caption(
                            f"Excluded from totals (price unavailable): "
                            f"{', '.join(summary['unpriced_tickers'])}"
                        )

                if summary["total_pnl_pct"] is not None:
                    indicator = "🟢" if summary["total_pnl"] >= 0 else "🔴"
                    response = (
                        f"Your portfolio ({len(summary['holdings'])} holding"
                        f"{'s' if len(summary['holdings']) != 1 else ''}) is at "
                        f"{indicator} **${summary['total_pnl']:.2f} "
                        f"({summary['total_pnl_pct']:.1f}%)** total P&L. "
                        "See the breakdown above, or check the **🎯 Suggested "
                        "rebalancing** panel in the sidebar for allocation advice. "
                        "This is not financial advice."
                    )
                else:
                    response = (
                        "I couldn't compute a total P&L for your portfolio right now "
                        "— see the per-holding breakdown above for what's available."
                    )
                st.write(response)

        else:  # intent == "stock_query"
            with st.spinner("Retrieving financial data..."):
                tickers, truncated = extract_tickers_with_truncation_info(user_query, history=recent_history)

                if truncated:
                    st.info(
                        f"You mentioned more than {MAX_TICKERS} companies — this tool "
                        f"compares up to {MAX_TICKERS} at a time. Comparing: "
                        f"{', '.join(tickers)}."
                    )

                if not tickers:
                    response = (
                        "I could not identify a stock ticker in your query. "
                        "Please mention a company name or ticker symbol, "
                        "for example: *'Tell me about Apple'* or *'What is TSLA's P/E ratio?'*"
                    )
                    st.write(response)

                elif len(tickers) == 1:
                    stock_data = get_stock_summary(tickers[0])

                    if "error" in stock_data:
                        response = (
                            f"Could not retrieve data for **{tickers[0]}**: {stock_data['error']}. "
                            "Please check the ticker symbol and try again."
                        )
                        st.write(response)
                    else:
                        with st.expander(f"📊 Data retrieved for {tickers[0]}", expanded=True):
                            # Problema 29 (Iteration 4, Sezione 4): this panel used to
                            # show a fixed subset of 8 fields, while build_data_context()
                            # (financial_data.py) already puts dividend yield, beta and
                            # sector into the same context block the model answers from
                            # — so the model could correctly cite a real, grounded figure
                            # (e.g. "dividend yield of 0.61%") that the user had no way to
                            # verify here, since it simply wasn't shown. Every field the
                            # model can cite should have a visible check in the same place.
                            dividend_display = (
                                f"{round(stock_data['dividend_yield'], 2)}%"
                                if stock_data.get("dividend_yield")
                                else "None"
                            )
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Company:** {stock_data.get('name', tickers[0])}")
                                st.write(f"**Current price:** ${stock_data.get('price', 'N/A')}")
                                st.write(f"**Day change:** {stock_data.get('change_pct', 'N/A')}%")
                                st.write(f"**52-week range:** {stock_data.get('52_week_range', 'N/A')}")
                                st.write(f"**Sector:** {stock_data.get('sector', 'N/A')}")
                            with col2:
                                st.write(f"**P/E ratio:** {stock_data.get('pe_ratio', 'N/A')}")
                                st.write(f"**EPS:** {stock_data.get('eps', 'N/A')}")
                                st.write(f"**Dividend yield:** {dividend_display}")
                                st.write(f"**Beta:** {stock_data.get('beta', 'N/A')}")
                                st.write(f"**Analyst rating:** {stock_data.get('recommendation', 'N/A')}")
                                st.write(f"**Target price:** ${stock_data.get('target_price', 'N/A')}")
                            st.caption(f"Data retrieved at: {stock_data.get('timestamp', 'N/A')}")

                            render_price_chart(tickers[0])

                        with st.expander(f"🧬 Strategy backtest — {tickers[0]}", expanded=False):
                            with st.spinner("Evolving strategy parameters..."):
                                render_backtest(tickers[0])

                        with st.spinner("Fetching recent news..."):
                            news_items = get_news_for_company(stock_data.get("name"), tickers[0])
                        if news_items:
                            with st.expander(f"📰 Recent news — {tickers[0]}", expanded=False):
                                render_news(news_items, tickers[0])
                        news_context = build_news_context(news_items)

                        messages = build_prompt(
                            stock_data, user_query, news_context=news_context,
                            simplified_mode=simplified_mode, history=recent_history,
                        )
                        response = escape_dollars(get_advice(messages))
                        st.write(response)

                else:  # multi-ticker comparative path (2 or 3 tickers)
                    stock_data_list = get_multiple_stock_summaries(tickers)
                    valid = [d for d in stock_data_list if "error" not in d]
                    failed = [d for d in stock_data_list if "error" in d]

                    with st.expander(f"📊 Data retrieved for {', '.join(tickers)}", expanded=True):
                        for stock_data in valid:
                            st.write(f"**{stock_data['ticker']} — {stock_data.get('name', '')}**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"Price: ${stock_data.get('price', 'N/A')}")
                                st.write(f"Day change: {stock_data.get('change_pct', 'N/A')}%")
                            with col2:
                                st.write(f"P/E ratio: {stock_data.get('pe_ratio', 'N/A')}")
                                st.write(f"Analyst rating: {stock_data.get('recommendation', 'N/A')}")
                            st.divider()
                        for stock_data in failed:
                            st.warning(f"⚠️ Could not retrieve data for {stock_data['ticker']}: {stock_data['error']}")

                    if not valid:
                        response = (
                            "I could not retrieve data for any of the requested tickers "
                            f"({', '.join(tickers)}). Please check the symbols and try again."
                        )
                        st.write(response)
                    else:
                        all_news_items = []
                        with st.spinner("Fetching recent news..."):
                            for stock_data in valid:
                                items = get_news_for_company(stock_data.get("name"), stock_data["ticker"])
                                all_news_items.extend(items)
                                if items:
                                    with st.expander(f"📰 Recent news — {stock_data['ticker']}", expanded=False):
                                        render_news(items, stock_data["ticker"])
                        news_context = build_news_context(all_news_items)

                        messages = build_prompt(
                            stock_data_list, user_query, news_context=news_context,
                            simplified_mode=simplified_mode, history=recent_history,
                        )
                        response = escape_dollars(get_advice(messages))
                        st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        save_message(st.session_state.session_id, "assistant", response)
