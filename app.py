"""
Financial Advisor Bot — Feature Prototype
Iteration 1: adds query intent classification, a clarification flow for
open-ended/unclear queries, and multi-ticker (up to 3) comparative retrieval.
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.financial_data import get_stock_summary, get_multiple_stock_summaries
from src.rag_pipeline import classify_query_intent, extract_tickers_with_truncation_info, build_prompt, MAX_TICKERS
from src.advisor import get_advice


def escape_dollars(text: str) -> str:
    """
    Streamlit's markdown renderer treats a pair of '$' as LaTeX math
    delimiters. LLM responses routinely mention two or more dollar amounts
    in the same paragraph (e.g. "target price of $423.40 ... current price
    of $419.77"), which Streamlit then renders as a single garbled math
    block instead of plain text.

    A backslash escape ("\\$") is NOT enough — Streamlit's math-detection
    still pairs up escaped dollar signs and swallows everything between
    them. Replacing '$' with the HTML entity '&#36;' sidesteps this: the
    raw '$' character never appears in the text Streamlit scans for math
    delimiters, but the browser still renders the entity as a normal '$'.
    """
    return text.replace("$", "&#36;")


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Financial Advisor Bot", page_icon="📈")
st.title("📈 Financial Advisor Bot")
st.caption("Ask a question about one or more publicly traded stocks (up to 3).")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display conversation history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_query = st.chat_input(
    "e.g. Should I buy Apple stock? Compare Tesla and Ford."
)

if user_query:
    # Show user message
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Understanding your question..."):
            intent = classify_query_intent(user_query)

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

        else:  # intent == "stock_query"
            with st.spinner("Retrieving financial data..."):
                tickers, truncated = extract_tickers_with_truncation_info(user_query)

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
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Company:** {stock_data.get('name', tickers[0])}")
                                st.write(f"**Current price:** ${stock_data.get('price', 'N/A')}")
                                st.write(f"**Day change:** {stock_data.get('change_pct', 'N/A')}%")
                                st.write(f"**52-week range:** {stock_data.get('52_week_range', 'N/A')}")
                            with col2:
                                st.write(f"**P/E ratio:** {stock_data.get('pe_ratio', 'N/A')}")
                                st.write(f"**EPS:** {stock_data.get('eps', 'N/A')}")
                                st.write(f"**Analyst rating:** {stock_data.get('recommendation', 'N/A')}")
                                st.write(f"**Target price:** ${stock_data.get('target_price', 'N/A')}")
                            st.caption(f"Data retrieved at: {stock_data.get('timestamp', 'N/A')}")

                        messages = build_prompt(stock_data, user_query)
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
                        messages = build_prompt(stock_data_list, user_query)
                        response = escape_dollars(get_advice(messages))
                        st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This tool is for educational purposes only and does not constitute "
    "financial advice. Always consult a qualified financial advisor before "
    "making investment decisions."
)
