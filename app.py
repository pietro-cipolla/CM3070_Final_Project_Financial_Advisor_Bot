"""
Financial Advisor Bot — Feature Prototype
Preliminary Report submission — simplified version
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # Loads OPENAI_API_KEY from .env file

from src.financial_data import get_stock_summary
from src.rag_pipeline import extract_ticker_from_query, build_prompt
from src.advisor import get_advice

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Financial Advisor Bot", page_icon="📈")
st.title("📈 Financial Advisor Bot")
st.caption("Ask a question about any publicly traded stock.")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display conversation history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_query = st.chat_input("e.g. Should I buy Apple stock? What is Tesla's P/E ratio?")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving financial data..."):

            # Step 1 — Extract ticker
            ticker = extract_ticker_from_query(user_query)

            if not ticker:
                response = (
                    "I could not identify a stock ticker in your query. "
                    "Please mention a company name or ticker symbol, "
                    "for example: *'Tell me about Apple'* or *'What is TSLA's P/E ratio?'*"
                )
                st.write(response)
            else:
                # Step 2 — Retrieve financial data
                stock_data = get_stock_summary(ticker)

                if "error" in stock_data:
                    response = (
                        f"Could not retrieve data for **{ticker}**: {stock_data['error']}. "
                        "Please check the ticker symbol and try again."
                    )
                    st.write(response)
                else:
                    # Step 3 — Show raw data retrieved (transparency)
                    with st.expander(f"📊 Data retrieved for {ticker}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Company:** {stock_data.get('name', ticker)}")
                            st.write(f"**Current price:** ${stock_data.get('price', 'N/A')}")
                            st.write(f"**Day change:** {stock_data.get('change_pct', 'N/A')}%")
                            st.write(f"**52-week range:** {stock_data.get('52_week_range', 'N/A')}")
                        with col2:
                            st.write(f"**P/E ratio:** {stock_data.get('pe_ratio', 'N/A')}")
                            st.write(f"**EPS:** {stock_data.get('eps', 'N/A')}")
                            st.write(f"**Analyst rating:** {stock_data.get('recommendation', 'N/A')}")
                            st.write(f"**Target price:** ${stock_data.get('target_price', 'N/A')}")
                        st.caption(f"Data retrieved at: {stock_data.get('timestamp', 'N/A')}")

                    # Step 4 — Build RAG prompt and call LLM
                    messages = build_prompt(stock_data, user_query)
                    response = get_advice(messages)
                    st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This tool is for educational purposes only and does not constitute "
    "financial advice. Always consult a qualified financial advisor before "
    "making investment decisions."
)
