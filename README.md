# Financial Advisor Bot

**University of London — BSc Computer Science — CM3070 Final Project**

---

## Overview

A conversational AI assistant that helps retail investors understand financial data and make more informed investment decisions. The system uses a **Retrieval-Augmented Generation (RAG)** architecture: it retrieves live market data, news, and (optionally) genetic-algorithm backtests and Modern Portfolio Theory optimisation for any publicly traded stock, and uses a Large Language Model (GPT-4o-mini) to reason over that data and generate plain-language, data-grounded analysis.

The interface is a browser-based chat application built with Streamlit.

**Live demo:** _add your Streamlit Community Cloud URL here_

---

## Project Status

This repository tracks the development of the system across Phase 2 of the project (July–September 2026).

| Iteration | Description | Status |
|---|---|---|
| 0 - Baseline | Simple prototype from preliminary report | ✅ Done |
| 1 - Multi-ticker + Intent detection | Handle multi-stock queries and open-ended questions | ✅ Done |
| 2 - NewsAPI + Visualisation | Real-time news and interactive price charts | ✅ Done |
| 3 - Memory + Portfolio tracker | SQLite persistence and portfolio P&L | ✅ Done |
| 4 - Sentiment + Backtesting + Ticker resolution | VADER sentiment analysis, genetic-algorithm backtesting, MPT portfolio optimisation, and non-US ticker resolution hardening | ✅ Done |
| 5 - Deployment + User testing | Streamlit Community Cloud deployment; 5-participant study, 20-query evaluation | ✅ Done |
| 6 - Final polish | Documentation, refactoring, submission prep | ✅ Done |

---

## Architecture

```
User query
    │
    ▼
Query Intent Classification (LLM, temperature=0)
    │  stock_query / open_ended / unclear
    ▼
Ticker Extraction + Resolution (LLM guess → yfinance.Search() fallback)
    │  single ticker or up to 3, for comparisons
    ▼
Data Retrieval (yfinance, NewsAPI) ──── SQLite (conversation memory, portfolio)
    │
    ▼
RAG Prompt Construction (context + query)
    │  + optional: sentiment score, GA backtest, MPT optimisation
    ▼
LLM Reasoning (GPT-4o-mini)
    │
    ▼
Response + Transparency Panel + Disclaimer (Streamlit UI)
```

### File structure

```
CM3070_Final_Project_Financial_Advisor_Bot/
├── app.py                  # Streamlit entry point
├── conftest.py             # Shared pytest fixtures
├── src/
│   ├── __init__.py
│   ├── financial_data.py   # Data retrieval layer (yfinance) + ticker resolution
│   ├── rag_pipeline.py     # RAG pipeline: intent classification, ticker extraction, prompt construction
│   ├── advisor.py          # LLM reasoning layer (OpenAI API)
│   ├── news_data.py        # NewsAPI retrieval + relevance filtering (Iteration 2)
│   ├── database.py         # SQLite persistence: conversation memory + portfolio (Iteration 3)
│   ├── portfolio.py        # Portfolio P&L calculation (Iteration 3)
│   ├── finance_lexicon.py  # Finance-specific sentiment lexicon override (Iteration 4)
│   ├── backtesting.py      # Genetic-algorithm strategy backtester (Iteration 4)
│   └── optimizer.py        # Mean-variance (MPT) portfolio optimiser (Iteration 4)
├── tests/                  # pytest unit and integration tests, one file per iteration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── .gitignore
```

---

## Setup

### Requirements
- Python 3.10 or higher
- An OpenAI API key
- A NewsAPI.org API key (for live news and sentiment analysis, added in Iteration 2)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/pietro-cipolla/CM3070_Final_Project_Financial_Advisor_Bot.git
cd CM3070_Final_Project_Financial_Advisor_Bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API keys
cp .env.example .env
# Then edit .env and add your OpenAI and NewsAPI keys
# On Mac/Linux you can also use:
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
echo "NEWSAPI_KEY=your-newsapi-key-here" >> .env

# 4. Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Usage

Type any question about one or more publicly traded stocks in the chat input. Examples:

- "What is Tesla's P/E ratio?" a single-ticker query, shown alongside a live price chart and a transparency panel with the underlying data.
- "Compare Tesla and Ford" a multi-ticker comparison, with figures clearly attributed to each company.
- "What's the latest sentiment on Novartis?" live news retrieval with a three-colour sentiment indicator.
- "Backtest NVS over the past year" a genetic-algorithm-evolved trading strategy compared against buy-and-hold.
- "What's a good tech stock to buy right now?" an open-ended query, which prompts for a sector, market-cap range or specific company rather than answering from unsupported parametric knowledge.

You can also track real holdings in the **Portfolio Tracker** tab and reload a previous conversation and portfolio at any time by pasting its **Session ID** into the sidebar, both are persisted locally via SQLite.

The bot retrieves live data from Yahoo Finance and NewsAPI.org and provides analysis grounded in that data, never from the model's own training knowledge alone.

---

## Disclaimer

⚠️ This tool is for **educational purposes only** and does not constitute regulated financial advice. Always consult a qualified financial advisor before making investment decisions. Past performance is not a reliable indicator of future results.

---

## Academic context

This project is submitted as part of the CM3070 Final Project module, BSc Computer Science, University of London. The RAG-based approach was chosen over classical reinforcement learning for financial advisory, as it provides more reliable, data-grounded responses with lower infrastructure requirements.
