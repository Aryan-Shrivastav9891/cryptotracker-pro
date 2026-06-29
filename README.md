# CryptoTracker Pro — Python (Streamlit) edition

A single, smooth **Streamlit** crypto dashboard: track live cryptocurrency prices
(in ₹ INR), explore detailed per-coin pages with interactive charts, and get
**honest, backtested AI price forecasts** with buy/sell signals.

> ⚠️ For educational use only. Nothing here is financial advice.

---

## 🎯 What it does (functionality summary)

- **Live market data** — pulls the top ~250 coins from the **CoinGecko** API (prices in ₹ INR), cached so it stays fast and avoids rate limits.
- **7 pages** with a sidebar navigation: All, Altcoins, Memecoins, Profitable, AI Predicted, Future Gains, and Coin Details.
- **Fast price estimate on every card** — a per-coin, scale-correct log-trend projection (~1 day ahead) shown on the coin grids.
- **Real AI forecasting (Future page)** — for a chosen coin it fetches real daily history, runs a **walk-forward backtest** on unseen data, then forecasts ahead with an **80% confidence band**, reports honest accuracy (**MAPE / directional accuracy / MAE**), and gives a **Strong Buy → Sell** signal grounded in that measured error.
- **Deep coin detail** — 7-day price chart, market cap / volume / supply, multi-currency prices (INR / USD / EUR / BTC), all-time high/low with dates, community stats (Twitter / Reddit / Telegram), developer stats (GitHub stars / forks / commits), official links, and full description (HTML-sanitized).
- **Crypto news** — latest headlines per coin from **CryptoCompare**.
- **Interactive charts** — Plotly line charts, forecast charts with confidence bands.
- **UX** — global search, **dark mode**, market overview (market cap, 24h volume, top movers), and one-click **data refresh**.
- **Optional ML data pipeline** (`scripts/`) — download multi-year OHLCV from Binance and prepare a leakage-free, time-based train/test split.

---

## 📑 Pages

| Page | What you can do |
|------|-----------------|
| **All** | Searchable, paginated grid of all coins (slider for 6–120). Market overview + per-card price estimate. |
| **Altcoins** | Same grid, filtered to coins ranked outside the top 10 by market cap. |
| **Memecoins** | Popular meme/community tokens (DOGE, SHIB, PEPE, …) with a high-risk warning. |
| **Profitable** | Coins up **>10% in 24h**, shown as a sortable table **and** a card grid. |
| **AI Predicted** | Coins ranked by predicted upside; slider to set a minimum-upside threshold. |
| **Future Gains** | Pick a coin → train + **backtest** a forecast model → confidence band, accuracy metrics, signal, news. |
| **Coin Details** | Full single-coin breakdown: chart, market data, prices, ATH/ATL, community + dev stats, links, about. |

---

## 🧰 Libraries used

| Library | Used for |
|---------|----------|
| **streamlit** | The whole web UI — multipage navigation, session state, caching, widgets. |
| **requests** | HTTP calls to the CoinGecko (market data) and CryptoCompare (news) APIs. |
| **pandas** | Tabular data + the table on the Profitable page (and the data scripts). |
| **numpy** | Numerical math behind the trend estimate and the forecasting pipeline. |
| **plotly** | Interactive price charts, forecast charts, and confidence-band fills. |
| **statsmodels** | The forecasting engine — Holt damped-trend Exponential Smoothing + intervals. |
| **ccxt** *(scripts only)* | Free, key-less OHLCV candle download from Binance and other exchanges. |
| **scikit-learn** *(scripts only)* | `StandardScaler` + `TimeSeriesSplit` for the train/test split script. |

> The **app** needs: `streamlit, requests, pandas, numpy, plotly, statsmodels` (see `requirements.txt`).
> The **scripts** additionally need: `ccxt, scikit-learn` (see `scripts/requirements.txt`).

---

## 🧩 How it's built (modules)

| File | Role |
|------|------|
| `app.py` | Entry point — page config, sidebar, `st.navigation` over the views. |
| `lib/coingecko.py` | Cached CoinGecko access: `get_markets`, `get_coin`, `get_market_chart`. |
| `lib/forecast.py` | Backtest + multi-day forecast + confidence intervals (`forecast_coin`). |
| `lib/model.py` | Fast per-coin ~1-day trend estimate for the card grids (`predict_prices`). |
| `lib/charts.py` | Plotly `line_chart` and `forecast_interval_chart`. |
| `lib/enrich.py` | Attaches prediction fields to coins; `filter_by_search`. |
| `lib/formatting.py` | INR / compact / % / date formatting, `strip_html`, `safe_float`. |
| `lib/news.py` | CryptoCompare news (`get_news`). |
| `lib/ui.py` | Theme, sidebar, market overview, coin cards/grid, footer, navigation. |
| `views/*.py` | The 7 pages (one file each). |
| `scripts/fetch_crypto_data.py` | Download OHLCV + engineer indicators → CSV per symbol. |
| `scripts/time_split.py` | Time-based (no-shuffle) train/test split + walk-forward CV. |

---

## 🚀 Quickstart

> 📖 New here? See **[HOW_TO_RUN.md](HOW_TO_RUN.md)** for a step-by-step guide with troubleshooting.

Requires **Python 3.9+**.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open the URL it prints (default <http://localhost:8501>).

The forecasting engine (**statsmodels**) is in `requirements.txt`. If it is ever
missing, the app falls back to a random-walk-with-drift baseline so it still runs.

---

## 📊 Data pipeline for ML (optional)

The `scripts/` folder is a **standalone** toolkit for building your own training
dataset — using free, no-API-key public data.

```bash
pip install -r scripts/requirements.txt

# 1. Download 2–3 years of OHLCV candles from Binance (ccxt) + technical features
python scripts/fetch_crypto_data.py     # -> data/<SYMBOL>_<TIMEFRAME>.csv

# 2. Make a time-based, no-shuffle train/test split (train on past, test on recent)
python scripts/time_split.py
```

- **`fetch_crypto_data.py`** — paginates OHLCV with rate-limiting + retries, builds a clean datetime-indexed DataFrame, adds indicators (returns/log-returns, SMA, EMA, MACD, RSI, Bollinger, rolling volatility), and writes one CSV per symbol. Configure symbols / timeframe / start date at the top.
- **`time_split.py`** — chronological split (by fraction *and* by cutoff date), leakage-free target, scaler fit on **train only**, plus walk-forward `TimeSeriesSplit` cross-validation.

---

## 🗂️ Project structure

```
app.py                  # Entry point: page config + st.navigation
.streamlit/config.toml  # Theme + server config
requirements.txt        # App dependencies
HOW_TO_RUN.md           # Step-by-step run guide + troubleshooting
lib/                    # coingecko, forecast, model, charts, enrich, formatting, news, ui
views/                  # home, altcoins, memecoins, profitable, predicted, future, coin_details
scripts/                # fetch_crypto_data.py, time_split.py, requirements.txt
data/                   # Downloaded CSV datasets (git-ignored, regenerable)
```

---

## 🔁 What changed from the original (Next.js → Python)

This started as a Next.js/React app; it is now a pure Python project.

| Original (Next.js/React) | Python (Streamlit) |
| ------------------------ | ------------------ |
| TensorFlow.js LSTM-GRU (browser) | statsmodels backtested forecast + per-coin trend estimate |
| Chart.js / react-chartjs-2 | Plotly |
| axios | requests (cached with `st.cache_data`) |
| React state + Next.js routing | `st.session_state` + `st.navigation` |
| framer-motion animations | (omitted — not needed for a data app) |

### Why the prediction was rebuilt

The original model pooled **every coin into one model** with a single global
min-max normalisation — which collapses small-cap coins toward zero — and then
"predicted" the very price it was trained on (near-identity). That isn't
forecasting. The Future page now trains and **backtests per coin on real
history** and reports its measured error, so you can see how reliable (or not)
each forecast is. Crypto is largely a random walk over short horizons — no
library can truly beat the market, so the app is honest about uncertainty rather
than showing a confident-looking but meaningless number.

> Note: CoinGecko caps `per_page` at 250 (the original requested 1000, which the API silently clamps).
> The original Next.js source has been removed — recoverable from git history if ever needed.
