# How to Run — CryptoTracker Pro (Python / Streamlit)

A simple, step-by-step guide to run this project.

---

## ✅ Quick start (copy–paste)

From the project folder (`new-model-crypto-p`):

```bash
# 1. Create a virtual environment (one time only)
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell/CMD)

# 3. Install dependencies (one time, or after they change)
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Then open the URL it prints — usually:

### 👉 http://localhost:8501

That's it. 🎉

> ⚠️ The correct address is **`:8501`** (Streamlit).
> `http://localhost:3000` was the old Next.js app and **no longer exists** — don't use it.

---

## ▶️ Already set up? Fastest way to start

If `.venv` already exists, you can run it in one line (no activation needed):

```bash
.venv/bin/python -m streamlit run app.py
```

---

## ⏹️ Stop the app

- In the terminal where it's running: press **Ctrl + C**.
- If it's running in the background:

```bash
pkill -f "streamlit run app.py"
```

---

## 🔁 Common problems & fixes

| Problem | Why | Fix |
|--------|-----|-----|
| **"This site can't be reached" / page fails to load** | The server isn't running | Run `streamlit run app.py` again, then refresh the browser |
| Wrong page at **`:3000`** | That was the old Next.js app (removed) | Use **`http://localhost:8501`** |
| **"Port 8501 is already in use"** | Another instance is running | Use a different port: `streamlit run app.py --server.port 8502` |
| First page load is slow | It's fetching live data from CoinGecko | Normal — wait a few seconds; it's cached after that |
| `command not found: streamlit` | venv not activated / not installed | `source .venv/bin/activate` then `pip install -r requirements.txt` |
| `ModuleNotFoundError` | Dependencies not installed | `pip install -r requirements.txt` |

Check the log if it crashes:

```bash
streamlit run app.py        # errors print right in the terminal
```

---

## 📊 (Optional) Data pipeline scripts

These are **separate** from the app — only needed if you want to download raw
historical data for training your own ML model.

```bash
# Install the scripts' dependencies (one time)
pip install -r scripts/requirements.txt

# 1. Download 2–3 years of OHLCV data from Binance (no API key needed)
python scripts/fetch_crypto_data.py        # writes data/<SYMBOL>_<TIMEFRAME>.csv

# 2. Make a time-based (no-shuffle) train/test split
python scripts/time_split.py
```

Edit the config block at the top of each script (symbols, timeframe, dates).

---

## 📁 What's in the project

```
app.py            → the app's entry point (run this)
lib/              → data fetching, forecasting, charts, shared UI
views/            → the pages (Home, Altcoins, Future Gains, Coin Details, …)
scripts/          → optional data downloader + train/test split
data/             → downloaded CSV datasets
requirements.txt  → Python dependencies for the app
.streamlit/       → theme & config
```

---

## ℹ️ Requirements

- **Python 3.9 or newer** (`python3 --version` to check)
- Internet connection (live prices from CoinGecko)
- For educational use only — **not financial advice**.
```
