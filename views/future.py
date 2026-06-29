"""Future Gains — AI Forecast Lab (multi-horizon, backtested, honest).

Shows forecasts at 1D / 4D / 1W / 1M, each validated out-of-sample with its own
MAPE / directional / skill-vs-naive, and flagged unreliable when it has no edge.
Bentobox layout: combined chart, per-horizon table, news + sentiment, plain-English note.
"""
import pandas as pd
import streamlit as st

from lib import charts, coingecko, forecast, glossary, news, ui
from lib.formatting import format_inr

dark = st.session_state.get("dark_mode", False)

# Global CSS (incl. .bento-h / .bento-big) comes from lib/ui.apply_theme — single
# source of truth, injected once per run in app.py. Do not emit CSS from pages.

SIG_COLORS = {"Strong Buy": "#16a34a", "Buy": "#22c55e", "Hold": "#3b82f6",
              "Sell": "#f97316", "Strong Sell": "#ef4444"}

st.markdown("## ⏳ AI Forecast Lab")
st.caption("Forecasts at **1D · 4D · 1W · 1M**, each backtested on unseen data. "
           "Beats a naive baseline — or it tells you it doesn't.")

# --- controls ---------------------------------------------------------------
market = coingecko.load_markets(per_page=100, sparkline=False)
if not market:
    st.warning("No market data available right now.")
    st.stop()

by_id = {c.get("id"): c for c in market}
labels = {f"{c.get('name')} ({(c.get('symbol') or '').upper()})": c.get("id") for c in market}
label_list = list(labels.keys())

preselected = st.session_state.get("selected_coin_id")
default_index = 0
if preselected in by_id:
    for i, lbl in enumerate(label_list):
        if labels[lbl] == preselected:
            default_index = i
            break

with st.container(border=True):
    ctrl = st.columns([3, 1])
    chosen_label = ctrl[0].selectbox("Coin", label_list, index=default_index)
    coin_id = labels[chosen_label]
    days = ctrl[1].selectbox("History", [365, 540, 730], index=1, format_func=lambda d: f"{d} days")

coin = by_id.get(coin_id, {})
name = coin.get("name", coin_id)
symbol = (coin.get("symbol") or "").upper()

# --- train + backtest (all horizons) + sentiment ----------------------------
with st.status(f"Training the multi-model ensemble for {name}…", expanded=True) as status:
    st.write(f"📥 Loading ~{days} days of daily prices (Binance → CoinGecko fallback)…")
    st.write("🧪 Walk-forward backtest of every model, h-step-ahead, for 1D / 4D / 1W / 1M…")
    st.write("🧠 Dynamic selection (drop models worse than naive) → 3 combiners → conformal bands…")
    st.caption("⏳ First run for a coin trains ~14 models across many windows (~1–2 min). "
               "It's cached for an hour after that.")
    result = forecast.forecast_coin(coin_id, symbol, days=days)
    if not result:
        status.update(label="Not enough history to backtest", state="error")
        st.warning(f"Not enough daily history to backtest {name} honestly (need ~70+ daily "
                   f"candles). Try a longer history window, or another coin.")
        st.stop()
    st.write("📰 Fetching news + scoring sentiment…")
    sent = news.news_sentiment(symbol, name=name, limit=8)
    status.update(label=f"Done · {result['engine'][:70]}… · data: {result['source']}",
                  state="complete")

horizons = result["horizons"]
current = result["current_price"]

# --- Animated bento hero (anime.js): price + forecast KPIs + signal ----------
_primary = next((h for h in horizons if h["reliable"]), horizons[0])
_dec = 0 if current >= 100 else (2 if current >= 1 else 6)


def _pctval(x):  # metric -> percentage number for count-up (0 if missing)
    return float(x) * 100 if x is not None else 0.0


ui.animated_bento(
    dark,
    hero={"label": f"{name} ({symbol}) · current price", "value": current, "fmt": "inr",
          "decimals": _dec, "delta": _primary["predicted_change"],
          "spark": result["history_prices"][-40:]},
    kpis=[
        {"label": "MAPE", "value": _pctval(_primary["mape"]), "fmt": "pct", "decimals": 1,
         "tip": glossary.get_definition("MAPE")},
        {"label": "Directional", "value": _pctval(_primary["directional"]), "fmt": "pct",
         "decimals": 0, "tip": glossary.get_definition("Directional accuracy")},
        {"label": "Skill vs naive", "value": _pctval(_primary["skill"]), "fmt": "pct",
         "decimals": 0, "signed": True, "tip": glossary.get_definition("Skill score")},
        {"label": "Coverage", "value": _pctval(_primary["coverage"]), "fmt": "pct",
         "decimals": 0, "tip": glossary.get_definition("Confidence band")},
    ],
    signal={"text": f"{_primary['label']}: {_primary['signal']}",
            "kind": ui._signal_kind(_primary["signal"])},
    height=250,
)

# --- BENTO ROW 1: combined chart (wide) + sentiment (narrow) ----------------
left, right = st.columns([2, 1])
with left:
    with st.container(border=True):
        st.markdown('<p class="bento-h">🔮 Forecast — 1D / 4D / 1W / 1M (with 80% '
                    + ui.annotate("confidence band") + ")</p>", unsafe_allow_html=True)
        fig = charts.multi_forecast_chart(result["history_dates"], result["history_prices"],
                                          horizons, dark=dark, height=420)
        st.plotly_chart(fig, use_container_width=True, key=f"mfc_{coin_id}_{days}")
with right:
    with st.container(border=True):
        st.markdown('<p class="bento-h">📰 News sentiment</p>', unsafe_allow_html=True)
        if sent["available"]:
            s_color = {"Bullish": "#16a34a", "Bearish": "#ef4444"}.get(sent["label"], "#3b82f6")
            st.markdown(f'<span class="bento-big" style="color:{s_color}">{sent["label"]}</span>',
                        unsafe_allow_html=True)
            st.caption(f"score {sent['score']:+.2f} · {sent['n']} headlines")
        else:
            st.markdown('<span class="bento-big" style="color:#6b7280">N/A</span>', unsafe_allow_html=True)
            st.caption("No news / sentiment unavailable")

# --- BENTO ROW 2: per-horizon table -----------------------------------------
with st.container(border=True):
    st.markdown('<p class="bento-h">📊 Per-horizon forecast & out-of-sample backtest</p>',
                unsafe_allow_html=True)
    rows = []
    for hd in horizons:
        # News sentiment applies only as a soft ±1-notch tilt, and only when reliable.
        disp_sig = (forecast.tilt_signal(hd["signal"], sent["label"])
                    if (sent["available"] and hd["reliable"]) else hd["signal"])
        # Too few out-of-sample windows OR 2-fold (selection==test) -> metrics are noisy/
        # optimistic, so don't present skill/coverage as trustworthy.
        low = hd["score_windows"] < forecast.MIN_SCORE or hd.get("select_on_test", False)
        rows.append({
            "Horizon": hd["label"],
            "Predicted": format_inr(hd["predicted_price"]),
            "Exp. move": f"{hd['predicted_change']:+.2f}%",
            "MAPE": f"{hd['mape']*100:.1f}%" if hd["mape"] is not None else "—",
            "Directional": f"{hd['directional']*100:.0f}%" if hd["directional"] is not None else "—",
            "Skill": "—" if (low or hd["skill"] is None) else f"{hd['skill']*100:+.0f}%",
            "Coverage": "—" if (low or hd["coverage"] is None) else f"{hd['coverage']*100:.0f}%",
            "Combiner": hd["combiner"] or "—",
            "Reliable": "✅" if hd["reliable"] else ("🔸 low data" if low else "⚠️ no edge"),
            "Signal": disp_sig,
        })
    # Column tooltips built dynamically from the glossary (add a term = one line there).
    _COL_TERMS = {"MAPE": "MAPE", "Directional": "Directional accuracy", "Skill": "Skill score",
                  "Coverage": "Confidence band", "Combiner": "Ensemble"}
    st.dataframe(
        pd.DataFrame(rows), width="stretch", hide_index=True,
        column_config={c: st.column_config.TextColumn(c, help=glossary.get_definition(t))
                       for c, t in _COL_TERMS.items()},
    )
    # Transparency: which models survived selection per horizon (the spec's requirement).
    bits = []
    for hd in horizons:
        used = ", ".join(hd["models_used"]) if hd["models_used"] else "naive only"
        bits.append(f"**{hd['label']}** → {hd['combiner'] or 'n/a'} of [{used}]")
    st.caption("🏁 Winning combiner + models: " + " · ".join(bits))
    st.caption(f"📐 Coverage = measured share of real prices that fell inside the 80% band "
               f"(target 80%). Engine: {len(result['models_available'])} models — "
               f"{', '.join(result['models_available'])}.")
    if sent["available"] and any(h["reliable"] for h in horizons):
        st.caption(f"Signals for reliable horizons include a soft tilt from {sent['label'].lower()} news.")

# --- BENTO ROW 3: news list + plain-English note ----------------------------
a, b = st.columns([1, 1])
with a:
    with st.container(border=True):
        st.markdown(f'<p class="bento-h">📰 Latest news — {name}</p>', unsafe_allow_html=True)
        if sent["items"]:
            tag = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "⚪"}
            for art in sent["items"][:6]:
                lab = "Bullish" if art["sentiment"] > 0.05 else "Bearish" if art["sentiment"] < -0.05 else "Neutral"
                title = art["title"] or "Untitled"
                title = title[:120] + ("…" if len(title) > 120 else "")
                st.markdown(
                    f"- {tag[lab]} [{title}]({art['url']}) "
                    f"<span class='ctp-muted'>· {art['source']} {art['published']}</span>",
                    unsafe_allow_html=True)
        else:
            st.caption(f"No recent news for {name}. (Add a free API key for more — see HOW_TO_RUN.md.)")
with b:
    with st.container(border=True):
        st.markdown('<p class="bento-h">📖 What these numbers mean (hover terms)</p>',
                    unsafe_allow_html=True)
        explainer = (
            "- **MAPE** = average % the forecast was off, on unseen data (lower is better).\n"
            "- **Directional accuracy** = how often it got tomorrow's up/down right (50% = a coin flip).\n"
            "- **Skill score** = % less error than a naive baseline; **>0 means it has an edge**.\n"
            "- The **Confidence band** shows the likely range; longer horizons (1W, 1M) are far less reliable.\n"
            "- A horizon shows Buy/Sell only when it's reliable **and** the expected move beats its own error."
        )
        st.markdown(ui.annotate(explainer), unsafe_allow_html=True)
        st.markdown(ui.annotate("Model features include RSI, MACD, Bollinger Bands, SMA, EMA, "
                                "Volatility and Drift."), unsafe_allow_html=True)
        st.markdown(
            "<span class='ctp-muted' style='font-size:0.85rem'>"
            + ui.annotate("Daily crypto is close to a Random walk; no model reliably beats the "
                          "market. ⚠️ Educational only — not financial advice.")
            + "</span>",
            unsafe_allow_html=True,
        )

cols = st.columns(2)
if cols[0].button("📄 Full coin details →", width="stretch"):
    ui.go_to_coin(coin_id)
cols[1].link_button("🔗 View on CoinGecko", f"https://www.coingecko.com/en/coins/{coin_id}",
                    width="stretch")
