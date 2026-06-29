"""CryptoTracker Pro — shared library package.

Modules:
    formatting  -> number / currency / date helpers (INR, compact, %, HTML-strip)
    coingecko   -> live market + coin + historical data (cached HTTP)
    news        -> CryptoCompare crypto news
    model       -> fast per-coin trend estimate for card grids (cached)
    forecast    -> backtested multi-day price forecast + confidence intervals (statsmodels)
    charts      -> Plotly line / forecast / confidence-band charts
    enrich      -> attach prediction fields to coins; search filter
    ui          -> shared Streamlit UI (theme, sidebar, cards, market overview)
"""
