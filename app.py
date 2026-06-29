"""CryptoTracker Pro — Streamlit edition (Python port of the Next.js app).

Entry point. Sets up the page, the persistent sidebar (search / dark mode /
model status), and the multipage navigation, then runs the selected page.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="CryptoTracker Pro",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

from lib import ui  # noqa: E402  (must follow set_page_config)

ui.init_state()

# ---- Multipage navigation (replaces the React <Tabs> + Next.js pages) ----
pages = {
    "Markets": [
        st.Page("views/home.py", title="All", icon=":material/grid_view:", default=True),
        st.Page("views/altcoins.py", title="Altcoins", icon=":material/star:"),
        st.Page("views/memecoins.py", title="Memecoins", icon=":material/bolt:"),
        st.Page("views/profitable.py", title="Profitable", icon=":material/bar_chart:"),
    ],
    "AI": [
        st.Page("views/predicted.py", title="AI Predicted", icon=":material/trending_up:"),
        st.Page("views/future.py", title="Future Gains", icon=":material/schedule:"),
        st.Page("views/intraday.py", title="Intraday Signal Lab", icon=":material/bolt:"),
        st.Page("views/live.py", title="Live Paper-Trading", icon=":material/sensors:"),
    ],
    "Explore": [
        st.Page("views/coin_details.py", title="Coin Details", icon=":material/info:"),
    ],
}

nav = st.navigation(pages)

ui.sidebar_controls()
ui.apply_theme(st.session_state.dark_mode)
ui.top_bar()

nav.run()

ui.footer()
