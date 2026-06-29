# Deploying CryptoTracker Pro

This is a **Streamlit** app — a long-running web server, not a serverless function.

## ❌ Why Vercel / Netlify don't work
Vercel and Netlify host **serverless functions** (a Python entrypoint that exports a
WSGI/ASGI `app`/`handler`, runs briefly, then exits). Streamlit is the opposite: it runs
`streamlit run app.py` as a **persistent process** and talks to the browser over a live
**WebSocket**. There is no `app` object to export, and a stateless, time-limited function
can't keep a Streamlit session alive. The error

> `Found app.py but it does not export a top-level "app", "application", or "handler"`

is Vercel trying to load `app.py` as a function. Don't deploy this on Vercel — use one of
the platforms below, which run a real server.

---

## ✅ Option 1 — Streamlit Community Cloud (recommended, free, zero config)
Made specifically for Streamlit. Nothing to configure beyond this repo.

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub.**
3. Repository: `Aryan-Shrivastav9891/cryptotracker-pro` · Branch: `main` · Main file: `app.py`.
4. (Optional) **Advanced settings → Python version: 3.11.**
5. **Deploy.** It installs `requirements.txt` and gives you a public URL.

> Secrets (e.g. a CryptoCompare news key) go in **App → Settings → Secrets**, TOML format:
> `CRYPTOCOMPARE_API_KEY = "..."`. The app runs fine without any keys (it degrades gracefully).

---

## ✅ Option 2 — Render (uses `render.yaml` in this repo)
1. Push this repo to GitHub (already done).
2. On **https://render.com → New + → Blueprint**, pick this repo. It reads `render.yaml`.
3. Render runs:
   `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
4. Deploy → you get a public `*.onrender.com` URL. (Free tier sleeps when idle.)

---

## ✅ Option 3 — Railway (uses the `Procfile` in this repo)
1. **https://railway.app → New Project → Deploy from GitHub repo**, pick this repo.
2. Railway detects Python + the `Procfile` and runs the Streamlit server on `$PORT`.
3. Add a public domain in the service's **Settings → Networking**.

---

## ✅ Option 4 — Hugging Face Spaces
1. **https://huggingface.co/new-space → SDK: Streamlit.**
2. Push this repo's contents to the Space (or link the GitHub repo). It uses
   `requirements.txt` and `app.py` automatically.

---

## Local run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Files that make remote deploys work
- `requirements.txt` — Python dependencies (all platforms).
- `runtime.txt` — pins Python 3.11.9 (Render/Heroku-style).
- `Procfile` — start command for Railway/Render/Heroku-style platforms.
- `render.yaml` — one-click Render blueprint.
- `.streamlit/config.toml` — theme + `headless = true` for servers.
