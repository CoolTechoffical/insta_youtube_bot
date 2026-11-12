import asyncio
import os
import signal
import threading
import time
import logging

import requests
from flask import Flask

from bot import bot

# ---------------- LOGGING ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("app")


# ---------------- FLASK ----------------

app = Flask(__name__)


@app.route("/")
def home():
    return "✅ Bot is alive"


@app.route("/health")
def health():
    return {"status": "ok", "uptime": time.time() - _START_TIME}


_START_TIME = time.time()


def run_flask():
    # threaded=True so /health doesn't block behind slow renders
    app.run(host="0.0.0.0", port=10000, threaded=True)


# ---------------- KEEPALIVE (free-tier) ----------------

def _keepalive_loop():
    """
    Ping our own public URL every 10 minutes so Render's free tier
    doesn't idle-kill the service after 15 minutes of no HTTP traffic.
    RENDER_EXTERNAL_URL is set automatically by Render.
    """
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        log.info("keepalive disabled (no RENDER_EXTERNAL_URL)")
        return
    target = url.rstrip("/") + "/health"
    log.info("keepalive started -> %s every 600s", target)
    while True:
        time.sleep(600)
        try:
            r = requests.get(target, timeout=10)
            log.info("keepalive ping ok status=%s", r.status_code)
        except Exception as error:
            log.warning("keepalive ping failed: %s", error)


# ---------------- GRACEFUL SHUTDOWN ----------------

_shutdown_event = threading.Event()


def _handle_signal(signum, _frame):
    log.warning("signal received: %s - shutting down gracefully", signum)
    _shutdown_event.set()


def _install_signal_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError) as error:
            log.warning("could not install handler for %s: %s", sig, error)


# ---------------- MAIN ----------------

if __name__ == "__main__":
    _install_signal_handlers()

    threading.Thread(target=run_flask, daemon=True, name="flask").start()
    threading.Thread(target=_keepalive_loop, daemon=True, name="keepalive").start()

    log.info("starting pyrogram bot in main thread")

    try:
        # Pyrogram's bot.run() blocks until stopped.
        bot.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt - exiting")
    finally:
        if _shutdown_event.is_set():
            log.info("shutdown event was set - clean exit")
        else:
            log.info("bot.run() returned - exiting")
