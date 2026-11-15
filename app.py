import asyncio
import os
import signal
import threading
import time
import logging

import requests
from flask import Flask

# Import the shutdown-aware entry point, not `bot` directly
from bot import run_bot_forever, request_shutdown, is_shutting_down, SHUTDOWN_FLAG

# ---------------- LOGGING ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("app")


# ---------------- FLASK ----------------

app = Flask(__name__)

_START_TIME = time.time()


@app.route("/")
def home():
    return "✅ Bot is alive"


@app.route("/health")
def health():
    return {
        "status": "ok",
        "uptime": time.time() - _START_TIME,
        "shutting_down": is_shutting_down(),
    }


def run_flask():
    app.run(host="0.0.0.0", port=10000, threaded=True)


# ---------------- KEEPALIVE (free-tier) ----------------

def _keepalive_loop():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        log.info("keepalive disabled (no RENDER_EXTERNAL_URL)")
        return
    target = url.rstrip("/") + "/health"
    log.info("keepalive started -> %s every 600s", target)
    while not is_shutting_down():
        time.sleep(600)
        if is_shutting_down():
            return
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
    # Tell bot.py to stop, write the SHUTDOWN_FLAG file so in-flight
    # renders abort cleanly instead of dying mid-write.
    try:
        request_shutdown()
    except Exception as error:
        log.warning("request_shutdown failed: %s", error)


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

    log.info("starting pyrogram bot (run_bot_forever) in main thread")

    try:
        run_bot_forever()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt - exiting")
    except Exception as error:
        log.exception("bot loop error: %s", error)
    finally:
        if _shutdown_event.is_set():
            log.info("shutdown event was set - clean exit")
        else:
            log.info("run_bot_forever() returned - exiting")
        try:
            if os.path.exists(SHUTDOWN_FLAG):
                os.remove(SHUTDOWN_FLAG)
        except Exception:
            pass
