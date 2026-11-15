
import os
import signal
import threading
import time
import logging

import requests
from flask import Flask

from bot import (
    run_bot_forever,
    request_shutdown,
    is_shutting_down,
    SHUTDOWN_FLAG,
)

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
    return "Bot is alive", 200


@app.route("/health")
def health():
    return {
        "status": "ok",
        "uptime": time.time() - _START_TIME,
        "shutting_down": is_shutting_down(),
    }, 200


def run_flask():
    port = int(os.environ.get("PORT", "10000"))

    app.run(
        host="0.0.0.0",
        port=port,
        threaded=True,
        use_reloader=False,
    )


# ---------------- KEEPALIVE ----------------

def _keepalive_loop():
    url = os.environ.get("RENDER_EXTERNAL_URL")

    if not url:
        log.info("Keepalive disabled: RENDER_EXTERNAL_URL not set")
        return

    target = url.rstrip("/") + "/health"

    log.info("Keepalive started: %s", target)

    while not is_shutting_down():
        time.sleep(600)

        if is_shutting_down():
            break

        try:
            response = requests.get(target, timeout=10)
            log.info(
                "Keepalive response: HTTP %s",
                response.status_code,
            )
        except Exception as error:
            log.warning("Keepalive failed: %s", error)


# ---------------- GRACEFUL SHUTDOWN ----------------

_shutdown_event = threading.Event()


def _handle_signal(signum, _frame):
    log.warning("Received signal %s", signum)

    _shutdown_event.set()

    try:
        request_shutdown()
    except Exception:
        log.exception("Could not request bot shutdown")


def _install_signal_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            log.exception("Could not install signal handler: %s", sig)


# ---------------- MAIN ----------------

if __name__ == "__main__":
    _install_signal_handlers()

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True,
        name="flask",
    )
    flask_thread.start()

    keepalive_thread = threading.Thread(
        target=_keepalive_loop,
        daemon=True,
        name="keepalive",
    )
    keepalive_thread.start()

    log.info("Flask and keepalive threads started")
    log.info("Starting Pyrogram bot")

    try:
        run_bot_forever()

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")

    except Exception:
        log.exception("Fatal bot startup/runtime error")

    finally:
        log.info("Application shutting down")

        _shutdown_event.set()

        try:
            if os.path.exists(SHUTDOWN_FLAG):
                os.remove(SHUTDOWN_FLAG)
        except OSError:
            log.exception("Could not remove shutdown flag")

        log.info("Application stopped")
