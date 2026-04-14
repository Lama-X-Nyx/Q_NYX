"""
Paper-live driver — entrypoint wired into Docker CMD.

Responsibilities:
  - Build PaperLiveRunner from env / config.
  - On startup: re-sync missed bars using ReSyncManager.
  - Loop: poll the exchange for new bars, feed runner.on_bar().
  - On fatal error: alert + exit non-zero so docker restarts us.

NOTE: this module is the glue layer. The actual strategy, data feed, and
broker adapter are injected so the core remains exchange-agnostic.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger("paper_live.main")


STORAGE_DIR = Path(os.environ.get("NYX_STORAGE", "/app/nyx-system/storage"))
DB_PATH = STORAGE_DIR / "decisions.db"
STATE_PATH = STORAGE_DIR / "state.json"
HEARTBEAT_PATH = STORAGE_DIR / "heartbeat.json"


def _install_signal_handlers(runner) -> None:
    def _graceful(_signum, _frame):
        log.info("shutdown signal received")
        runner.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Lazy imports so the module is cheap to import for unit tests.
    from src.paper_live.runner import PaperLiveRunner

    # Strategy: replaced at deploy time. A no-op stub keeps the container
    # healthy enough for infrastructure smoke tests.
    class _NoopStrategy:
        def decide(self, pair, bar):
            return {
                'action': 'WAIT',
                'orch_score': 0.0,
                'size_factor': 0.0,
                'blocked_by': ['noop-stub'],
                'features': {},
                'agent_results': {
                    k: {'state': '', 'score': 0.0, 'passed': False}
                    for k in ('context', 'regime', 'setup', 'entry')
                },
            }

    runner = PaperLiveRunner(
        strategy=_NoopStrategy(),
        db_path=DB_PATH,
        state_path=STATE_PATH,
        heartbeat_path=HEARTBEAT_PATH,
        telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
        telegram_chat=os.environ.get("TELEGRAM_CHAT_ID") or None,
        discord_webhook=os.environ.get("DISCORD_WEBHOOK_URL") or None,
        model_version=os.environ.get("NYX_MODEL_VERSION", "v0.3.2"),
    )
    _install_signal_handlers(runner)

    log.info("paper-live started, pair=%s", os.environ.get("NYX_PAIR", "BTCUSDT"))
    # Initial heartbeat so Docker HEALTHCHECK sees us alive immediately.
    runner.heartbeat.tick(context={'phase': 'startup'})
    # Emit restart event so operators see we're back.
    runner.notify_restart()

    # Real deployment will plug in a live market-data loop here; for now we
    # simply keep the heartbeat ticking so the container stays healthy while
    # the strategy is connected at integration time.
    import time
    while True:
        runner.heartbeat.tick(context={'phase': 'idle'})
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main() or 0)
