"""
NYX Live Feed — connects Binance WS to the existing runtime path.

Ticket 22 — no new engine, no shadow pipeline. The canonical flow :

  Binance WS (btcusdt@kline_15m)
    → BarBuilder (normalize + duplicate reject + closed-only)
    → FeedHealth (stale / monotonicity / gap checks)
    → NYXLiveDecider.on_15m_bar() (existing canonical runtime)
    → Signal emitted (logged via EventAlerter if configured)

Usage :
  BINANCE_SYMBOL=btcusdt python scripts/run_live_feed.py

Requires : pip install websocket-client

The script is blocking. Kill with Ctrl+C or SIGTERM.
Feed health status is printed every 5 minutes.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
log = logging.getLogger('nyx.live')

SYMBOL = os.environ.get('BINANCE_SYMBOL', 'btcusdt').lower()
INTERVAL = '15m'
MODELS_DIR = HERE / 'models' / SYMBOL.upper()


def main() -> int:
    from src.live.binance_ws import BinanceKlineStream
    from src.live.bar_builder import BarBuilder
    from src.live.feed_health import FeedHealth
    from src.ml.nyx_live_decider import NYXLiveDecider

    if not MODELS_DIR.is_dir():
        log.error('No model artefact at %s', MODELS_DIR)
        return 1

    decider = NYXLiveDecider(
        symbol=SYMBOL.upper(),
        artifact_dir=MODELS_DIR,
    )
    bar_builder = BarBuilder()
    feed_health = FeedHealth(stale_seconds=120, expected_interval_ms=900_000)
    stream = BinanceKlineStream(symbol=SYMBOL, interval=INTERVAL)

    last_health_log = time.time()
    n_bars = 0
    n_signals = 0

    def on_message(event: dict) -> None:
        nonlocal n_bars, n_signals, last_health_log

        feed_health.on_event(time.time())
        bar = bar_builder.on_event(event)
        if bar is None:
            return

        # Closed bar received — check feed health before acting.
        k = event.get('k', {})
        feed_health.on_bar_timestamp(int(k.get('t', 0)))

        if not feed_health.is_healthy():
            log.warning('feed unhealthy — skipping bar %s', bar['timestamp'])
            return

        # Feed the canonical runtime path (NYXLiveDecider).
        sig = decider.on_15m_bar(bar)
        n_bars += 1

        if sig.direction != 0:
            n_signals += 1
            log.info(
                'SIGNAL %s dir=%+d conv=%.3f ts=%s',
                SYMBOL.upper(), sig.direction, sig.conviction,
                bar['timestamp'],
            )

        # Periodic health log.
        now = time.time()
        if now - last_health_log > 300:
            log.info(
                'health: %s | bars=%d signals=%d',
                feed_health.status(), n_bars, n_signals,
            )
            last_health_log = now

    def on_error(e: Exception) -> None:
        log.error('WS error: %s', e)

    log.info('starting NYX live feed for %s', SYMBOL.upper())
    log.info('model: %s', MODELS_DIR)

    try:
        stream.run(on_message=on_message, on_error=on_error)
    except KeyboardInterrupt:
        log.info('shutting down...')
        stream.stop()

    log.info('final: bars=%d signals=%d', n_bars, n_signals)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
