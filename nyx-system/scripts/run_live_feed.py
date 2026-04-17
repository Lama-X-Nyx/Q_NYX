"""
NYX Live Feed — ONE entry point, ONE orchestrator.

Binance WS → BarBuilder → NYXRuntime.on_bar()

NYXRuntime handles EVERYTHING internally :
  GBM → Jesse → Fractal Quality → Risk → OMS → Portfolio → State → Monitoring

Nothing is called outside NYXRuntime. If it's not in NYXRuntime,
it doesn't exist in the system.

Usage :
  BINANCE_SYMBOL=btcusdt python scripts/run_live_feed.py

Requires : pip install websocket-client
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
MODELS_DIR = HERE / 'models' / SYMBOL.upper()


def main() -> int:
    from src.live.binance_ws import BinanceKlineStream
    from src.live.nyx_runtime import NYXRuntime

    if not MODELS_DIR.is_dir():
        log.error('No model artefact at %s', MODELS_DIR)
        return 1

    # ONE orchestrator. Everything goes through here.
    runtime = NYXRuntime(
        symbol=SYMBOL.upper(),
        models_dir=MODELS_DIR,
        state_dir=HERE / 'state' / SYMBOL.upper(),
        initial_capital=10_000.0,
    )

    # Recover state from previous run (if any).
    runtime.recover()

    stream = BinanceKlineStream(symbol=SYMBOL, interval='15m')
    last_health_log = time.time()

    def on_message(event: dict) -> None:
        nonlocal last_health_log

        runtime.feed_health.on_event(time.time())
        bar = runtime.bar_builder.on_event(event)
        if bar is None:
            return

        k = event.get('k', {})
        runtime.feed_health.on_bar_timestamp(int(k.get('t', 0)))

        if not runtime.feed_health.is_healthy():
            log.warning('feed unhealthy — skipping bar %s', bar['timestamp'])
            return

        # THE ONLY CALL. Everything happens inside.
        result = runtime.on_bar(bar)

        if result['action'] != 'FLAT':
            log.info('ACTION=%s layers=%s', result['action'],
                     {k: v for k, v in result['layers'].items()
                      if k != 'jesse'})

        now = time.time()
        if now - last_health_log > 300:
            log.info('metrics: %s', runtime.metrics.snapshot())
            last_health_log = now

    def on_error(e: Exception) -> None:
        log.error('WS error: %s', e)

    log.info('starting NYX for %s (single orchestrator)', SYMBOL.upper())

    try:
        stream.run(on_message=on_message, on_error=on_error)
    except KeyboardInterrupt:
        log.info('shutting down...')
        stream.stop()

    runtime.shutdown()
    log.info('done — final metrics: %s', runtime.metrics.snapshot())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
