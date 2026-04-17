"""
NYX Multi-Asset Live Feed — one NYXRuntime per asset.

Spawns one independent NYXRuntime per configured asset.
Each runtime has its own model, OMS, portfolio, risk engine,
state store, and metrics — full isolation.

BTC: live (real execution)
ETH: paper (PostOnlyPaperBroker, no exchange)
SOL: paper (PostOnlyPaperBroker, no exchange)

Usage:
  python scripts/run_multi_asset.py

Each asset connects to its own Binance WS kline stream.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
log = logging.getLogger('nyx.multi')

ASSETS = {
    'BTCUSDT': {'mode': 'live', 'capital': 10_000.0},
    'ETHUSDT': {'mode': 'paper', 'capital': 10_000.0},
    'SOLUSDT': {'mode': 'paper', 'capital': 10_000.0},
}


def run_asset(
    symbol: str, config: dict,
    dependency_layer: object = None,
    portfolio_allocator: object = None,
) -> None:
    """Run a single asset's NYXRuntime in its own thread."""
    from src.live.binance_ws import BinanceKlineStream
    from src.live.nyx_runtime import NYXRuntime

    asset_log = logging.getLogger(f'nyx.{symbol}')
    models_dir = HERE / 'models' / symbol
    if not models_dir.is_dir():
        asset_log.error('No model artefact at %s — skipping', models_dir)
        return

    runtime = NYXRuntime(
        symbol=symbol,
        models_dir=models_dir,
        state_dir=HERE / 'state' / symbol,
        initial_capital=config['capital'],
        dependency_layer=dependency_layer,
        portfolio_allocator=portfolio_allocator,
    )
    runtime.recover()
    runtime.start()

    ws_symbol = symbol.lower()
    stream = BinanceKlineStream(symbol=ws_symbol, interval='15m')
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
            asset_log.warning('feed unhealthy — skipping bar')
            return

        result = runtime.on_bar(bar)
        if result['action'] != 'FLAT':
            asset_log.info(
                '[%s] ACTION=%s layers=%s',
                config['mode'].upper(), result['action'],
                {k: v for k, v in result['layers'].items() if k != 'jesse'},
            )

        now = time.time()
        if now - last_health_log > 300:
            asset_log.info('metrics: %s', runtime.metrics.snapshot())
            last_health_log = now

    def on_error(e: Exception) -> None:
        asset_log.error('WS error: %s', e)

    asset_log.info('starting %s [%s]', symbol, config['mode'])
    try:
        stream.run(on_message=on_message, on_error=on_error)
    except Exception as exc:
        asset_log.error('stream died: %s', exc)
    finally:
        runtime.shutdown()
        asset_log.info('shutdown complete')


def main() -> int:
    selected = os.environ.get('NYX_ASSETS', '')
    if selected:
        symbols = [s.strip().upper() for s in selected.split(',')]
        assets = {s: ASSETS[s] for s in symbols if s in ASSETS}
    else:
        assets = ASSETS

    if not assets:
        log.error('No valid assets configured')
        return 1

    for sym in assets:
        model_dir = HERE / 'models' / sym
        if not model_dir.is_dir():
            log.error('Missing model: %s', model_dir)
            return 1

    log.info('NYX Multi-Asset starting: %s', list(assets.keys()))

    from src.live.inter_asset_dependency import InterAssetDependencyLayer
    from src.live.portfolio_allocator import PortfolioAllocator
    dep_layer = InterAssetDependencyLayer(
        symbols=list(assets.keys()),
        leader='BTCUSDT',
        min_bars=100,
    )
    allocator = PortfolioAllocator()

    threads: Dict[str, threading.Thread] = {}
    for symbol, config in assets.items():
        t = threading.Thread(
            target=run_asset,
            args=(symbol, config, dep_layer, allocator),
            name=f'nyx-{symbol}',
            daemon=True,
        )
        threads[symbol] = t
        t.start()

    try:
        while True:
            alive = {s: t.is_alive() for s, t in threads.items()}
            dead = [s for s, a in alive.items() if not a]
            if dead:
                log.warning('dead threads: %s', dead)
            if all(not a for a in alive.values()):
                log.error('all threads dead — exiting')
                return 1
            time.sleep(30)
    except KeyboardInterrupt:
        log.info('shutting down all assets...')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
