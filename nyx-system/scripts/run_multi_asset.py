"""
NYX Multi-Asset Live Feed — synchronized portfolio-level allocation.

Ticket 36 architecture:

  Per-asset threads → CandidateStore → CentralOrchestrator → execution

Each asset's NYXRuntime produces CandidateDecisions only (no
direct execution). The CentralOrchestrator collects candidates per
15m time bucket, runs dependency + portfolio allocator on the FULL
set, then dispatches approved trades.

Usage:
  python scripts/run_multi_asset.py
  NYX_ASSETS=BTCUSDT,ETHUSDT python scripts/run_multi_asset.py
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
    candidate_store: object,
    dependency_layer: object,
) -> None:
    """Run a single asset's NYXRuntime in candidate-only mode."""
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
        candidate_store=candidate_store,
        dependency_layer=dependency_layer,
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
        if result['action'] == 'CANDIDATE_EMITTED':
            asset_log.info(
                '[%s] CANDIDATE emitted: conf=%.2f dir=%s',
                config['mode'].upper(),
                result['layers'].get('candidate', {}).get('confidence', 0),
                result['layers'].get('candidate', {}).get('direction', '?'),
            )
        elif result['action'] != 'FLAT':
            asset_log.info('[%s] ACTION=%s', config['mode'].upper(), result['action'])

        now = time.time()
        if now - last_health_log > 300:
            asset_log.info('metrics: %s', runtime.metrics.snapshot())
            last_health_log = now

    def on_error(e: Exception) -> None:
        asset_log.error('WS error: %s', e)

    asset_log.info('starting %s [%s] (candidate mode)', symbol, config['mode'])
    try:
        stream.run(on_message=on_message, on_error=on_error)
    except Exception as exc:
        asset_log.error('stream died: %s', exc)
    finally:
        runtime.shutdown()
        asset_log.info('shutdown complete')


def run_orchestrator(
    candidate_store: object,
    orchestrator: object,
    check_interval: float = 5.0,
) -> None:
    """Poll the candidate store and run allocation cycles."""
    orch_log = logging.getLogger('nyx.orchestrator')
    orch_log.info('orchestrator started')

    while True:
        try:
            store = candidate_store  # type: ignore[assignment]
            for bucket_key in list(store._buckets.keys()):
                if store.is_bucket_ready(bucket_key):
                    candidates = store.consume(bucket_key)
                    if candidates:
                        results = orchestrator.run_cycle(candidates)  # type: ignore[union-attr]
                        approved = [r for r in results if r.get('decision', '').startswith('APPROVE')]
                        orch_log.info(
                            'CYCLE %s: %d candidates → %d approved',
                            bucket_key, len(candidates), len(approved),
                        )
            time.sleep(check_interval)
        except Exception as exc:
            orch_log.error('orchestrator error: %s', exc)
            time.sleep(check_interval)


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

    log.info('NYX Multi-Asset starting (synchronized): %s', list(assets.keys()))

    from src.live.central_orchestrator import CandidateStore, CentralOrchestrator
    from src.live.inter_asset_dependency import InterAssetDependencyLayer
    from src.live.portfolio_allocator import PortfolioAllocator

    dep_layer = InterAssetDependencyLayer(
        symbols=list(assets.keys()),
        leader='BTCUSDT',
        min_bars=100,
    )
    candidate_store = CandidateStore(
        expected_symbols=list(assets.keys()),
        tolerance_seconds=60.0,
    )
    allocator = PortfolioAllocator()
    orchestrator = CentralOrchestrator(
        symbols=list(assets.keys()),
        dependency_layer=dep_layer,
        portfolio_allocator=allocator,
    )

    orch_thread = threading.Thread(
        target=run_orchestrator,
        args=(candidate_store, orchestrator),
        name='nyx-orchestrator',
        daemon=True,
    )
    orch_thread.start()

    threads: Dict[str, threading.Thread] = {}
    for symbol, config in assets.items():
        t = threading.Thread(
            target=run_asset,
            args=(symbol, config, candidate_store, dep_layer),
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
            snap = orchestrator.snapshot()
            log.info('orchestrator: %d cycles completed', snap['total_cycles'])
            time.sleep(30)
    except KeyboardInterrupt:
        log.info('shutting down all assets...')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
