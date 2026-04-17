"""
BinanceKlineStream — thin WebSocket adapter for Binance kline
events (Ticket 22).

Connects to `wss://stream.binance.com:9443/ws/<symbol>@kline_<interval>`
and yields raw kline events. Handles reconnect with exponential
backoff. No exchange logic leaks past this layer.

The `on_message` callback receives the raw JSON dict — the caller
is responsible for normalization via `BarBuilder`.

Requirements at runtime : `websocket-client` (pip install websocket-client).
Tests mock the connection — no real network needed.

Usage :
    stream = BinanceKlineStream('btcusdt', '15m')
    stream.run(on_message=my_callback)  # blocking
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

WS_BASE = 'wss://stream.binance.com:9443/ws'


class BinanceKlineStream:
    """Thin Binance WebSocket kline client with reconnect."""

    def __init__(
        self,
        symbol: str = 'btcusdt',
        interval: str = '15m',
        max_reconnect_attempts: int = 10,
        base_backoff_s: float = 2.0,
    ) -> None:
        self.symbol = symbol.lower()
        self.interval = interval
        self.url = f'{WS_BASE}/{self.symbol}@kline_{self.interval}'
        self._max_reconnect = int(max_reconnect_attempts)
        self._base_backoff = float(base_backoff_s)
        self._running = False

    def run(
        self,
        on_message: Callable[[Dict[str, Any]], None],
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """Blocking event loop. Reconnects on failure.

        `on_message` receives a parsed JSON dict for each WS message.
        `on_error` is called on connection errors (optional).

        Call `stop()` from another thread to exit cleanly.
        """
        try:
            import websocket  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                'websocket-client is required for live feed. '
                'Install via: pip install websocket-client'
            )

        self._running = True
        attempt = 0

        while self._running and attempt < self._max_reconnect:
            try:
                log.info('connecting to %s (attempt %d)', self.url, attempt + 1)
                ws = websocket.WebSocket()
                ws.connect(self.url)
                attempt = 0  # reset on successful connect
                log.info('connected')

                while self._running:
                    raw = ws.recv()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning('malformed WS payload, skipping')
                        continue
                    on_message(data)

            except Exception as e:
                attempt += 1
                wait = self._base_backoff * (2 ** (attempt - 1))
                log.warning(
                    'WS error: %s — reconnecting in %.1fs (attempt %d/%d)',
                    e, wait, attempt, self._max_reconnect,
                )
                if on_error:
                    on_error(e)
                time.sleep(wait)
            finally:
                try:
                    ws.close()  # type: ignore[possibly-undefined]
                except Exception:
                    pass

        self._running = False
        log.info('stream stopped')

    def stop(self) -> None:
        """Signal the run loop to exit cleanly."""
        self._running = False
