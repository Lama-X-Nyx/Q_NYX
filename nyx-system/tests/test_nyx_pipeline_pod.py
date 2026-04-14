"""
TDD Tests — `NYXPipelinePod` (task b.A replay mode).

A `SignalPod` that wraps `NYXPipeline.run()` batch output and exposes
it bar-by-bar via `on_bar(bar) -> Signal`. Used to re-validate A/B/C
through the hub-and-spoke architecture (`HubSpokeRunner` +
`PortfolioAllocator` + `PostOnlyPaperBroker`) on HISTORICAL data.

This is NOT the live real-time path — that is task (b.B). The replay
pod pre-computes all trades in batch, indexes them by timestamp, and
returns actionable Signals only at bars that correspond to a
pre-computed trade.
"""
from pathlib import Path

import pytest
import pandas as pd

from tests.conftest import _load_eth_ohlcv  # noqa: F401  (ensures fixture import)


DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'


# ===========================================================================
class TestConstruction:

    def test_pod_builds_on_eth_window(self, eth_mtf_data, eth_mtf_features):
        """Pod should construct without error on ETH 2022-2023 window."""
        from src.assets.nyx_pipeline_pod import NYXPipelinePod
        pod = NYXPipelinePod(
            symbol='ETHUSDT',
            mtf_data=eth_mtf_data,
            mtf_features=eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-12-31',
        )
        assert pod.symbol == 'ETHUSDT'
        # Pod must have produced SOME trades during construction.
        assert pod.n_precomputed_trades >= 5, \
            f"expected ≥ 5 trades, got {pod.n_precomputed_trades}"

    def test_pod_has_cluster_group(self, eth_mtf_data, eth_mtf_features):
        """Pod should expose a cluster group attribute used by allocator."""
        from src.assets.nyx_pipeline_pod import NYXPipelinePod
        pod = NYXPipelinePod(
            symbol='ETHUSDT',
            mtf_data=eth_mtf_data,
            mtf_features=eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-12-31',
            cluster_group='alts',
        )
        assert pod.cluster_group == 'alts'


# ===========================================================================
class TestOnBarContract:
    """`on_bar(bar)` must return a typed `Signal`."""

    @pytest.fixture(scope='class')
    def built_pod(self, eth_mtf_data, eth_mtf_features):
        from src.assets.nyx_pipeline_pod import NYXPipelinePod
        return NYXPipelinePod(
            symbol='ETHUSDT',
            mtf_data=eth_mtf_data,
            mtf_features=eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-12-31',
            cluster_group='alts',
        )

    def test_unknown_timestamp_returns_flat(self, built_pod):
        """Bar that does not match any precomputed trade → direction=0."""
        from src.assets.signal import Signal
        bar = {
            'timestamp': '1999-01-01T00:00:00',  # way outside test window
            'open': 100, 'high': 101, 'low': 99,
            'close': 100, 'volume': 1.0,
        }
        sig = built_pod.on_bar(bar)
        assert isinstance(sig, Signal)
        assert sig.direction == 0

    def test_trade_timestamp_returns_actionable(self, built_pod):
        """Bar at a precomputed trade timestamp → direction != 0."""
        from src.assets.signal import Signal
        ts = built_pod._precomputed_trade_timestamps[0]  # first trade ts
        bar = {
            'timestamp': ts,
            'open': 1000, 'high': 1005, 'low': 995,
            'close': 1002, 'volume': 1.0,
        }
        sig = built_pod.on_bar(bar)
        assert isinstance(sig, Signal)
        assert sig.direction in (-1, +1)

    def test_all_trades_are_reachable(self, built_pod):
        """For every precomputed trade timestamp, the Pod emits
        direction != 0 when asked at that timestamp."""
        reached = 0
        for ts in built_pod._precomputed_trade_timestamps:
            bar = {
                'timestamp': ts, 'open': 1000, 'high': 1005, 'low': 995,
                'close': 1002, 'volume': 1.0,
            }
            sig = built_pod.on_bar(bar)
            if sig.direction != 0:
                reached += 1
        total = built_pod.n_precomputed_trades
        assert reached == total, \
            f"only {reached}/{total} trades reachable via on_bar"


# ===========================================================================
class TestSignalPodProtocol:
    """Pod must satisfy the HubSpokeRunner `SignalPod` protocol."""

    def test_has_symbol_attr(self, eth_mtf_data, eth_mtf_features):
        from src.assets.nyx_pipeline_pod import NYXPipelinePod
        pod = NYXPipelinePod(
            symbol='ETHUSDT',
            mtf_data=eth_mtf_data, mtf_features=eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01', test_end='2023-12-31',
        )
        assert isinstance(pod.symbol, str)

    def test_on_bar_is_callable(self, eth_mtf_data, eth_mtf_features):
        from src.assets.nyx_pipeline_pod import NYXPipelinePod
        pod = NYXPipelinePod(
            symbol='ETHUSDT',
            mtf_data=eth_mtf_data, mtf_features=eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01', test_end='2023-12-31',
        )
        assert callable(pod.on_bar)


# ===========================================================================
class TestHubSpokeIntegration:
    """Pod → HubSpokeRunner → post-only broker → fills / misses measured."""

    def test_hub_spoke_runs_with_pod(self, tmp_path,
                                     eth_mtf_data, eth_mtf_features):
        from src.assets.nyx_pipeline_pod import NYXPipelinePod
        from src.assets.hub_spoke_runner import HubSpokeRunner

        pod = NYXPipelinePod(
            symbol='ETHUSDT',
            mtf_data=eth_mtf_data, mtf_features=eth_mtf_features,
            train_end='2022-12-31',
            test_start='2023-01-01', test_end='2023-12-31',
            cluster_group='alts',
        )
        runner = HubSpokeRunner(
            pods=[pod],
            storage_root=tmp_path,
            max_total_risk=0.04,
            max_asset_risk=0.015,
            max_cluster_risk={'alts': 0.02},
            max_open_positions=2,
        )
        try:
            # Feed a few bars at pre-computed trade timestamps.
            for ts in pod._precomputed_trade_timestamps[:3]:
                bar = {
                    'timestamp': ts,
                    'open': 1000, 'high': 1005, 'low': 995,
                    'close': 1002, 'volume': 1.0,
                    'atr': 10.0, 'volume_ratio': 3.0,
                }
                runner.on_bars({'ETHUSDT': bar})
            # No exception + bars processed.
            assert runner._runners['ETHUSDT'].bars_processed == 3
        finally:
            runner.shutdown()
