"""
TDD Tests — `NYXLivePod`.

Task (b.B) layer 4. Thin adapter that exposes `NYXLiveDecider` via
the `SignalPod` Protocol that `HubSpokeRunner` expects:

    class SignalPod(Protocol):
        symbol: str
        def on_bar(self, bar: dict) -> Signal: ...

The Pod delegates `on_bar(bar)` to `decider.on_15m_bar(bar)`. All
real logic lives in the decider; this layer exists so the
hub-and-spoke runner can drive the live decider without knowing it
is multi-TF.
"""
from pathlib import Path

import pytest


MODELS_DIR = Path(__file__).parent.parent / 'models'


def _bar(ts: str, price: float = 1800.0) -> dict:
    return {
        'timestamp': ts,
        'open':   price,
        'high':   price * 1.002,
        'low':    price * 0.998,
        'close':  price * 1.001,
        'volume': 500.0,
    }


# ===========================================================================
class TestConstruction:

    def test_build_from_eth_artefact(self):
        from src.assets.nyx_live_pod import NYXLivePod
        pod = NYXLivePod(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
            cluster_group='alts',
        )
        assert pod.symbol == 'ETHUSDT'
        assert pod.cluster_group == 'alts'
        assert pod.decider is not None

    def test_exposes_symbol_attr_for_protocol(self):
        """HubSpokeRunner reads `.symbol` from every pod at construction."""
        from src.assets.nyx_live_pod import NYXLivePod
        pod = NYXLivePod(symbol='ETHUSDT',
                         artifact_dir=MODELS_DIR / 'ETHUSDT')
        assert isinstance(pod.symbol, str)


# ===========================================================================
class TestOnBar:

    @pytest.fixture(scope='class')
    def pod(self):
        from src.assets.nyx_live_pod import NYXLivePod
        return NYXLivePod(symbol='ETHUSDT',
                          artifact_dir=MODELS_DIR / 'ETHUSDT',
                          cluster_group='alts')

    def test_on_bar_returns_signal(self, pod):
        from src.assets.signal import Signal
        sig = pod.on_bar(_bar('2023-06-01T10:00:00'))
        assert isinstance(sig, Signal)

    def test_on_bar_delegates_to_decider(self, pod):
        """Two calls must both advance the decider's `_bars_seen`."""
        n0 = pod.decider._bars_seen
        pod.on_bar(_bar('2023-06-01T10:00:00'))
        pod.on_bar(_bar('2023-06-01T10:15:00'))
        assert pod.decider._bars_seen == n0 + 2


# ===========================================================================
class TestHubSpokeIntegration:
    """Pod must be usable by `HubSpokeRunner(pods=[...])`."""

    def test_hub_spoke_can_drive_pod(self, tmp_path):
        from src.assets.nyx_live_pod import NYXLivePod
        from src.assets.hub_spoke_runner import HubSpokeRunner

        pod = NYXLivePod(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
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
            # Feed 5 bars, none expected to pass hard gate without warmup.
            for i in range(5):
                ts = f'2023-06-01T{10 + (i // 4):02d}:{(i % 4) * 15:02d}:00'
                runner.on_bars({'ETHUSDT': _bar(ts)})
            assert runner._runners['ETHUSDT'].bars_processed == 5
        finally:
            runner.shutdown()
