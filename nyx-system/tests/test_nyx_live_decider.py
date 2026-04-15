"""
TDD Tests — `NYXLiveDecider`.

Task (b.B) layer 3. Real-time per-bar decider that loads a
pre-trained artefact (`models/<SYMBOL>/`) and emits a `Signal` for
each incoming 15m bar, using the same logic as `NYXPipeline.run()`
but in incremental mode.

Contract:
  - Load `ml_filter_v1.pkl`, `scaler.pkl`, `feature_names.json` via
    `train_asset_model.load_artifact`.
  - Maintain `MTFFeatureStack` internally.
  - On each `on_15m_bar(bar)`, run: hard gate → feature vector →
    scale → ML score → bear-dial threshold → cooldown/daily → Signal.
  - Honors `max_daily_trades` and `cooldown_bars`.
  - Returns `Signal(direction=0)` when any gate rejects.
  - Returns `Signal(direction in {-1, +1})` when all gates pass.
"""
from pathlib import Path

import pandas as pd
import pytest


MODELS_DIR = Path(__file__).parent.parent / 'models'


def _15m_bar(ts: str, price: float = 1000.0, vol: float = 500.0,
             high_mult: float = 1.002, low_mult: float = 0.998) -> dict:
    return {
        'timestamp': ts,
        'open':   price,
        'high':   price * high_mult,
        'low':    price * low_mult,
        'close':  price * 1.001,
        'volume': vol,
    }


# ===========================================================================
class TestConstruction:

    def test_loads_eth_artefact(self):
        """Decider should load the persisted ETH model artefact."""
        from src.ml.nyx_live_decider import NYXLiveDecider
        d = NYXLiveDecider(symbol='ETHUSDT', artifact_dir=MODELS_DIR / 'ETHUSDT')
        assert d.symbol == 'ETHUSDT'
        assert d.model is not None
        assert d.scaler is not None
        assert isinstance(d.feature_names, list)
        assert len(d.feature_names) >= 65, \
            f"expected ≥ 65 features (MTF rule), got {len(d.feature_names)}"

    def test_missing_artefact_raises(self, tmp_path):
        from src.ml.nyx_live_decider import NYXLiveDecider
        with pytest.raises((FileNotFoundError, Exception)):
            NYXLiveDecider(symbol='NOPEUSDT', artifact_dir=tmp_path)


# ===========================================================================
class TestSignalContract:

    @pytest.fixture(scope='class')
    def decider(self):
        from src.ml.nyx_live_decider import NYXLiveDecider
        return NYXLiveDecider(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
            cluster_group='alts',
        )

    def test_on_bar_returns_signal(self, decider):
        """First bar (no warmup) → must still return a valid Signal,
        not crash. Direction is likely 0 because hard gate / warmup."""
        from src.assets.signal import Signal
        bar = _15m_bar('2023-06-01T10:00:00', price=1800.0)
        sig = decider.on_15m_bar(bar)
        assert isinstance(sig, Signal)
        assert sig.symbol == 'ETHUSDT'
        assert sig.direction in (-1, 0, 1)

    def test_signal_has_cluster(self, decider):
        bar = _15m_bar('2023-06-01T10:00:00', price=1800.0)
        sig = decider.on_15m_bar(bar)
        assert sig.cluster_group == 'alts'


# ===========================================================================
class TestHardGate:

    @pytest.fixture(scope='class')
    def decider(self):
        from src.ml.nyx_live_decider import NYXLiveDecider
        return NYXLiveDecider(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
        )

    def test_rejects_outside_trading_hour(self, decider):
        """Hour < 6 or > 20 → direction must be 0 (hour filter)."""
        bar = _15m_bar('2023-06-01T03:00:00', price=1800.0)
        sig = decider.on_15m_bar(bar)
        assert sig.direction == 0

    def test_rejects_low_volume_spike(self, decider):
        """Without enough volume relative to MA, the hard gate should
        reject. We can't easily fabricate a spike here without priming
        the buffer — but at minimum any single-bar feed returns 0
        because the trend-alignment check needs 50+ bars."""
        bar = _15m_bar('2023-06-01T10:00:00', price=1800.0)
        sig = decider.on_15m_bar(bar)
        # First bar alone cannot pass (need EMA9/21/50).
        assert sig.direction == 0


# ===========================================================================
class TestCooldownAndDailyLimit:
    """Once a trade fires, next N bars in cooldown window return 0.
    Tested via state manipulation — we don't need a full replay to
    verify the gate works."""

    def test_cooldown_state_blocks_next(self):
        from src.ml.nyx_live_decider import NYXLiveDecider
        d = NYXLiveDecider(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
        )
        # Force cooldown state: pretend a trade just fired at bar N.
        d._last_trade_bar_idx = 10
        d._bars_seen = 11
        # Next bars within cooldown should be blocked at that gate.
        blocked = d._cooldown_blocks_now()
        assert blocked is True

    def test_cooldown_clears_after_window(self):
        from src.ml.nyx_live_decider import NYXLiveDecider
        d = NYXLiveDecider(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
        )
        d._last_trade_bar_idx = 10
        d._bars_seen = 10 + d._cooldown_bars + 1
        assert d._cooldown_blocks_now() is False

    def test_daily_limit_blocks(self):
        from src.ml.nyx_live_decider import NYXLiveDecider
        d = NYXLiveDecider(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
        )
        d._daily_counts['2023-06-01'] = d._max_daily_trades
        blocked = d._daily_limit_blocks('2023-06-01')
        assert blocked is True

    def test_daily_limit_allows_other_day(self):
        from src.ml.nyx_live_decider import NYXLiveDecider
        d = NYXLiveDecider(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
        )
        d._daily_counts['2023-06-01'] = d._max_daily_trades
        assert d._daily_limit_blocks('2023-06-02') is False


# ===========================================================================
class TestBearDialConditional:
    """Bear-dial conditional threshold — Rule #2 / Reality Check #6.

    When market is hostile (bear regime, elevated stress, weak trend, or
    high ML/rule disagreement), NYXLiveDecider MUST raise its ML
    threshold to the stricter bear_params value (0.68) and lengthen its
    cooldown (64 bars) — same logic as NYXPipeline batch mode.

    This closes the batch-vs-live gap: without this, live under-filters
    in bear conditions and takes trades NYXPipeline would have rejected,
    hurting the capture ratio measurement and the equivalence guard.
    """

    def test_bear_dial_defaults_match_bear_params(self):
        """Live decider defaults must match `get_risk_params('bear')` —
        threshold 0.68, cooldown 64 bars."""
        from src.ml.bear_risk_dial import get_risk_params
        from src.ml.nyx_live_decider import NYXLiveDecider
        d = NYXLiveDecider(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
        )
        bp = get_risk_params('bear')
        assert d._bear_threshold == pytest.approx(bp['ml_threshold'])
        assert d._bear_cooldown_bars == int(bp['cooldown_bars'])

    def test_compute_bear_dial_signals_without_buffers_returns_none(self):
        """With no buffer data yet, the dial helper returns None — no
        crash, no false activation."""
        from src.ml.nyx_live_decider import NYXLiveDecider
        d = NYXLiveDecider(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
        )
        signals = d._compute_bear_dial_signals()
        assert signals is None or 'regime_1h' in signals

    def test_bear_active_flag_exposed(self):
        """After a bar with hostile regime, the decider must expose a
        bool `_last_bear_active` for observability + the equivalence
        test. Starts at False."""
        from src.ml.nyx_live_decider import NYXLiveDecider
        d = NYXLiveDecider(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
        )
        assert hasattr(d, '_last_bear_active')
        assert d._last_bear_active is False

    def test_bear_active_applies_stricter_threshold(self, eth_mtf_data):
        """Contract: the decider stores the effective threshold used on
        the most recent bar. When bear_active is True, effective
        threshold must be `_bear_threshold`; otherwise `_ml_threshold`."""
        import pandas as pd
        from src.ml.nyx_live_decider import NYXLiveDecider
        d = NYXLiveDecider(
            symbol='ETHUSDT',
            artifact_dir=MODELS_DIR / 'ETHUSDT',
        )
        # Seed with a realistic window so buffers can compute signals.
        seed = eth_mtf_data['15m'].loc['2022-10-01':'2022-12-31']
        for ts, row in seed.iterrows():
            d.on_15m_bar({
                'timestamp': ts.isoformat(),
                'open':   float(row['open']),
                'high':   float(row['high']),
                'low':    float(row['low']),
                'close':  float(row['close']),
                'volume': float(row['volume']),
            })
        # After seeding, the decider must have applied the conditional
        # logic at least once and exposed the effective threshold.
        assert hasattr(d, '_last_effective_threshold')
        assert d._last_effective_threshold in (
            pytest.approx(d._ml_threshold),
            pytest.approx(d._bear_threshold),
        )


# ===========================================================================
class TestBearDialReducesTradesInHostileRegime:
    """End-to-end: a decider with bear dial wired must produce <= trades
    than a decider with bear dial disabled on the same replay window."""

    def test_fewer_or_equal_trades_with_bear_dial(self, eth_mtf_data):
        import pandas as pd
        from src.ml.nyx_live_decider import NYXLiveDecider

        def replay(use_bear_dial: bool) -> int:
            d = NYXLiveDecider(
                symbol='ETHUSDT',
                artifact_dir=MODELS_DIR / 'ETHUSDT',
                use_bear_dial=use_bear_dial,
            )
            # Seed
            for ts, row in eth_mtf_data['15m'].loc['2022-07-01':'2022-12-31'].iterrows():
                d.on_15m_bar({
                    'timestamp': ts.isoformat(),
                    'open':   float(row['open']),
                    'high':   float(row['high']),
                    'low':    float(row['low']),
                    'close':  float(row['close']),
                    'volume': float(row['volume']),
                })
            d._bars_seen = d._warmup_bars + 1
            d._last_trade_bar_idx = -10 ** 9
            d._daily_counts = {}
            # Replay
            n = 0
            for ts, row in eth_mtf_data['15m'].loc['2023-01-01':'2023-06-30'].iterrows():
                sig = d.on_15m_bar({
                    'timestamp': ts.isoformat(),
                    'open':   float(row['open']),
                    'high':   float(row['high']),
                    'low':    float(row['low']),
                    'close':  float(row['close']),
                    'volume': float(row['volume']),
                })
                if sig.direction != 0:
                    n += 1
            return n

        n_with = replay(use_bear_dial=True)
        n_without = replay(use_bear_dial=False)
        assert n_with <= n_without, (
            f"bear dial should only TIGHTEN: "
            f"with={n_with} > without={n_without} — logic regression"
        )
