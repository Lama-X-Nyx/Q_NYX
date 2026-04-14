"""
TDD Tests — ExecutionProfile

Turns a generic `Asset` (from AssetRegistry) into concrete execution
parameters for the PostOnlyPaperBroker:

  - maker_fee, taker_fee       — directly from Asset
  - max_wait_bars              — derived from session_profile:
      liquid         → 4 bars (tolerant)
      event_sensitive → 3 bars (strict)
      momentum_fast  → 2 bars (very strict, price escapes fast)
  - slippage_bps               — derived from slippage_model:
      low            → 2  bps
      medium         → 5  bps
      medium_high    → 8  bps
      high           → 12 bps
  - maker_viability_threshold  — derived from slippage_model:
      more friction → harder threshold before we accept post-only

Values are not magic — each is motivated by the architecture doc:
  "SOL: timeout plus court, seuil maker plus dur"
  "ETH: plus tolérant, meilleur candidat pour démarrer"
  "XRP: attention aux impulsions news, filtre spread/vol plus strict"
"""
import pytest


def _asset(slippage='medium', session='liquid'):
    from src.assets.registry import Asset
    return Asset(
        symbol='XYZUSDT', enabled=True, group='majors',
        price_precision=2, qty_precision=3,
        maker_fee=0.0002, taker_fee=0.0004,
        slippage_model=slippage, session_profile=session,
        risk_cap=0.4, model_profile='xyz_v1',
    )


class TestDerivations:

    def test_max_wait_per_session(self):
        from src.assets.execution_profile import ExecutionProfile
        assert ExecutionProfile.from_asset(_asset(session='liquid')).max_wait_bars == 4
        assert ExecutionProfile.from_asset(_asset(session='event_sensitive')).max_wait_bars == 3
        assert ExecutionProfile.from_asset(_asset(session='momentum_fast')).max_wait_bars == 2

    def test_slippage_bps_per_model(self):
        from src.assets.execution_profile import ExecutionProfile
        assert ExecutionProfile.from_asset(_asset(slippage='low')).slippage_bps == 2
        assert ExecutionProfile.from_asset(_asset(slippage='medium')).slippage_bps == 5
        assert ExecutionProfile.from_asset(_asset(slippage='medium_high')).slippage_bps == 8
        assert ExecutionProfile.from_asset(_asset(slippage='high')).slippage_bps == 12

    def test_maker_viability_threshold_harder_with_slippage(self):
        from src.assets.execution_profile import ExecutionProfile
        low_fric = ExecutionProfile.from_asset(_asset(slippage='low'))
        high_fric = ExecutionProfile.from_asset(_asset(slippage='high'))
        assert high_fric.maker_viability_threshold > low_fric.maker_viability_threshold

    def test_fees_pass_through(self):
        from src.assets.execution_profile import ExecutionProfile
        ep = ExecutionProfile.from_asset(_asset())
        assert ep.maker_fee == 0.0002
        assert ep.taker_fee == 0.0004


class TestAssetSpecificProfiles:
    """Spot-check the concrete profiles for ETH / XRP / SOL match the
    architectural intent."""

    def _real_asset(self, sym):
        from pathlib import Path
        from src.assets.registry import AssetRegistry
        reg = AssetRegistry.load(Path(__file__).parent.parent / 'config' / 'assets.yaml')
        return reg.get(sym)

    def test_eth_is_most_tolerant(self):
        from src.assets.execution_profile import ExecutionProfile
        eth = ExecutionProfile.from_asset(self._real_asset('ETHUSDT'))
        sol = ExecutionProfile.from_asset(self._real_asset('SOLUSDT'))
        assert eth.max_wait_bars > sol.max_wait_bars
        assert eth.maker_viability_threshold < sol.maker_viability_threshold

    def test_sol_has_tightest_timeout(self):
        from src.assets.execution_profile import ExecutionProfile
        for sym in ('ETHUSDT', 'XRPUSDT'):
            other = ExecutionProfile.from_asset(self._real_asset(sym))
            sol = ExecutionProfile.from_asset(self._real_asset('SOLUSDT'))
            assert sol.max_wait_bars <= other.max_wait_bars

    def test_xrp_stricter_than_eth(self):
        """XRP sits between ETH and SOL in strictness."""
        from src.assets.execution_profile import ExecutionProfile
        eth = ExecutionProfile.from_asset(self._real_asset('ETHUSDT'))
        xrp = ExecutionProfile.from_asset(self._real_asset('XRPUSDT'))
        assert xrp.max_wait_bars < eth.max_wait_bars
        assert xrp.slippage_bps > eth.slippage_bps
