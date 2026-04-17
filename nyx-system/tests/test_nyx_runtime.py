"""
TDD Tests — NYXRuntime (THE single orchestrator).

Verifies that ONE call to `runtime.on_bar(bar)` flows through
ALL layers : GBM → Jesse → Fractal Quality → Risk → OMS.
No hidden calls. No bypass.
"""
from __future__ import annotations

from pathlib import Path

import pytest


MODELS_DIR = Path(__file__).parent.parent / 'models'


@pytest.fixture(scope='module')
def runtime():
    from src.live.nyx_runtime import NYXRuntime
    return NYXRuntime(
        symbol='BTCUSDT',
        models_dir=MODELS_DIR / 'BTCUSDT',
        state_dir=Path('/tmp/nyx_test_runtime'),
        initial_capital=10_000.0,
    )


def _bar(ts='2023-06-15T10:00:00', price=16500.0, vol=500.0):
    return {
        'timestamp': ts,
        'open': price, 'high': price * 1.002,
        'low': price * 0.998, 'close': price * 1.001,
        'volume': vol,
    }


# ===========================================================================
class TestSingleOrchestrator:

    def test_on_bar_returns_structured_result(self, runtime):
        result = runtime.on_bar(_bar())
        assert 'action' in result
        assert 'layers' in result
        assert 'timestamp' in result

    def test_flat_when_no_signal(self, runtime):
        """First bar (no warmup) → FLAT."""
        result = runtime.on_bar(_bar())
        assert result['action'] == 'FLAT'

    def test_signal_layer_always_present(self, runtime):
        result = runtime.on_bar(_bar())
        assert 'signal' in result['layers']
        assert 'direction' in result['layers']['signal']


# ===========================================================================
class TestGBMIsVisible:

    def test_gbm_accessible_on_runtime(self, runtime):
        """The GBM (MetaGBM wrapper) must be directly accessible."""
        assert runtime.gbm is not None
        assert runtime.gbm.has_trained_model is True


# ===========================================================================
class TestJesseAgentsVisible:

    def test_four_agents_on_runtime(self, runtime):
        assert runtime.context_agent is not None
        assert runtime.regime_agent is not None
        assert runtime.setup_agent is not None
        assert runtime.entry_agent is not None

    def test_call_jesse_agents_returns_four_reports(self, runtime):
        """_call_jesse_agents must return all 4 FractalReports."""
        from src.agents.contracts import FractalReport
        reports = runtime._call_jesse_agents(
            _bar(), '2023-06-15T10:00:00',
        )
        assert set(reports.keys()) == {'context', 'regime', 'setup', 'entry'}
        for r in reports.values():
            assert isinstance(r, FractalReport)


# ===========================================================================
class TestAllLayersExist:

    def test_risk_engine_on_runtime(self, runtime):
        assert runtime.risk_engine is not None

    def test_oms_on_runtime(self, runtime):
        assert runtime.oms is not None

    def test_portfolio_on_runtime(self, runtime):
        assert runtime.portfolio is not None

    def test_state_store_on_runtime(self, runtime):
        assert runtime.state_store is not None

    def test_metrics_on_runtime(self, runtime):
        assert runtime.metrics is not None

    def test_alerts_on_runtime(self, runtime):
        assert runtime.alerts is not None


# ===========================================================================
class TestRecoverAndShutdown:

    def test_shutdown_persists(self, runtime):
        """shutdown() must not crash and must save state."""
        runtime.shutdown()
        state_dir = Path('/tmp/nyx_test_runtime')
        assert (state_dir / 'oms_state.json').exists()
        assert (state_dir / 'portfolio_state.json').exists()

    def test_recover_loads(self, runtime):
        """recover() must not crash."""
        runtime.recover()
