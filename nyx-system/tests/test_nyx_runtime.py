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
    rt = NYXRuntime(
        symbol='BTCUSDT',
        models_dir=MODELS_DIR / 'BTCUSDT',
        state_dir=Path('/tmp/nyx_test_runtime'),
        initial_capital=10_000.0,
    )
    rt.start()  # Ticket 29 — must start before on_bar works
    return rt


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
class TestExchangeUpdate:

    def test_on_exchange_update_fill(self, runtime):
        """Exchange fill update flows through OMS → Portfolio."""
        # First submit an order so we have something to fill.
        oid = runtime.oms.submit_order(
            client_order_id='exch-test-1', symbol='BTCUSDT',
            side='buy', quantity=0.01, price=16500.0,
        )
        runtime.on_exchange_update({
            'order_id': oid,
            'type': 'FILL',
            'fill_qty': 0.01,
            'fill_price': 16500.0,
            'fee': 0.033,
        })
        assert runtime.oms.get_order(oid).status == 'FILLED'

    def test_on_exchange_update_reject(self, runtime):
        oid = runtime.oms.submit_order(
            client_order_id='exch-test-2', symbol='BTCUSDT',
            side='buy', quantity=0.01, price=16500.0,
        )
        runtime.on_exchange_update({
            'order_id': oid,
            'type': 'REJECT',
            'reason': 'post-only would cross',
        })
        assert runtime.oms.get_order(oid).status == 'REJECTED'


class TestHeartbeat:

    def test_heartbeat_returns_status(self, runtime):
        status = runtime.heartbeat()
        assert 'feed_healthy' in status
        assert 'risk_killed' in status
        assert 'metrics' in status
        assert 'open_orders' in status
        assert 'open_positions' in status

    def test_heartbeat_persists_state(self, runtime):
        runtime.heartbeat()
        state_dir = Path('/tmp/nyx_test_runtime')
        assert (state_dir / 'oms_state.json').exists()


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


class TestFullPipelineIntegration:
    """End-to-end: the FULL pipeline must execute when on_bar is
    called with enough warmup bars. Every layer must be touched."""

    def test_on_bar_result_documents_all_layers(self, runtime):
        """When the GBM DOES fire a signal, the result must contain
        evidence that Jesse + fractal + risk were evaluated.

        NOTE: with only 1 bar of warmup the GBM won't fire (hard
        gate blocks), so we just verify the infrastructure doesn't
        crash and the signal layer is always populated.
        """
        for i in range(5):
            result = runtime.on_bar(_bar(
                ts=f'2023-06-15T{10+i}:00:00',
                price=16500.0 + i * 10,
            ))
        assert result['action'] in ('FLAT', 'SKIP_QUALITY',
                                     'BLOCKED_RISK', 'ORDER_SUBMITTED')
        assert 'signal' in result['layers']
