"""
TDD Tests — Agent 2: JesseRegimeAgent (1H)

Rôle: identification du régime de marché dominant.
  - trend_plus → momentum haussier fort
  - trend_minus → momentum baissier fort
  - range → consolidation latérale
  - squeeze → compression vol, block entries
  - distribution / liquidation → danger, block entries

Specs (COMPLETE_ARCHITECTURE_SUMMARY.md):
  - Input:    OHLCV 1H + HSMM gamma (6 probs)
  - Features: momentum 4/12/48h + vol + HSMM proba (6)
  - Label:    max high > 1% dans 4 barres → 1, else 0
  - Output:   AgentResult(agent='regime')

Tests:
  1. Contract compliance
  2. Trend detection on synthetic data
  3. Range/squeeze detection
  4. Features include HSMM probs
  5. Standalone backtest per regime type
"""
import pytest
import numpy as np
import pandas as pd


def make_hourly_bull(n: int = 500, start: float = 20000.0) -> pd.DataFrame:
    np.random.seed(42)
    pct = 0.001 + np.random.randn(n) * 0.0005
    close = start * np.exp(np.cumsum(pct))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.003 + 0.001)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.002)
    open_ = np.roll(close, 1); open_[0] = start
    volume = np.random.randint(500, 3000, n).astype(float)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=pd.date_range('2023-01-01', periods=n, freq='1h'))


def make_hourly_bear(n: int = 500, start: float = 40000.0) -> pd.DataFrame:
    np.random.seed(43)
    pct = -0.001 + np.random.randn(n) * 0.0005
    close = start * np.exp(np.cumsum(pct))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.002)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.003 - 0.001)
    open_ = np.roll(close, 1); open_[0] = start
    volume = np.random.randint(500, 3000, n).astype(float)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=pd.date_range('2023-04-01', periods=n, freq='1h'))


def make_hourly_range(n: int = 500, center: float = 30000.0) -> pd.DataFrame:
    np.random.seed(44)
    close = center * (1 + 0.002 * np.sin(np.linspace(0, 10 * np.pi, n))
                      + np.random.randn(n) * 0.0003)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.001 + 0.0005)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.001 - 0.0005)
    open_ = close * (1 + np.random.randn(n) * 0.0002)
    volume = np.random.randint(500, 3000, n).astype(float)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=pd.date_range('2023-07-01', periods=n, freq='1h'))


def make_mixed_hourly():
    return pd.concat([make_hourly_bull(400), make_hourly_bear(400),
                      make_hourly_range(400)]).sort_index()


# ===========================================================================
# TEST 1: Contract
# ===========================================================================
class TestRegimeContract:

    def test_returns_agent_result(self):
        from src.ml.jesse_agents import JesseRegimeAgent
        from src.agents.contracts import AgentResult
        agent = JesseRegimeAgent()
        agent.train(make_mixed_hourly())
        result = agent.analyze(make_hourly_bull(200))
        assert isinstance(result, AgentResult)
        assert result.agent == 'regime'

    def test_state_is_valid_regime(self):
        from src.ml.jesse_agents import JesseRegimeAgent
        agent = JesseRegimeAgent()
        agent.train(make_mixed_hourly())
        result = agent.analyze(make_hourly_bull(200))
        valid_states = ('trend_plus', 'trend_minus', 'range', 'squeeze',
                        'distribution', 'liquidation')
        assert result.state in valid_states, f"Invalid state: {result.state}"

    def test_metadata_has_regime_info(self):
        from src.ml.jesse_agents import JesseRegimeAgent
        agent = JesseRegimeAgent()
        agent.train(make_mixed_hourly())
        result = agent.analyze(make_hourly_bull(200))
        assert 'dominant_state' in result.metadata
        assert 'adx' in result.metadata


# ===========================================================================
# TEST 2: Detection
# ===========================================================================
class TestRegimeDetection:

    def test_trend_plus_on_uptrend(self):
        from src.ml.jesse_agents import JesseRegimeAgent
        agent = JesseRegimeAgent()
        agent.train(make_mixed_hourly())
        result = agent.analyze(make_hourly_bull(300))
        assert result.state == 'trend_plus', f"Expected trend_plus, got {result.state}"
        assert result.passed is True

    def test_trend_minus_on_downtrend(self):
        from src.ml.jesse_agents import JesseRegimeAgent
        agent = JesseRegimeAgent()
        agent.train(make_mixed_hourly())
        result = agent.analyze(make_hourly_bear(300))
        assert result.state == 'trend_minus', f"Expected trend_minus, got {result.state}"

    def test_range_on_sideways(self):
        from src.ml.jesse_agents import JesseRegimeAgent
        agent = JesseRegimeAgent()
        agent.train(make_mixed_hourly())
        result = agent.analyze(make_hourly_range(300))
        assert result.state in ('range', 'squeeze'), f"Expected range/squeeze, got {result.state}"


# ===========================================================================
# TEST 3: Features
# ===========================================================================
class TestRegimeFeatures:

    def test_features_include_hsmm_cols(self):
        from src.ml.jesse_agents import JesseRegimeAgent
        agent = JesseRegimeAgent()
        features = agent.compute_features(make_hourly_bull(200))
        hsmm_cols = [c for c in features.columns if c.startswith('hsmm_')]
        assert len(hsmm_cols) == 6, f"Expected 6 HSMM cols, got {len(hsmm_cols)}"

    def test_features_stationary(self):
        from src.ml.jesse_agents import JesseRegimeAgent
        agent = JesseRegimeAgent()
        features = agent.compute_features(make_hourly_bull(300))
        for col in features.columns:
            valid = features[col].dropna()
            if len(valid) > 0:
                assert valid.abs().max() < 100, f"Feature '{col}' not stationary"


# ===========================================================================
# TEST 4: Standalone Backtest
# ===========================================================================
class TestRegimeBacktest:

    def test_standalone_backtest_on_bull(self):
        from src.ml.jesse_agents import JesseRegimeAgent
        agent = JesseRegimeAgent()
        mixed = make_mixed_hourly()
        agent.train(mixed)
        bull = make_hourly_bull(200)
        results = []
        for i in range(1, len(bull) + 1):
            r = agent.analyze(bull.iloc[:i])
            results.append(r)
        trend_plus_pct = sum(1 for r in results if r.state == 'trend_plus') / len(results)
        assert trend_plus_pct >= 0.3, \
            f"Only {trend_plus_pct:.0%} trend_plus on bull 1H data"

    def test_standalone_backtest_returns_metrics(self):
        from src.ml.jesse_agents import JesseRegimeAgent
        agent = JesseRegimeAgent()
        mixed = make_mixed_hourly()
        metrics = agent.backtest(mixed, train_ratio=0.75)
        assert 'n_bars' in metrics
        assert 'state_distribution' in metrics
        assert metrics['n_bars'] > 0
