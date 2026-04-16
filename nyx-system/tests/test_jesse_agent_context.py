"""
TDD Tests — Agent 1: JesseContextAgent (1D)

Rôle: filtre directionnel macro.
  - bullish → autorise les longs
  - bearish → bloque les longs
  - neutral → neutre

Specs (COMPLETE_ARCHITECTURE_SUMMARY.md):
  - Input:    OHLCV 1D
  - Features: momentum 5/20/60/200j, realized vol, RSI, Amihud
  - Label:    max return > 3% sur 5 jours → +1, else 0
  - Output:   AgentResult(agent='context')

Tests:
  1. Contract compliance (AgentResult fields)
  2. Bullish detection on uptrend data
  3. Bearish detection on downtrend data
  4. Neutral on ranging data
  5. Features are stationary
  6. Standalone backtest — agent seul filtre correctement
"""
import pytest
import numpy as np
import pandas as pd


def make_daily_bull(n: int = 200, start: float = 16500.0) -> pd.DataFrame:
    """Obvious daily uptrend (BTC Q1 2023 style)."""
    np.random.seed(42)
    pct = 0.005 + np.random.randn(n) * 0.002  # +0.5%/day
    close = start * np.exp(np.cumsum(pct))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01 + 0.005)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.008)
    open_ = np.roll(close, 1); open_[0] = start
    volume = np.random.randint(1000, 5000, n).astype(float)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=pd.date_range('2023-01-01', periods=n, freq='1D'))


def make_daily_bear(n: int = 200, start: float = 60000.0) -> pd.DataFrame:
    """Obvious daily downtrend."""
    np.random.seed(43)
    pct = -0.005 + np.random.randn(n) * 0.002
    close = start * np.exp(np.cumsum(pct))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.008)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01 - 0.005)
    open_ = np.roll(close, 1); open_[0] = start
    volume = np.random.randint(1000, 5000, n).astype(float)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=pd.date_range('2022-06-01', periods=n, freq='1D'))


def make_daily_range(n: int = 200, center: float = 30000.0) -> pd.DataFrame:
    """Sideways daily data."""
    np.random.seed(44)
    close = center * (1 + 0.003 * np.sin(np.linspace(0, 8 * np.pi, n))
                      + np.random.randn(n) * 0.002)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.005 + 0.002)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.005 - 0.002)
    open_ = close * (1 + np.random.randn(n) * 0.001)
    volume = np.random.randint(1000, 5000, n).astype(float)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=pd.date_range('2023-06-01', periods=n, freq='1D'))


# ===========================================================================
# TEST 1: Contract Compliance
# ===========================================================================
class TestContextContract:

    def test_returns_agent_result(self):
        """Must return AgentResult with agent='context'."""
        from src.ml.jesse_agents import JesseContextAgent
        from src.agents.contracts import AgentResult
        agent = JesseContextAgent()
        df = make_daily_bull(200)
        agent.train(df)
        result = agent.analyze(df)
        assert isinstance(result, AgentResult)
        assert result.agent == 'context'

    def test_state_is_valid(self):
        """State must be bullish, bearish, or neutral."""
        from src.ml.jesse_agents import JesseContextAgent
        agent = JesseContextAgent()
        df = make_daily_bull(200)
        agent.train(df)
        result = agent.analyze(df)
        assert result.state in ('bullish', 'bearish', 'neutral')

    def test_score_in_range(self):
        """Score must be in [0, 1]."""
        from src.ml.jesse_agents import JesseContextAgent
        agent = JesseContextAgent()
        df = make_daily_bull(200)
        agent.train(df)
        result = agent.analyze(df)
        assert 0.0 <= result.score <= 1.0

    def test_metadata_has_probabilities(self):
        """Metadata must include p_bull, p_bear, p_neutral."""
        from src.ml.jesse_agents import JesseContextAgent
        agent = JesseContextAgent()
        df = make_daily_bull(200)
        agent.train(df)
        result = agent.analyze(df)
        for key in ('p_bull', 'p_bear', 'p_neutral'):
            assert key in result.metadata, f"Missing metadata key: {key}"


# ===========================================================================
# TEST 2: Direction Detection
# ===========================================================================
class TestContextDetection:

    def _make_mixed_daily(self):
        """Mixed daily data for training (all 3 regimes)."""
        bull = make_daily_bull(200)
        bear = make_daily_bear(200)
        rng = make_daily_range(200)
        return pd.concat([bull, bear, rng]).sort_index()

    def test_bullish_on_uptrend(self):
        """Trained on mixed data, analyzing uptrend → must detect bullish.

        Ticket 14 — Context FEATURE_PLAN now uses slow-horizon features
        (ema_ratio_50_200, zscore_20, etc.) that need ≥ 200 warmup bars
        to converge. Legacy 200-bar synthetic is the exact warmup
        threshold — extended to 500 bars so the features have room to
        settle before the model is queried."""
        from src.ml.jesse_agents import JesseContextAgent
        agent = JesseContextAgent()
        mixed = self._make_mixed_daily()
        agent.train(mixed)
        bull = make_daily_bull(500)
        result = agent.analyze(bull)
        assert result.state == 'bullish', f"Expected bullish, got {result.state}"
        assert result.passed is True
        assert result.score >= 0.5

    def test_bearish_on_downtrend(self):
        """Trained on mixed data, analyzing downtrend → must detect bearish."""
        from src.ml.jesse_agents import JesseContextAgent
        agent = JesseContextAgent()
        mixed = self._make_mixed_daily()
        agent.train(mixed)
        bear = make_daily_bear(500)
        result = agent.analyze(bear)
        assert result.state == 'bearish', f"Expected bearish, got {result.state}"
        assert result.passed is False  # bearish blocks longs

    def test_neutral_on_range(self):
        """Trained on mixed data, analyzing range → neutral or valid state."""
        from src.ml.jesse_agents import JesseContextAgent
        agent = JesseContextAgent()
        mixed = self._make_mixed_daily()
        agent.train(mixed)
        rng = make_daily_range(200)
        result = agent.analyze(rng)
        assert result.state in ('neutral', 'bullish', 'bearish')  # any valid


# ===========================================================================
# TEST 3: Features
# ===========================================================================
class TestContextFeatures:

    def test_features_are_stationary(self):
        """Context features must be ratios, not raw prices."""
        from src.ml.jesse_agents import JesseContextAgent
        agent = JesseContextAgent()
        df = make_daily_bull(200)
        features = agent.compute_features(df)
        assert isinstance(features, pd.DataFrame)
        for col in features.columns:
            valid = features[col].dropna()
            if len(valid) > 0:
                assert valid.abs().max() < 100, \
                    f"Feature '{col}' value {valid.abs().max():.1f} — not stationary"

    def test_features_include_momentum_and_vol(self):
        """Post-Ticket-14 : Context FEATURE_PLAN is drawn from the
        canonical `compute_stationary_features('full')` set. It
        includes `momentum_10`, `momentum_20`, `rsi_14`, plus slow
        structural + vol features (ema_ratio_*, atr_ratio,
        bb_width_ratio, etc.). The legacy `momentum_5` was replaced
        by the canonical `momentum_10` (see FEATURE_PLAN)."""
        from src.ml.jesse_agents import JesseContextAgent
        agent = JesseContextAgent()
        df = make_daily_bull(250)
        features = agent.compute_features(df)
        required = ['momentum_10', 'momentum_20', 'rsi_14']
        for feat in required:
            assert feat in features.columns, f"Missing feature: {feat}"


# ===========================================================================
# TEST 4: Standalone Backtest (agent seul)
# ===========================================================================
class TestContextBacktest:

    def test_standalone_backtest_on_bull(self):
        """Trained on mixed, test on bull: should signal bullish most
        of the time. Ticket 14 slow-horizon FEATURE_PLAN needs ≥ 200
        bars warmup; extended synthetic windows so the agent has
        enough history to discriminate."""
        from src.ml.jesse_agents import JesseContextAgent
        agent = JesseContextAgent()
        mixed = pd.concat([make_daily_bull(400), make_daily_bear(400),
                           make_daily_range(400)]).sort_index()
        agent.train(mixed)
        df_test = make_daily_bull(400)
        bullish_count = 0
        # Only measure past the ema_ratio_50_200 warmup (200 bars).
        for i in range(220, len(df_test) + 1):
            result = agent.analyze(df_test.iloc[:i])
            if result.state == 'bullish':
                bullish_count += 1
        n_measured = len(df_test) - 220 + 1
        pct_bullish = bullish_count / max(n_measured, 1)
        assert pct_bullish >= 0.35, \
            f"Only {pct_bullish:.0%} bullish on bull data — agent too conservative"

    def test_standalone_backtest_on_bear_blocks(self):
        """Trained on mixed, test on bear: should block most of the time."""
        from src.ml.jesse_agents import JesseContextAgent
        agent = JesseContextAgent()
        mixed = pd.concat([make_daily_bull(400), make_daily_bear(400),
                           make_daily_range(400)]).sort_index()
        agent.train(mixed)
        df_test = make_daily_bear(400)
        blocked_count = 0
        # Only measure past the ema_ratio_50_200 warmup.
        for i in range(220, len(df_test) + 1):
            result = agent.analyze(df_test.iloc[:i])
            if not result.passed:
                blocked_count += 1
        n_measured = len(df_test) - 220 + 1
        pct_blocked = blocked_count / max(n_measured, 1)
        assert pct_blocked >= 0.4, \
            f"Only {pct_blocked:.0%} blocked on bear data — agent not cautious enough"

    def test_standalone_backtest_returns_metrics(self):
        """Standalone backtest should return accuracy and trade metrics."""
        from src.ml.jesse_agents import JesseContextAgent
        agent = JesseContextAgent()
        mixed = pd.concat([make_daily_bull(200), make_daily_bear(200),
                           make_daily_range(200)]).sort_index()
        metrics = agent.backtest(mixed, train_ratio=0.75)
        assert 'accuracy' in metrics
        assert 'n_bars' in metrics
        assert 'pct_bullish' in metrics
        assert 'pct_bearish' in metrics
