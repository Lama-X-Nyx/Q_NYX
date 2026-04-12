"""
PrecomputedRunner — O(T·N²) total instead of O(T·window·N²)

Bottleneck analysis (70ms/bar):
  SMC detect_all         : 21ms  (30%) — precompute rolling window
  HSMM initialize_params : 22ms  (31%) — compute once, then streaming forward
  _prepare_data ×2       : 10ms  (14%) — vectorized on full dataset
  Other                  :  7ms  ( 5%)

Strategy
--------
Precompute phase (once, ~10s for full year):
  1. Vectorized feature arrays  (1H / 15M) — pandas rolling on full dataset
  2. Streaming HSMM forward     (1H / 15M) — O(T·N²) total, causal
  3. SMA200 context             (1D)       — pandas rolling
  4. SMC patterns               (15M)      — rolling 200-bar windows

Backtest loop per bar (~1ms total):
  context_state     = context_arr[i_1d]
  regime_probs      = gamma_1h[i_1h]       (numpy row)
  setup_probs       = gamma_15m[i_15m]     (numpy row)
  smc_patterns      = smc_cache[i_15m]     (dict)
  decision          = _score(...)           (pure arithmetic)
"""

import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.agents.contracts import AgentResult, OrchestratorDecision
from src.core.hsmm import SemiMarkovHMM, _logsumexp


# ---------------------------------------------------------------------------
# Streaming forward pass (causal, no backward pass = no look-ahead)
# ---------------------------------------------------------------------------

def forward_streaming(hsmm: SemiMarkovHMM,
                      log_B: np.ndarray) -> np.ndarray:
    """
    Run the HMM forward algorithm on a pre-built emission matrix.

    For the LAST bar of any window, smoothed probs == forward probs
    (because β_T = 1 for all states).  This is mathematically equivalent
    to the current window-based forward_backward for our use case
    (we only read hsmm state at the MOST RECENT bar of each window).

    Parameters
    ----------
    hsmm    : SemiMarkovHMM with locked transition matrix
    log_B   : (T, n_states) pre-computed emission log-likelihoods

    Returns
    -------
    gamma   : (T, n_states) normalised forward probabilities (causal)
    """
    T, n = log_B.shape
    tm = hsmm.transition_matrix if hsmm.transition_matrix is not None else np.ones((n, n)) / n
    log_A = np.log(tm + 1e-10)          # (n, n)

    log_alpha = np.empty((T, n), dtype=np.float64)
    ip = hsmm.initial_probs if hsmm.initial_probs is not None else np.ones(n) / n
    log_alpha[0] = np.log(ip + 1e-10) + log_B[0]

    for t in range(1, T):
        # Vectorised: (n,1) + (n,n) → broadcast (n,n), logsumexp over axis=0 → (n,)
        log_alpha[t] = (
            _logsumexp(log_alpha[t - 1, :, np.newaxis] + log_A, axis=0)
            + log_B[t]
        )

    log_Z = _logsumexp(log_alpha, axis=1, keepdims=True)     # (T, 1)
    gamma = np.exp(log_alpha - log_Z)                        # (T, n)
    return gamma


# ---------------------------------------------------------------------------
# Feature preparation (vectorized, mirrors _prepare_data in agents)
# ---------------------------------------------------------------------------

def _prepare_features_full(df: pd.DataFrame,
                             df_htf: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Vectorized version of RegimeAgent._prepare_data / SetupAgent._prepare_data
    applied to the FULL dataset.  No per-bar Python loops.
    """
    out = df[['open', 'high', 'low', 'close', 'volume']].copy()

    out['returns']     = out['close'].pct_change()
    high = out['high'].values
    low  = out['low'].values
    cl   = out['close'].values
    prev = np.roll(cl, 1); prev[0] = cl[0]
    tr   = np.maximum(high - low,
                      np.maximum(np.abs(high - prev), np.abs(low - prev)))
    out['atr_14'] = pd.Series(tr, index=out.index).rolling(14).mean()
    out['sma_20'] = out['close'].rolling(20).mean()
    out['sma_50'] = out['close'].rolling(50).mean()
    out['atr_50'] = out['atr_14'].rolling(50).mean()
    if 'volume' in out.columns:
        out['volume_ma20'] = out['volume'].rolling(20).mean()

    # HTF context feature (merge_asof, no look-ahead)
    if df_htf is not None and not df_htf.empty:
        htf_sma20 = df_htf['close'].rolling(20).mean().rename('_htf_sma20')
        htf_ref = htf_sma20.reset_index()
        htf_ref.columns = ['_ts', '_htf_sma20']
        htf_ref = htf_ref.dropna(subset=['_htf_sma20']).sort_values('_ts')

        cur_ref = out[['close']].copy().reset_index()
        cur_ref.columns = ['_ts', '_close']
        cur_ref = cur_ref.sort_values('_ts')

        merged = pd.merge_asof(cur_ref, htf_ref, on='_ts', direction='backward')
        merged.index = out.index
        with np.errstate(invalid='ignore', divide='ignore'):
            htf_pos = (merged['_close'] - merged['_htf_sma20']) / merged['_htf_sma20'].replace(0, np.nan)
        out['htf_pos'] = htf_pos.values

    return out


def _build_obs_arrays(df_prepared: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extract observation arrays from a prepared DataFrame.
    Returns dict with 'price', 'atr', 'context' (if present).
    These are passed to HSMM._compute_log_B via a thin wrapper.
    """
    price = df_prepared['returns'].values.astype(float)
    atr   = df_prepared['atr_14'].values.astype(float)
    obs   = {'price': price, 'atr': atr}
    if 'htf_pos' in df_prepared.columns:
        obs['context'] = df_prepared['htf_pos'].values.astype(float)
    return obs


def _compute_log_B_from_arrays(hsmm: SemiMarkovHMM,
                                 obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Vectorized log-emission matrix from numpy arrays.
    Equivalent to hsmm._compute_log_B(list_of_dicts) but ~10× faster
    because it avoids building the intermediate list.
    """
    from scipy import stats
    T    = len(obs['price'])
    n    = hsmm.n_states
    ep: Dict   = hsmm.emission_params or {}

    pm   = np.array([ep[s]['price_mu']    for s in hsmm.states])
    ps   = np.array([ep[s]['price_sigma'] for s in hsmm.states])
    am   = np.array([ep[s]['atr_mu']      for s in hsmm.states])
    as_  = np.array([ep[s]['atr_sigma']   for s in hsmm.states])

    prices = obs['price']
    atrs   = obs['atr']

    log_B = np.zeros((T, n))

    valid_p = ~np.isnan(prices)
    if valid_p.any():
        log_B[valid_p] += stats.norm.logpdf(
            prices[valid_p, np.newaxis], loc=pm, scale=ps)

    valid_a = ~np.isnan(atrs)
    if valid_a.any():
        log_B[valid_a] += stats.norm.logpdf(
            atrs[valid_a, np.newaxis], loc=am, scale=as_)

    if 'context' in obs:
        ctx = obs['context']
        has_ctx = all('context_mu' in ep.get(s, {}) for s in hsmm.states)
        if has_ctx:
            cm = np.array([ep.get(s, {}).get('context_mu', 0.0)   for s in hsmm.states])
            cs = np.array([ep.get(s, {}).get('context_sigma', 0.02) for s in hsmm.states])
            valid_c = ~np.isnan(ctx)
            if valid_c.any():
                log_B[valid_c] += stats.norm.logpdf(
                    ctx[valid_c, np.newaxis], loc=cm, scale=cs)

    return log_B


# ---------------------------------------------------------------------------
# SMC rolling precomputation
# ---------------------------------------------------------------------------

def _precompute_smc(df_15m_prepared: pd.DataFrame,
                     smc_detector,
                     window: int = 200,
                     start_iloc: int = 0,
                     end_iloc: Optional[int] = None) -> tuple:
    """
    Run SMC detect_all on a rolling `window`-bar slice only for bars in
    [start_iloc, end_iloc).  Bars before start_iloc are used as context
    but not stored.

    Returns
    -------
    (smc_list, smc_start_iloc)
      smc_list       : list of dicts, length = end_iloc - start_iloc
      smc_start_iloc : absolute iloc of smc_list[0]
    """
    n = len(df_15m_prepared)
    if end_iloc is None:
        end_iloc = n

    start_iloc = max(0, start_iloc)
    end_iloc   = min(end_iloc, n)

    empty = {
        'bullish_ob': False, 'bearish_ob': False,
        'bullish_fvg': False, 'bearish_fvg': False,
        'bullish_choch': False, 'bearish_choch': False,
        'bullish_at_zone': False, 'bearish_at_zone': False,
        'smc_score_bullish': 0.0, 'smc_score_bearish': 0.0,
    }

    out = []
    for i in range(start_iloc, end_iloc):
        ctx_start = max(0, i + 1 - window)
        slice_df  = df_15m_prepared.iloc[ctx_start: i + 1]
        if len(slice_df) < 50:
            out.append(empty.copy())
            continue
        try:
            patterns = smc_detector.detect_all(slice_df)
            out.append(patterns)
        except Exception:
            out.append(empty.copy())

    return out, start_iloc


# ---------------------------------------------------------------------------
# Context precomputation (SMA200 on 1D)
# ---------------------------------------------------------------------------

def _precompute_context(df_1d: pd.DataFrame,
                         trend_threshold: float = 0.02) -> np.ndarray:
    """
    Returns string array of length len(df_1d) with values
    'bullish' | 'bearish' | 'neutral' | 'insufficient'.
    """
    close   = df_1d['close'].values.astype(float)
    sma200  = pd.Series(close).rolling(200).mean().values
    diff    = (close - sma200) / sma200

    ctx = np.full(len(df_1d), 'insufficient', dtype=object)
    valid = ~np.isnan(sma200)
    ctx[valid & (diff > trend_threshold)]  = 'bullish'
    ctx[valid & (diff < -trend_threshold)] = 'bearish'
    ctx[valid & (np.abs(diff) <= trend_threshold)] = 'neutral'
    return ctx


# ---------------------------------------------------------------------------
# Main precomputed states container
# ---------------------------------------------------------------------------

class PrecomputedStates:
    """
    Holds all precomputed arrays for the backtest hot loop.

    Usage
    -----
    states = PrecomputedStates()
    states.precompute(mtf_all, orchestrator)

    # In the backtest loop (aligned_idx already built):
    decision = states.decide_fast(aligned_idx[bar_i], current_price)
    """

    CACHE_VERSION = 'v1_precomp'

    def __init__(self, orchestrator, cache_dir: str = 'data/pretrain_cache'):
        self._orch      = orchestrator
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Precomputed arrays (populated by .precompute())
        self.context_arr: Optional[np.ndarray] = None   # (T_1d,) str
        self.gamma_1h:    Optional[np.ndarray] = None   # (T_1h, n_regime)
        self.gamma_15m:   Optional[np.ndarray] = None   # (T_15m, n_setup)
        self.smc_list:    Optional[List[Dict]] = None   # (T_smc,) dicts

        # SMC offset: smc_list[i] corresponds to absolute 15m iloc (i + _smc_start_iloc)
        self._smc_start_iloc: int = 0

        # Index → iloc lookup (used by decide_fast)
        self._idx_1d:  Optional[np.ndarray] = None
        self._idx_1h:  Optional[np.ndarray] = None
        self._idx_15m: Optional[np.ndarray] = None

        self._ready = False

    # -----------------------------------------------------------------------
    # Precompute phase
    # -----------------------------------------------------------------------

    def precompute(self, mtf_all: Dict[str, pd.DataFrame],
                   cache_key: str = '',
                   start_ts: Optional[str] = None,
                   end_ts: Optional[str] = None) -> None:
        """
        Precompute all agent states for the entire dataset.

        Parameters
        ----------
        mtf_all  : full MTF data dict (1d / 4h / 1h / 15m)
        cache_key: if non-empty, try to load/save from disk
        start_ts : ISO date — only compute SMC from this date forward
                   (HSMM forward pass still runs on all history)
        end_ts   : ISO date — only compute SMC up to this date
        """
        if cache_key:
            path = self._cache_dir / f'precomp_{cache_key}_{self.CACHE_VERSION}.pkl'
            if self._load_cache(path):
                return

        t0 = time.time()
        print('  [PRECOMPUTE] Building precomputed states…')

        # ---- Context (1D SMA200) ----
        df_1d = mtf_all['1d']
        trend_thr = self._orch.context_agent.trend_threshold
        ctx_arr = _precompute_context(df_1d, trend_thr)
        self.context_arr = ctx_arr
        self._idx_1d = df_1d.index.values.astype(np.int64)
        print(f'    context    : {len(ctx_arr)} bars  '
              f'({(ctx_arr=="bullish").sum()} bullish / '
              f'{(ctx_arr=="bearish").sum()} bearish)')

        # ---- Regime HSMM (1H + HTF=4H) ----
        df_1h  = mtf_all['1h']
        df_4h  = mtf_all.get('4h')
        df_1h_prep = _prepare_features_full(df_1h, df_htf=df_4h)

        regime_hsmm = self._orch.regime_agent.hsmm
        regime_hsmm.initialize_parameters(df_1h_prep.dropna().tail(500))

        obs_1h   = _build_obs_arrays(df_1h_prep)
        log_B_1h = _compute_log_B_from_arrays(regime_hsmm, obs_1h)
        self.gamma_1h  = forward_streaming(regime_hsmm, log_B_1h)
        self._idx_1h   = df_1h.index.values.astype(np.int64)
        print(f'    regime 1H  : {len(self.gamma_1h)} bars  '
              f'({regime_hsmm.n_states} states)')

        # ---- Setup HSMM (15M + HTF=1H) ----
        df_15m = mtf_all['15m']
        df_15m_prep = _prepare_features_full(df_15m, df_htf=df_1h)

        setup_hsmm = self._orch.setup_agent.hsmm
        setup_hsmm.initialize_parameters(df_15m_prep.dropna().tail(500))

        obs_15m   = _build_obs_arrays(df_15m_prep)
        log_B_15m = _compute_log_B_from_arrays(setup_hsmm, obs_15m)
        self.gamma_15m  = forward_streaming(setup_hsmm, log_B_15m)
        self._idx_15m   = df_15m.index.values.astype(np.int64)
        print(f'    setup 15M  : {len(self.gamma_15m)} bars  '
              f'({setup_hsmm.n_states} states)')

        # ---- SMC patterns (15M rolling 200-bar, restricted range) ----
        # SMC only needed for bars in the actual backtest window.
        # Determine iloc range for [start_ts, end_ts) to avoid computing
        # 151K bars when we only need ~2880.
        df_15m = mtf_all['15m']
        if start_ts is not None:
            smc_start = max(0, int(np.searchsorted(
                df_15m.index.astype(np.int64),
                pd.Timestamp(start_ts).value, side='left'
            )))
        else:
            smc_start = 0
        if end_ts is not None:
            smc_end = int(np.searchsorted(
                df_15m.index.astype(np.int64),
                pd.Timestamp(end_ts).value, side='right'
            ))
        else:
            smc_end = len(df_15m_prep)

        n_smc = smc_end - smc_start
        print(f'    SMC 15M    : {n_smc} bars (iloc [{smc_start}:{smc_end}])… ',
              end='', flush=True)
        t_smc = time.time()
        self.smc_list, self._smc_start_iloc = _precompute_smc(
            df_15m_prep,
            self._orch.setup_agent.smc,
            window=200,
            start_iloc=smc_start,
            end_iloc=smc_end,
        )
        print(f'{time.time()-t_smc:.1f}s')

        self._ready = True
        elapsed = time.time() - t0
        print(f'  [PRECOMPUTE] Done in {elapsed:.1f}s')

        if cache_key:
            self._save_cache(path)

    # -----------------------------------------------------------------------
    # Fast per-bar decision
    # -----------------------------------------------------------------------

    def decide_fast(self, pos: Dict[str, tuple],
                    current_price: float) -> OrchestratorDecision:
        """
        Produce an OrchestratorDecision from precomputed arrays.
        Equivalent to orchestrator.decide() but O(1) in data access.

        Parameters
        ----------
        pos           : aligned_idx entry — {tf: (start_iloc, end_iloc)}
        current_price : float
        """
        if not self._ready:
            raise RuntimeError('Call precompute() first.')

        cfg = self._orch.config

        # ---- Context ----
        i_1d = pos['1d'][1] - 1        # last closed 1D bar
        if i_1d < 0 or i_1d >= len(self.context_arr):
            return self._wait('Context: index out of range')
        ctx = str(self.context_arr[i_1d])
        if ctx == 'insufficient':
            return self._wait('Context: SMA200 NaN (warmup)', ready=False)

        context_result = self._score_context(ctx)

        # ---- Regime (1H) ----
        i_1h = pos['1h'][1] - 1
        readiness_cfg = cfg.get('fractal_readiness', {})
        if i_1h < readiness_cfg.get('regime_min_bars', 100):
            return self._wait('Regime: warmup', ready=False)

        regime_probs = self.gamma_1h[i_1h]
        regime_result = self._score_regime(regime_probs, ctx)

        # ---- Setup (15M) ----
        i_15m = pos['15m'][1] - 1
        if i_15m < readiness_cfg.get('setup_min_bars', 50):
            return self._wait('Setup: warmup', ready=False)

        setup_probs   = self.gamma_15m[i_15m]
        smc_idx = i_15m - self._smc_start_iloc
        if smc_idx < 0 or smc_idx >= len(self.smc_list):
            return self._wait('SMC: index out of precomputed range')
        smc_patterns  = self.smc_list[smc_idx]
        setup_result  = self._score_setup(setup_probs, smc_patterns, ctx)

        # ---- Pipeline gate ----
        components = {
            'context': context_result,
            'regime':  regime_result,
            'setup':   setup_result,
        }

        all_ready = all(r.ready  for r in components.values())
        if not all_ready:
            not_ready = [k for k, r in components.items() if not r.ready]
            return OrchestratorDecision(
                action='WAIT', score=0.0,
                reason=f'Not ready: {not_ready}',
                blocked_by=['readiness'], components=components
            )

        all_passed = all(r.passed for r in components.values())
        blocked_by = [k for k, r in components.items() if not r.passed]
        if not all_passed:
            return OrchestratorDecision(
                action='WAIT',
                score=self._agg_score(components),
                reason=f'Blocked: {blocked_by}',
                blocked_by=blocked_by, components=components
            )

        # ---- Risk check (delegated to real risk manager) ----
        risk = self._orch.risk_manager.check_risk_conditions(
            entry_price=current_price,
            fractal_states={'4h': regime_result.metadata.get('hsmm_states', {})},
            smc_patterns=smc_patterns,
            intent_daily=ctx,
            transition_matrix=self._orch.regime_agent.hsmm.transition_matrix,
            emission_params=self._orch.regime_agent.hsmm.emission_params,
            hsmm_states_list=self._orch.regime_agent.hsmm.states,
        )
        if not (risk['rr_ratio'] and risk['risk_hit']):
            return OrchestratorDecision(
                action='WAIT',
                score=self._agg_score(components),
                reason='Risk blocked',
                blocked_by=['risk'], components=components,
                risk_analysis=risk
            )

        # ---- Direction ----
        action = {'bullish': 'BUY', 'bearish': 'SELL'}.get(ctx, 'WAIT')
        return OrchestratorDecision(
            action=action,
            score=self._agg_score(components),
            reason=f'Precomputed: ctx={ctx} regime={regime_result.state} '
                   f'setup={setup_result.state}',
            blocked_by=[],
            components=components,
            risk_analysis=risk
        )

    # -----------------------------------------------------------------------
    # Scoring helpers (pure arithmetic on precomputed arrays)
    # -----------------------------------------------------------------------

    def _score_context(self, ctx: str) -> AgentResult:
        state = ctx  # 'bullish' | 'bearish' | 'neutral'
        score = {'bullish': 0.75, 'bearish': 0.75, 'neutral': 0.50}.get(ctx, 0.0)
        return AgentResult(
            agent='context', state=state, score=score,
            passed=(ctx != 'neutral'), ready=True,
            reason=f'Precomputed SMA200 → {ctx}'
        )

    def _score_regime(self, probs: np.ndarray, ctx: str) -> AgentResult:
        hsmm    = self._orch.regime_agent.hsmm
        cfg_mtf = self._orch.config.get('strategy', {}).get('mtf_conditions', {})
        sdc_min     = cfg_mtf.get('sdc_min', 4.5)
        stab_min    = cfg_mtf.get('stability_4h_min', 0.55)

        dom_idx  = int(np.argmax(probs))
        dom_name = hsmm.states[dom_idx]
        dom_prob = float(probs[dom_idx])
        sdc      = 10.0 * dom_prob
        stability = float(hsmm.transition_matrix[dom_idx, dom_idx])

        state_map = {
            'Trend+': 'trend_plus', 'Range': 'range', 'Trend-': 'trend_minus',
            'Squeeze': 'squeeze', 'Distribution': 'distribution',
            'Liquidation': 'liquidation',
        }
        state = state_map.get(dom_name, 'range')

        if state == 'liquidation':
            return AgentResult(agent='regime', state='liquidation',
                               score=0.0, passed=False, ready=True,
                               reason='Liquidation detected')

        stab_eff = 0.35 if state == 'squeeze' else stab_min
        ctx_ok   = True
        if ctx == 'bullish' and state in ('trend_minus', 'distribution'):
            ctx_ok = False
        elif ctx == 'bearish' and state in ('trend_plus', 'squeeze'):
            ctx_ok = False
        elif ctx == 'neutral' and state not in ('range', 'squeeze'):
            ctx_ok = False

        passed = (sdc > sdc_min) and (stability >= stab_eff) and ctx_ok
        hsmm_states = {s: float(probs[i]) for i, s in enumerate(hsmm.states)}

        return AgentResult(
            agent='regime', state=state,
            score=min(sdc / 10.0, 1.0),
            passed=passed, ready=True,
            reason=f'Precomputed HSMM {dom_name} SdC={sdc:.1f} stab={stability:.2f}',
            metadata={
                'hsmm_states': hsmm_states,
                'dominant_state': dom_name,
                'sdc': sdc, 'stability': stability,
                'transition_matrix': hsmm.transition_matrix.tolist(),
            }
        )

    def _score_setup(self, probs: np.ndarray,
                      smc_patterns: Dict,
                      ctx: str) -> AgentResult:
        hsmm    = self._orch.setup_agent.hsmm
        cfg_mtf = self._orch.config.get('strategy', {}).get('mtf_conditions', {})
        align_min = cfg_mtf.get('alignment_15m_min', 0.40)

        s2i = hsmm.state_to_idx
        p_tp   = float(probs[s2i['Trend+']])
        p_rng  = float(probs[s2i.get('Range', 1)])
        p_tm   = float(probs[s2i['Trend-']])
        p_sq   = float(probs[s2i['Squeeze']])  if 'Squeeze'      in s2i else 0.0
        p_dist = float(probs[s2i['Distribution']]) if 'Distribution' in s2i else 0.0
        p_liq  = float(probs[s2i['Liquidation']])  if 'Liquidation'  in s2i else 0.0

        if p_liq > 0.5:
            return AgentResult(agent='setup', state='liquidation',
                               score=0.0, passed=False, ready=True,
                               reason='Liquidation on 15M')

        bull_score = p_tp + 0.5 * p_sq + 0.25 * p_rng
        bear_score = p_tm + p_dist

        if ctx == 'bullish':
            alignment = bull_score
        elif ctx == 'bearish':
            alignment = bear_score
        else:
            alignment = max(bull_score, bear_score)

        # SMC bonus
        smc_q = float(smc_patterns.get(
            'smc_score_bullish' if ctx == 'bullish' else 'smc_score_bearish', 0.0))
        has_pat = (smc_patterns.get('bullish_ob') or smc_patterns.get('bullish_fvg')
                   or smc_patterns.get('bullish_choch')) if ctx == 'bullish' else (
                  smc_patterns.get('bearish_ob') or smc_patterns.get('bearish_fvg')
                  or smc_patterns.get('bearish_choch'))
        score = min(alignment + 0.15 * smc_q, 1.0)
        if not has_pat:
            score *= 0.85

        passed = alignment >= align_min
        state  = 'valid_setup' if (passed and has_pat) else \
                 'misaligned' if not passed else 'no_pattern'

        return AgentResult(
            agent='setup', state=state,
            score=score, passed=passed, ready=True,
            reason=f'Precomputed align={alignment:.3f} (thr={align_min}) '
                   f'SMC={smc_q:.2f}',
            metadata={
                'alignment': alignment,
                'patterns': smc_patterns,
                'hsmm_p_trend_plus': p_tp, 'hsmm_p_trend_minus': p_tm,
                'hsmm_p_range': p_rng, 'hsmm_p_squeeze': p_sq,
                'hsmm_p_distribution': p_dist,
            }
        )

    @staticmethod
    def _agg_score(components: Dict) -> float:
        w = {'context': 0.35, 'regime': 0.35, 'setup': 0.30}
        return sum(components[k].score * w.get(k, 0) for k in components)

    @staticmethod
    def _wait(reason: str, ready: bool = True) -> OrchestratorDecision:
        return OrchestratorDecision(
            action='WAIT', score=0.0, reason=reason,
            blocked_by=['readiness' if not ready else 'logic']
        )

    # -----------------------------------------------------------------------
    # Cache
    # -----------------------------------------------------------------------

    def _save_cache(self, path: Path) -> None:
        payload = {
            'context_arr':      self.context_arr,
            'gamma_1h':         self.gamma_1h,
            'gamma_15m':        self.gamma_15m,
            'smc_list':         self.smc_list,
            'smc_start_iloc':   self._smc_start_iloc,
            'idx_1d':           self._idx_1d,
            'idx_1h':           self._idx_1h,
            'idx_15m':          self._idx_15m,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=4)
        size_mb = path.stat().st_size / 1e6
        print(f'  [PRECOMPUTE SAVED]  {path.name}  ({size_mb:.1f} MB)')

    def _load_cache(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, 'rb') as f:
                p = pickle.load(f)
            self.context_arr       = p['context_arr']
            self.gamma_1h          = p['gamma_1h']
            self.gamma_15m         = p['gamma_15m']
            self.smc_list          = p['smc_list']
            self._smc_start_iloc   = p.get('smc_start_iloc', 0)
            self._idx_1d           = p['idx_1d']
            self._idx_1h           = p['idx_1h']
            self._idx_15m          = p['idx_15m']
            self._ready      = True
            size_mb = path.stat().st_size / 1e6
            print(f'  [PRECOMPUTE HIT]  {path.name}  ({size_mb:.1f} MB)')
            return True
        except Exception as e:
            print(f'  [PRECOMPUTE MISS]  {e}')
            return False
