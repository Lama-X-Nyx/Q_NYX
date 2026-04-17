"""
Fractal quality — post-decision modulation (Ticket 20).

The GBM (173 features, Sharpe 9.96) DECIDES whether to trade.
The 4 Jesse ML-native agents CONTEXTUALIZE the trade quality via
a multi-timeframe fractal assessment. The result modulates position
SIZING — not the GBM's threshold, not the direction.

Chain :
  GBM says "trade" (p_trade ≥ threshold)
    → 4 agents assess quality :
        Context (1D)  → macro direction quality
        Regime  (4H)  → structural state quality
        Setup   (1H)  → opportunity quality
        Entry   (15M) → timing quality
    → fractal_quality ∈ [0, 1] = fraction of agents that passed
    → size_multiplier maps quality to position sizing
    → skip_trade if fractal_quality < 0.25 (all agents reject)

This is NOT a signal — it is a CONFIDENCE MODULATOR.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from src.agents.contracts import FractalReport


# Canonical agent keys and their fractal contribution weight.
# Equal weight (0.25 each) so fractal_quality ∈ [0, 1].
_AGENT_KEYS = ('context', 'regime', 'setup', 'entry')
_WEIGHT_PER_AGENT = 1.0 / len(_AGENT_KEYS)

# Size multiplier schedule (fractal_quality → position size factor).
_SIZE_SCHEDULE = [
    (0.75, 1.25),   # ≥ 0.75 (3-4 pass) → conviction: +25 % size
    (0.50, 1.00),   # ≥ 0.50 (2 pass)   → normal: baseline size
    (0.25, 0.50),   # ≥ 0.25 (1 pass)   → cautious: half size
    (0.00, 0.00),   # < 0.25 (0 pass)   → skip trade entirely
]


def compute_fractal_quality(
    reports: Mapping[str, FractalReport],
) -> Dict[str, Any]:
    """Multi-timeframe quality assessment from 4 Jesse fractal reports.

    Returns a dict with :
      fractal_quality  : float [0, 1] — fraction of agents that passed
      size_multiplier  : float [0, 1.25] — position sizing factor
      quality_bucket   : 'high' / 'medium' / 'low'
      skip_trade       : bool — True when ALL agents reject (quality 0)
    """
    n_passed = 0
    for key in _AGENT_KEYS:
        r = reports.get(key)
        if r is not None and bool(r.passed):
            n_passed += 1
    fractal_quality = float(n_passed) * _WEIGHT_PER_AGENT

    # Map quality to size multiplier via schedule.
    size_multiplier = 0.0
    for threshold, mult in _SIZE_SCHEDULE:
        if fractal_quality >= threshold:
            size_multiplier = mult
            break

    # Quality bucket.
    if fractal_quality >= 0.75:
        quality_bucket = 'high'
    elif fractal_quality >= 0.50:
        quality_bucket = 'medium'
    else:
        quality_bucket = 'low'

    return {
        'fractal_quality': fractal_quality,
        'size_multiplier': size_multiplier,
        'quality_bucket':  quality_bucket,
        'skip_trade':      size_multiplier <= 0.0,
    }
