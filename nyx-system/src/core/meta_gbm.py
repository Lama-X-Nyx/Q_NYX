"""
MetaGBM — canonical strategy brain (Ticket 06).

Replaces the vote-based "ALL must pass → trade / any block → WAIT"
semantic of `JesseOrchestrator` (mono-file) and `Orchestrator`
(per-file) with a principled aggregation :

- **Disagreement is a feature**, not an automatic failure. A
  MetaDecision can still pass when 1 or 2 reports have
  `passed=False`, provided the aggregate score + direction clear
  the threshold.
- **Strategy-aware outputs** — the MetaDecision now carries
  `quality_bucket` and `risk_hint` (Ticket 06 additions to the
  contract) so the risk / execution layers can size and route
  orders with more context than a simple BUY/SELL/WAIT.

Chain (Tickets 03, 05, 06) :

    4 Jesse agents  →  4 FractalReport  →  MetaGBM.decide()
                                        →  MetaDecision
                                        →  TradePlan  →  ExecutionInstruction

**Not wired into NYXEngine yet** — per Ticket 06 out-of-scope
("Full risk manager integration"). NYXEngine continues to use
its internal GBM + proxy `rule_*` scalars. `MetaGBM` is the
**canonical interface** that future tickets can wire into the
runtime path in place of the vote-based orchestrators.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.agents.contracts import (
    CANONICAL_TIMEFRAMES,
    FractalReport,
    MetaDecision,
)


DEFAULT_THRESHOLD = 0.60
DEFAULT_DISAGREEMENT_WEIGHT = 0.3
DEFAULT_QUALITY_HIGH_CUTOFF = 0.75
DEFAULT_QUALITY_MEDIUM_CUTOFF = 0.60


class MetaGBM:
    """Canonical strategy brain for NYX.

    Interprets 4 fractal reports (context / regime / setup / entry)
    plus market features to emit a canonical `MetaDecision`.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        disagreement_weight: float = DEFAULT_DISAGREEMENT_WEIGHT,
        quality_high_cutoff: float = DEFAULT_QUALITY_HIGH_CUTOFF,
        quality_medium_cutoff: float = DEFAULT_QUALITY_MEDIUM_CUTOFF,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold={threshold} out of [0,1]")
        if not 0.0 <= disagreement_weight <= 1.0:
            raise ValueError(
                f"disagreement_weight={disagreement_weight} out of [0,1]"
            )
        self.threshold = float(threshold)
        self.disagreement_weight = float(disagreement_weight)
        self.quality_high_cutoff = float(quality_high_cutoff)
        self.quality_medium_cutoff = float(quality_medium_cutoff)

    # ------------------------------------------------------------------
    def decide(
        self,
        fractal_reports: Mapping[str, FractalReport],
        features: Dict[str, float],
        asset: str,
        timestamp: str,
        timeframe: str = '15m',
        hint_direction: int = 0,
    ) -> MetaDecision:
        """Emit a canonical `MetaDecision` from 4 fractal reports.

        Args :
          fractal_reports : dict keyed by agent name
              ({'context', 'regime', 'setup', 'entry'}).
          features        : market-state features (momentum, atr, vol,
              etc.). Added to the decision's `features_snapshot` for
              downstream traceability.
          asset / timestamp / timeframe : MetaDecision identity.
          hint_direction  : external direction hint (-1 / 0 / +1). The
              MetaGBM does not derive direction from reports today —
              it trusts the caller (hard gate / edge detector) to
              propose a direction, then validates via score + threshold.

        Returns a `MetaDecision` with :
          passed           : True iff direction != 0 AND probability >=
              threshold. NOT gated on "all reports passed=True".
          probability      : aggregate score × (1 − disagreement_weight
              × disagreement).
          candidate_quality: aggregate score (pre-penalty).
          quality_bucket   : high / medium / low from aggregate score.
          risk_hint        : 1 − disagreement, ∈ [0, 1].
          features_snapshot: caller features + injected
              'disagreement' + 'n_passed_agents' + 'aggregate_score'.
        """
        if timeframe not in CANONICAL_TIMEFRAMES:
            raise ValueError(
                f"timeframe={timeframe!r} must be in {CANONICAL_TIMEFRAMES}"
            )
        if int(hint_direction) not in (-1, 0, 1):
            raise ValueError(
                f"hint_direction={hint_direction} must be -1, 0, or +1"
            )

        reports = dict(fractal_reports)
        n = max(len(reports), 1)

        # 1. Aggregate score — mean of the report scores.
        scores = [float(r.score) for r in reports.values()]
        aggregate = sum(scores) / n if scores else 0.0

        # 2. Disagreement = fraction of reports that did NOT pass.
        n_passed = sum(1 for r in reports.values() if bool(r.passed))
        disagreement = 1.0 - (n_passed / n if n else 0.0)

        # 3. Probability = aggregate penalised by disagreement
        #    (but still not hard-blocked).
        probability = aggregate * (1.0 - self.disagreement_weight * disagreement)
        probability = max(0.0, min(1.0, probability))

        # 4. Gate — direction AND probability ≥ threshold.
        direction = int(hint_direction)
        block_reasons = []
        if direction == 0:
            block_reasons.append('no directional hint (hint_direction=0)')
        if probability < self.threshold:
            block_reasons.append(
                f'probability {probability:.3f} < threshold '
                f'{self.threshold:.3f}'
            )
        passed = not block_reasons

        # 5. Quality bucket from aggregate score.
        if aggregate >= self.quality_high_cutoff:
            quality_bucket = 'high'
        elif aggregate >= self.quality_medium_cutoff:
            quality_bucket = 'medium'
        else:
            quality_bucket = 'low'

        # 6. Risk hint — confidence in the decision, inversely related
        #    to disagreement.
        risk_hint = max(0.0, min(1.0, 1.0 - disagreement))

        # 7. Build the MetaDecision — features_snapshot carries caller
        #    features + our injected ones for downstream traceability.
        snapshot: Dict[str, float] = {}
        for k, v in features.items():
            try:
                snapshot[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        snapshot['disagreement'] = float(disagreement)
        snapshot['n_passed_agents'] = float(n_passed)
        snapshot['aggregate_score'] = float(aggregate)

        return MetaDecision(
            asset=asset,
            timestamp=timestamp,
            timeframe=timeframe,
            direction=direction if passed else 0,
            probability=float(probability),
            threshold_used=self.threshold,
            passed=passed,
            block_reasons=block_reasons,
            # Placeholder edge proxy — real edge comes from the GBM
            # trained on net outcomes. Kept simple here since feature
            # redesign is out of scope.
            expected_edge_net=float(aggregate * 100.0),
            candidate_quality=float(aggregate),
            fractal_reports=reports,
            features_snapshot=snapshot,
            quality_bucket=quality_bucket,
            risk_hint=float(risk_hint),
        )
