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

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.agents.contracts import (
    CANONICAL_TIMEFRAMES,
    FractalReport,
    MetaDecision,
)


DEFAULT_THRESHOLD = 0.60
DEFAULT_DISAGREEMENT_WEIGHT = 0.3
DEFAULT_QUALITY_HIGH_CUTOFF = 0.75
DEFAULT_QUALITY_MEDIUM_CUTOFF = 0.60

# Probability-source tags exposed in MetaDecision.features_snapshot
# for traceability. Numeric (not string) so features_snapshot stays
# Dict[str, float].
PROBABILITY_SOURCE_HEURISTIC = 0.0
PROBABILITY_SOURCE_PRECOMPUTED = 1.0
PROBABILITY_SOURCE_TRAINED_GBM = 2.0


class MetaGBM:
    """Canonical strategy brain for NYX.

    Interprets 4 fractal reports (context / regime / setup / entry)
    plus market features to emit a canonical `MetaDecision`.

    **Ticket 10 — frozen I/O contract.** `INPUT_SCHEMA` and
    `OUTPUT_SCHEMA` declare every key the `.decide()` method accepts
    and emits. Risk / execution layers consume the outputs by name
    (`MetaDecision.trade_decision`, `.confidence`, `.expected_edge`,
    `.trade_quality_bucket`, `.risk_hint`, `.direction`) without
    ad-hoc translation.
    """

    # ------------------------------------------------------------------
    # Ticket 10 — FROZEN I/O CONTRACT
    # ------------------------------------------------------------------
    #
    # INPUT schema — every key `.decide()` accepts. Mutating this dict
    # in code is forbidden (it is enforced as the canonical contract
    # between candidate generation, fractal reporters, and the meta
    # brain).
    INPUT_SCHEMA: Dict[str, str] = {
        # 4 Jesse fractal reports keyed by agent name
        # ({'context','regime','setup','entry'}). Empty dict accepted
        # while live agents are not yet wired into the runtime.
        'fractal_reports':   'Mapping[str, FractalReport]',
        # candidate-generation edge features + market state features
        # merged. Floats only; non-numeric values are silently dropped.
        'features':          'Dict[str, float]',
        # MetaDecision identity fields
        'asset':             'str',
        'timestamp':         'str  (ISO8601)',
        'timeframe':         'str  in CANONICAL_TIMEFRAMES',
        # External directional hint from candidate generation hard gate
        'hint_direction':    'int  in {-1, 0, +1}',
        # Optional batch helper — caller pre-computed proba via the
        # encapsulated GBM (Ticket 07 NYXEngine path).
        'precomputed_proba': 'Optional[float]  in [0, 1]',
        # Optional live helper — raw feature vector to be auto-scored
        # via the encapsulated trained GBM (Ticket 07 NYXLiveDecider
        # path).
        'feature_vector':    'Optional[Sequence[float] | np.ndarray]',
        # Whether `feature_vector` is already scaled (skip
        # scaler.transform). Default False.
        'already_scaled':    'bool',
    }

    # OUTPUT schema — every canonical field exposed on the returned
    # `MetaDecision`. Aliases (Ticket 10) are properties that reuse
    # internal field names so older callers keep working unchanged.
    OUTPUT_SCHEMA: Dict[str, str] = {
        # Meta-strategy semantic — derived from `passed` + `direction`.
        # 'BUY' | 'SELL' | 'WAIT'. Risk + execution layers route on
        # this string.
        'trade_decision':       "str  in {'BUY', 'SELL', 'WAIT'}",
        # Position direction.
        'direction':            'int  in {-1, 0, +1}',
        # Strategy-brain confidence in the decision.
        # Alias for `MetaDecision.probability` ∈ [0, 1].
        'confidence':           'float  in [0, 1]',
        # Expected edge net of fees (bps proxy). Alias for
        # `MetaDecision.expected_edge_net`.
        'expected_edge':        'float  (bps proxy, may be negative)',
        # Quality bucket from aggregate fractal-report score. Alias
        # for `MetaDecision.quality_bucket`.
        'trade_quality_bucket': "Optional[str]  in {'high', 'medium', 'low'}",
        # Risk hint ∈ [0, 1] — high = confident, low = disagreement.
        'risk_hint':            'Optional[float]  in [0, 1]',
    }

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        disagreement_weight: float = DEFAULT_DISAGREEMENT_WEIGHT,
        quality_high_cutoff: float = DEFAULT_QUALITY_HIGH_CUTOFF,
        quality_medium_cutoff: float = DEFAULT_QUALITY_MEDIUM_CUTOFF,
        model: Optional[Any] = None,
        scaler: Optional[Any] = None,
        feature_names: Optional[Sequence[str]] = None,
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

        # Trained-model encapsulation (Ticket 07, Option C wrapper).
        # When model + scaler are provided, MetaGBM owns the decision
        # and encapsulates the trained GBM as implementation detail.
        self._model = model
        self._scaler = scaler
        self._feature_names = (
            list(feature_names) if feature_names is not None else None
        )

    # ------------------------------------------------------------------
    @property
    def has_trained_model(self) -> bool:
        """True when MetaGBM encapsulates a trained GBM + scaler."""
        return self._model is not None and self._scaler is not None

    def score_vector(
        self,
        feature_row: Any,
        already_scaled: bool = False,
    ) -> float:
        """Run the encapsulated trained GBM on a single row.

        Returns the positive-class probability ∈ [0, 1].

        Raises `RuntimeError` if no trained model is wired (use the
        heuristic path via `.decide()` in that case).
        """
        if not self.has_trained_model:
            raise RuntimeError(
                'MetaGBM has no trained model — pass `model=` and '
                '`scaler=` to the constructor, or use `.decide()` '
                'without a feature_vector to fall back to the '
                'heuristic aggregate path.'
            )
        # has_trained_model==True ⇒ model + scaler are not None.
        model = self._model
        scaler = self._scaler
        assert model is not None and scaler is not None  # for pyright
        row = np.asarray(feature_row, dtype=float)
        if row.ndim == 1:
            row = row.reshape(1, -1)
        if not already_scaled:
            row = scaler.transform(row)
            row = np.nan_to_num(row, nan=0.0, posinf=3.0, neginf=-3.0)
        proba = model.predict_proba(row)[0]
        classes = list(model.classes_)
        p1_idx = classes.index(1) if 1 in classes else 0
        return float(proba[p1_idx])

    # ------------------------------------------------------------------
    def decide(
        self,
        fractal_reports: Mapping[str, FractalReport],
        features: Dict[str, float],
        asset: str,
        timestamp: str,
        timeframe: str = '15m',
        hint_direction: int = 0,
        precomputed_proba: Optional[float] = None,
        feature_vector: Any = None,
        already_scaled: bool = False,
    ) -> MetaDecision:
        """Emit a canonical `MetaDecision` — strategy brain entry point.

        Probability source precedence (Ticket 07) :
          1. `precomputed_proba` (caller has batch-scored upstream)
          2. `feature_vector` + trained GBM (auto-score)
          3. heuristic aggregate of FractalReport scores (Ticket 06 fallback)

        Fractal reports drive `disagreement`, `quality_bucket`,
        `risk_hint`, and `candidate_quality` — regardless of which
        probability source is used.

        Args :
          fractal_reports : dict keyed by agent name
              ({'context', 'regime', 'setup', 'entry'}); may be empty
              when Jesse agents are not yet wired (the GBM path does
              not need them).
          features        : market-state features (momentum, atr, vol,
              etc.); copied into the decision's `features_snapshot`.
          asset / timestamp / timeframe : MetaDecision identity.
          hint_direction  : external direction hint (-1 / 0 / +1). The
              MetaGBM does not derive direction from reports today —
              it trusts the caller (hard gate / edge detector).
          precomputed_proba : if supplied, used as probability.
          feature_vector  : used to auto-score via trained GBM.
          already_scaled  : if True, skip `scaler.transform()`.
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

        # 3. Resolve probability from one of 3 sources (Ticket 07).
        if precomputed_proba is not None:
            probability = float(precomputed_proba)
            probability_source = PROBABILITY_SOURCE_PRECOMPUTED
        elif feature_vector is not None and self.has_trained_model:
            probability = self.score_vector(feature_vector, already_scaled)
            probability_source = PROBABILITY_SOURCE_TRAINED_GBM
        else:
            # Heuristic path — aggregate penalised by disagreement.
            probability = aggregate * (
                1.0 - self.disagreement_weight * disagreement
            )
            probability_source = PROBABILITY_SOURCE_HEURISTIC
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
        snapshot['probability_source'] = float(probability_source)

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
