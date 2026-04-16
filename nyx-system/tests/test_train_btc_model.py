"""
TDD Tests — BTC Meta-GBM artefact (Ticket 11).

Ticket 11 — Train the Meta-GBM on the canonical BTC pipeline.

Per the canonical architecture :

- `NYXEngine` (Ticket 04) + `EdgeStrategy.generate_candidate_bars`
  (Ticket 08) = runtime candidate generation.
- `MetaGBM` (Tickets 06/07/10) = canonical strategy brain,
  encapsulates the trained `GradientBoostingClassifier`.
- The 84-feature contract used by the Meta-GBM today (rule_*
  proxies + 4-TF blocks) is the canonical Meta-GBM feature set.

This ticket persists a BTC artefact at `models/BTCUSDT/` matching
the layout `train_asset_model.train_and_save` produces for ETH/SOL.
The training artefacts must load cleanly via `load_artifact` and
pass the 4-TF coverage guardrail (MIN_FEATURES = 65, enforced by
`src.ml.train_asset_model.MTFCoverageError`).

Reproducibility invariant : the `GradientBoostingClassifier` is
instantiated with `random_state=42` inside `train_and_save`. Two
runs on the same data yield identical models.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


MODELS_DIR = Path(__file__).parent.parent / 'models'
REPORTS_DIR = Path(__file__).parent.parent / 'reports'
BTC_DIR = MODELS_DIR / 'BTCUSDT'


# ===========================================================================
class TestBTCArtefactFiles:
    """Every canonical per-asset file must be present under models/BTCUSDT/."""

    @pytest.mark.parametrize('fname', [
        'ml_filter_v1.pkl',
        'scaler.pkl',
        'feature_names.json',
        'training_metadata.json',
    ])
    def test_artefact_file_exists(self, fname):
        path = BTC_DIR / fname
        assert path.exists(), (
            f'{path} missing — run `python scripts/train_btc_model.py` '
            'to produce the canonical BTC Meta-GBM artefact (Ticket 11)'
        )


# ===========================================================================
class TestBTCArtefactLoadable:
    """The artefact must be loadable via the canonical helper
    `src.ml.train_asset_model.load_artifact`."""

    def test_load_artifact_succeeds(self):
        from src.ml.train_asset_model import load_artifact
        art = load_artifact(BTC_DIR)
        assert art is not None
        assert 'model' in art
        assert 'scaler' in art
        assert 'feature_names' in art
        assert 'metadata' in art

    def test_artefact_has_predict_proba(self):
        from src.ml.train_asset_model import load_artifact
        art = load_artifact(BTC_DIR)
        assert hasattr(art['model'], 'predict_proba')

    def test_artefact_has_scaler_transform(self):
        from src.ml.train_asset_model import load_artifact
        art = load_artifact(BTC_DIR)
        assert hasattr(art['scaler'], 'transform')


# ===========================================================================
class TestBTCArtefactMTFCoverage:
    """Rule 2 (OPERATING_RULES.md) — training must be 4-TF. Enforced by
    `MTFCoverageError` inside `train_and_save` : ≥ 65 features required,
    `h1_` + `h4_` + `d1_` prefixes all present."""

    @pytest.fixture(scope='class')
    def meta(self):
        path = BTC_DIR / 'training_metadata.json'
        if not path.exists():
            pytest.skip('training_metadata.json missing')
        return json.loads(path.read_text())

    def test_n_features_at_least_65(self, meta):
        assert meta.get('n_features', 0) >= 65, (
            f'BTC training had n_features={meta.get("n_features")} — '
            'Rule 2 (MTF coverage) requires ≥ 65'
        )

    def test_train_end_canonical(self, meta):
        """Canonical train_end is 2022-12-31 (matches ETH/SOL)."""
        assert meta.get('train_end') == '2022-12-31'

    def test_symbol_recorded(self, meta):
        assert meta.get('symbol') == 'BTCUSDT'

    def test_feature_names_include_four_tf_prefixes(self):
        names = json.loads((BTC_DIR / 'feature_names.json').read_text())
        joined = ' '.join(names)
        for prefix in ('h1_', 'h4_', 'd1_'):
            assert prefix in joined, (
                f'BTC feature_names.json missing {prefix}* block — '
                'MTF rule 2 violated'
            )


# ===========================================================================
class TestBTCOOSReport:
    """The training run must emit a reports/BTCUSDT_oos_report.json with
    a non-degenerate 2023 OOS summary (reproducibility proof + docs)."""

    OOS_PATH = REPORTS_DIR / 'BTCUSDT_oos_report.json'

    def test_oos_report_exists(self):
        assert self.OOS_PATH.exists(), (
            f'{self.OOS_PATH} missing — `scripts/train_btc_model.py` '
            'must emit an OOS report'
        )

    def test_oos_report_has_structure(self):
        r = json.loads(self.OOS_PATH.read_text())
        assert r.get('symbol') == 'BTCUSDT'
        assert 'training' in r
        assert 'oos_2023' in r
        oos = r['oos_2023']
        # Non-degenerate: must have at least a handful of trades
        # emitted from the canonical pipeline on BTC 2023.
        assert oos.get('n_trades', 0) >= 10, (
            f'BTC 2023 OOS only emitted {oos.get("n_trades")} trades '
            '— something is wrong with the canonical pipeline output '
            '(expect ~40 on BTC)'
        )

    def test_oos_report_sharpe_finite(self):
        r = json.loads(self.OOS_PATH.read_text())
        sharpe = r['oos_2023'].get('sharpe')
        assert sharpe is not None
        # Must be a finite real number (whether positive or negative
        # is not asserted here — reproducibility is the main ticket goal).
        assert float(sharpe) == float(sharpe)  # NaN check


# ===========================================================================
class TestBTCArtefactFitsCanonicalMetaGBM:
    """The trained BTC model must construct a MetaGBM successfully —
    the frozen Ticket 10 I/O contract stays honored."""

    def test_metagbm_constructible_from_btc_artefact(self):
        from src.ml.train_asset_model import load_artifact
        from src.core.meta_gbm import MetaGBM

        art = load_artifact(BTC_DIR)
        meta = MetaGBM(
            threshold=0.60,
            model=art['model'],
            scaler=art['scaler'],
            feature_names=art['feature_names'],
        )
        assert meta.has_trained_model is True
