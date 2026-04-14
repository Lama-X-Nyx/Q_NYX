"""
TDD Tests — 4-timeframe MTF feature coverage, for every asset.

**HARD RULE** (permanent): every training candidate MUST carry stationary
features from ALL 4 timeframes — regardless of which asset is trained.

  no prefix : 15m  (execution timeframe)
  h1_*      : 1h   (intra-day regime)
  h4_*      : 4h   (higher-order regime)    ← NEW
  d1_*      : 1d   (macro context)

Run this test on BTC, ETH, SOL in a single parameterization so the
guardrail catches any future asset that accidentally trains on a
reduced TF set.

Enforced invariants per asset:
  * ≥ 15 features per higher-TF block (h1_, h4_, d1_)
  * total feature count ≥ 65 per candidate
  * ffill alignment has no look-ahead (h4_<col> at time T = last 4h bar ≤ T)
"""
from pathlib import Path

import pytest
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
FEAT_DIR = Path(__file__).parent.parent / 'data' / 'features'
BTC_MTF_DIR = DATA_DIR / 'mtf'


_ASSETS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')


def _load_ohlcv(asset: str, tf: str) -> pd.DataFrame:
    """Asset-aware loader (SOL uses ms timestamp + different file names)."""
    if asset == 'BTCUSDT':
        p = BTC_MTF_DIR / f'BTCUSDT_{tf}.csv'
        df = pd.read_csv(p)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.set_index('datetime')
    if asset == 'ETHUSDT':
        p = DATA_DIR / f'ETHUSDT_{tf}.csv'
        df = pd.read_csv(p)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        for c in ('open', 'high', 'low', 'close'):
            df = df[df[c] > 0]
        return df
    # SOLUSDT
    mapping = {'15m': '15minutes', '1h': '1hour', '4h': '4hours', '1d': '1day'}
    p = DATA_DIR / f'SOLUSDT_{mapping[tf]}.csv'
    df = pd.read_csv(p, usecols=['timestamp', 'open', 'high', 'low',
                                   'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime').drop(columns='timestamp')
    for c in ('open', 'high', 'low', 'close'):
        df = df[df[c] > 0]
    return df


@pytest.fixture(scope='module', params=_ASSETS)
def asset_mtf(request):
    """Returns (asset, mtf_data, mtf_features) for each asset in _ASSETS."""
    asset = request.param
    mtf = {tf: _load_ohlcv(asset, tf) for tf in ('15m', '1h', '4h', '1d')}
    feats = {
        tf: pd.read_parquet(FEAT_DIR / f'{asset}_features_{tf}.parquet')
        for tf in ('15m', '1h', '4h', '1d')
    }
    return asset, mtf, feats


# ===========================================================================
class TestFullMTFFeatures_4TF:
    """Every asset must carry h1_*, h4_*, d1_* feature blocks."""

    def _cands(self, mtf, feats):
        from src.ml.nyx_pipeline import NYXPipeline
        p = NYXPipeline()
        ctx_1d = p._build_1d_context(mtf['1d'], feats['1d'])
        ctx_1h = p._build_1h_context(mtf['1h'], feats['1h'])
        df_15m = mtf['15m'].loc['2022-06-01':'2022-09-30']
        feat_15m = feats['15m'].loc['2022-06-01':'2022-09-30']
        return p._generate_candidates(
            df_15m, feat_15m, ctx_1d, ctx_1h,
            feat_1h=feats['1h'], feat_1d=feats['1d'],
            feat_4h=feats['4h'],
        )

    def test_h1_block_present(self, asset_mtf):
        asset, mtf, feats = asset_mtf
        cands = self._cands(mtf, feats)
        assert cands, f"{asset}: no candidates"
        h1 = [k for k in cands[0]['features'] if k.startswith('h1_')]
        assert len(h1) >= 15, f"{asset}: only {len(h1)} h1_* features"

    def test_h4_block_present(self, asset_mtf):
        """NEW invariant — 4h features must be injected for every asset."""
        asset, mtf, feats = asset_mtf
        cands = self._cands(mtf, feats)
        assert cands
        h4 = [k for k in cands[0]['features'] if k.startswith('h4_')]
        assert len(h4) >= 15, f"{asset}: only {len(h4)} h4_* features"

    def test_d1_block_present(self, asset_mtf):
        asset, mtf, feats = asset_mtf
        cands = self._cands(mtf, feats)
        assert cands
        d1 = [k for k in cands[0]['features'] if k.startswith('d1_')]
        assert len(d1) >= 15, f"{asset}: only {len(d1)} d1_* features"

    def test_total_count_at_least_65(self, asset_mtf):
        asset, mtf, feats = asset_mtf
        cands = self._cands(mtf, feats)
        assert cands
        n = len(cands[0]['features'])
        assert n >= 65, f"{asset}: only {n} total features"


class TestH4AlignmentSafety:
    """h4_<col> at 15m time T must equal the last 4h bar ≤ T (ffill)."""

    def test_h4_ffill_no_lookahead(self, asset_mtf):
        from src.ml.nyx_pipeline import NYXPipeline
        asset, mtf, feats = asset_mtf
        p = NYXPipeline()
        ctx_1d = p._build_1d_context(mtf['1d'], feats['1d'])
        ctx_1h = p._build_1h_context(mtf['1h'], feats['1h'])
        df_15m = mtf['15m'].loc['2022-06-01':'2022-06-10']
        feat_15m = feats['15m'].loc['2022-06-01':'2022-06-10']
        cands = p._generate_candidates(
            df_15m, feat_15m, ctx_1d, ctx_1h,
            feat_1h=feats['1h'], feat_1d=feats['1d'], feat_4h=feats['4h'],
        )
        if not cands:
            pytest.skip(f"no candidates for {asset}")

        cand = cands[0]
        ts = cand['timestamp']
        idx = feats['4h'].index.get_indexer([ts], method='ffill')[0]
        assert idx >= 0, f"{asset}: ffill found no prior 4h bar"
        expected = feats['4h'].iloc[idx]
        col = 'close_vs_ema50'
        assert f'h4_{col}' in cand['features'], \
            f"{asset}: missing h4_{col}"
        assert abs(cand['features'][f'h4_{col}'] - float(expected[col])) < 1e-9


class TestTrainingMetadataGuardrail:
    """Saved training metadata must reflect ≥ 65 features for every asset."""

    @pytest.mark.parametrize('asset', _ASSETS)
    def test_n_features_at_least_65(self, asset):
        import json
        meta_path = Path(__file__).parent.parent / 'models' / asset / 'training_metadata.json'
        if not meta_path.exists():
            pytest.skip(f"{asset}: no training metadata yet")
        meta = json.loads(meta_path.read_text())
        assert meta['n_features'] >= 65, \
            f"{asset} trained on {meta['n_features']} features — MTF rule violated"
