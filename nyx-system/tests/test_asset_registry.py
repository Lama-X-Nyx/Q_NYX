"""
TDD Tests — AssetRegistry

Loads config/assets.yaml and exposes per-asset configuration.

Contract:
  - `load(path)` reads yaml; each asset entry must have required fields
    (price_precision, qty_precision, maker_fee, taker_fee, slippage_model,
    session_profile, risk_cap, model_profile, group, enabled).
  - `get(symbol)` returns Asset dataclass, raises AssetNotFound if missing.
  - `list_enabled()` returns only enabled assets.
  - `cluster_of(symbol)` returns the cluster tag (majors/alts).
  - Validation: missing required field → AssetConfigError.
  - Validation: slippage_model must be one of {low, medium, medium_high, high}.
"""
from pathlib import Path

import pytest


YAML_VALID = """
assets:
  ETHUSDT:
    enabled: true
    group: majors
    price_precision: 2
    qty_precision: 3
    maker_fee: 0.0002
    taker_fee: 0.0004
    slippage_model: medium
    session_profile: liquid
    risk_cap: 0.40
    model_profile: eth_v1

  XRPUSDT:
    enabled: true
    group: alts
    price_precision: 4
    qty_precision: 1
    maker_fee: 0.0002
    taker_fee: 0.0004
    slippage_model: medium_high
    session_profile: event_sensitive
    risk_cap: 0.25
    model_profile: xrp_v1

  SOLUSDT:
    enabled: true
    group: alts
    price_precision: 2
    qty_precision: 2
    maker_fee: 0.0002
    taker_fee: 0.0004
    slippage_model: high
    session_profile: momentum_fast
    risk_cap: 0.25
    model_profile: sol_v1
"""


YAML_DISABLED_ONE = """
assets:
  ETHUSDT: {enabled: true, group: majors, price_precision: 2, qty_precision: 3,
            maker_fee: 0.0002, taker_fee: 0.0004, slippage_model: medium,
            session_profile: liquid, risk_cap: 0.4, model_profile: eth_v1}
  XRPUSDT: {enabled: false, group: alts, price_precision: 4, qty_precision: 1,
            maker_fee: 0.0002, taker_fee: 0.0004, slippage_model: medium_high,
            session_profile: event_sensitive, risk_cap: 0.25, model_profile: xrp_v1}
"""


YAML_MISSING_FIELD = """
assets:
  ETHUSDT:
    enabled: true
    group: majors
    # price_precision missing → should fail
    qty_precision: 3
    maker_fee: 0.0002
    taker_fee: 0.0004
    slippage_model: medium
    session_profile: liquid
    risk_cap: 0.4
    model_profile: eth_v1
"""


YAML_BAD_SLIPPAGE = """
assets:
  ETHUSDT:
    enabled: true
    group: majors
    price_precision: 2
    qty_precision: 3
    maker_fee: 0.0002
    taker_fee: 0.0004
    slippage_model: insane   # not allowed
    session_profile: liquid
    risk_cap: 0.4
    model_profile: eth_v1
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "assets.yaml"
    p.write_text(text)
    return p


# ===========================================================================
class TestLoadValid:

    def test_loads_three_assets(self, tmp_path):
        from src.assets.registry import AssetRegistry
        reg = AssetRegistry.load(_write(tmp_path, YAML_VALID))
        assert set(reg.symbols()) == {'ETHUSDT', 'XRPUSDT', 'SOLUSDT'}

    def test_get_returns_asset_dataclass(self, tmp_path):
        from src.assets.registry import AssetRegistry
        reg = AssetRegistry.load(_write(tmp_path, YAML_VALID))
        eth = reg.get('ETHUSDT')
        assert eth.symbol == 'ETHUSDT'
        assert eth.group == 'majors'
        assert eth.price_precision == 2
        assert eth.qty_precision == 3
        assert eth.maker_fee == 0.0002
        assert eth.taker_fee == 0.0004
        assert eth.slippage_model == 'medium'
        assert eth.session_profile == 'liquid'
        assert eth.risk_cap == 0.40
        assert eth.model_profile == 'eth_v1'
        assert eth.enabled is True


# ===========================================================================
class TestEnabledFilter:

    def test_list_enabled(self, tmp_path):
        from src.assets.registry import AssetRegistry
        reg = AssetRegistry.load(_write(tmp_path, YAML_DISABLED_ONE))
        enabled = reg.list_enabled()
        assert {a.symbol for a in enabled} == {'ETHUSDT'}

    def test_disabled_still_accessible(self, tmp_path):
        from src.assets.registry import AssetRegistry
        reg = AssetRegistry.load(_write(tmp_path, YAML_DISABLED_ONE))
        xrp = reg.get('XRPUSDT')
        assert xrp.enabled is False


# ===========================================================================
class TestCluster:

    def test_cluster_of_major(self, tmp_path):
        from src.assets.registry import AssetRegistry
        reg = AssetRegistry.load(_write(tmp_path, YAML_VALID))
        assert reg.cluster_of('ETHUSDT') == 'majors'

    def test_cluster_of_alt(self, tmp_path):
        from src.assets.registry import AssetRegistry
        reg = AssetRegistry.load(_write(tmp_path, YAML_VALID))
        assert reg.cluster_of('XRPUSDT') == 'alts'
        assert reg.cluster_of('SOLUSDT') == 'alts'

    def test_symbols_in_cluster(self, tmp_path):
        from src.assets.registry import AssetRegistry
        reg = AssetRegistry.load(_write(tmp_path, YAML_VALID))
        alts = reg.symbols_in_cluster('alts')
        assert set(alts) == {'XRPUSDT', 'SOLUSDT'}


# ===========================================================================
class TestValidation:

    def test_missing_field_raises(self, tmp_path):
        from src.assets.registry import AssetRegistry, AssetConfigError
        with pytest.raises(AssetConfigError):
            AssetRegistry.load(_write(tmp_path, YAML_MISSING_FIELD))

    def test_bad_slippage_model_raises(self, tmp_path):
        from src.assets.registry import AssetRegistry, AssetConfigError
        with pytest.raises(AssetConfigError):
            AssetRegistry.load(_write(tmp_path, YAML_BAD_SLIPPAGE))

    def test_unknown_symbol_raises(self, tmp_path):
        from src.assets.registry import AssetRegistry, AssetNotFound
        reg = AssetRegistry.load(_write(tmp_path, YAML_VALID))
        with pytest.raises(AssetNotFound):
            reg.get('DOGEUSDT')
