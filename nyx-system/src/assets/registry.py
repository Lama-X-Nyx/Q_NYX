"""
AssetRegistry — per-asset configuration loaded from YAML.

This is the single source of truth for every fee, precision, slippage
model and cluster the hub-and-spoke architecture needs. The rest of
the stack (execution profiles, portfolio allocator, feedback loop)
reads from this registry so adding an asset is a yaml-only change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union


class AssetConfigError(ValueError):
    """Raised when the yaml config is malformed / incomplete."""


class AssetNotFound(KeyError):
    """Raised when an unknown symbol is requested."""


_REQUIRED = (
    'enabled', 'group',
    'price_precision', 'qty_precision',
    'maker_fee', 'taker_fee',
    'slippage_model', 'session_profile',
    'risk_cap', 'model_profile',
)


_SLIPPAGE_MODELS = {'low', 'medium', 'medium_high', 'high'}
_SESSION_PROFILES = {'liquid', 'event_sensitive', 'momentum_fast'}


@dataclass(frozen=True)
class Asset:
    symbol: str
    enabled: bool
    group: str
    price_precision: int
    qty_precision: int
    maker_fee: float
    taker_fee: float
    slippage_model: str        # one of _SLIPPAGE_MODELS
    session_profile: str       # one of _SESSION_PROFILES
    risk_cap: float            # fraction of total risk this asset may use
    model_profile: str         # e.g. 'eth_v1' — used to locate per-asset model


class AssetRegistry:
    """Dictionary-like registry indexed by symbol."""

    def __init__(self, assets: Dict[str, Asset]):
        self._assets = dict(assets)

    # ---------------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------------
    @classmethod
    def load(cls, path: Union[str, Path]) -> "AssetRegistry":
        try:
            import yaml
        except ImportError as e:
            raise AssetConfigError(
                "PyYAML is required to load asset registry"
            ) from e

        text = Path(path).read_text()
        data = yaml.safe_load(text) or {}
        if 'assets' not in data or not isinstance(data['assets'], dict):
            raise AssetConfigError("yaml must have top-level `assets:` mapping")

        assets: Dict[str, Asset] = {}
        for symbol, cfg in data['assets'].items():
            if not isinstance(cfg, dict):
                raise AssetConfigError(f"{symbol}: expected mapping, got {type(cfg)}")

            # Required fields
            missing = [k for k in _REQUIRED if k not in cfg]
            if missing:
                raise AssetConfigError(
                    f"{symbol}: missing required fields: {missing}"
                )

            # Enum validation
            if cfg['slippage_model'] not in _SLIPPAGE_MODELS:
                raise AssetConfigError(
                    f"{symbol}: slippage_model={cfg['slippage_model']!r} "
                    f"must be one of {sorted(_SLIPPAGE_MODELS)}"
                )

            try:
                asset = Asset(
                    symbol=symbol,
                    enabled=bool(cfg['enabled']),
                    group=str(cfg['group']),
                    price_precision=int(cfg['price_precision']),
                    qty_precision=int(cfg['qty_precision']),
                    maker_fee=float(cfg['maker_fee']),
                    taker_fee=float(cfg['taker_fee']),
                    slippage_model=str(cfg['slippage_model']),
                    session_profile=str(cfg['session_profile']),
                    risk_cap=float(cfg['risk_cap']),
                    model_profile=str(cfg['model_profile']),
                )
            except (TypeError, ValueError) as e:
                raise AssetConfigError(f"{symbol}: {e}") from e

            assets[symbol] = asset

        return cls(assets)

    # ---------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------
    def symbols(self) -> List[str]:
        return list(self._assets.keys())

    def get(self, symbol: str) -> Asset:
        if symbol not in self._assets:
            raise AssetNotFound(symbol)
        return self._assets[symbol]

    def list_enabled(self) -> List[Asset]:
        return [a for a in self._assets.values() if a.enabled]

    def cluster_of(self, symbol: str) -> str:
        return self.get(symbol).group

    def symbols_in_cluster(self, group: str) -> List[str]:
        return [a.symbol for a in self._assets.values() if a.group == group]
