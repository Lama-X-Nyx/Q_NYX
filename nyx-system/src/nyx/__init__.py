"""
NYX — Canonical production system namespace.

This package re-exports all canonical classes from their source
locations. New code should import from here:

    from src.nyx import NYXRuntime, CentralOrchestrator
    from src.nyx.decision import MetaGBM, NYXLiveDecider
    from src.nyx.execution import OMS, RiskEngine

Existing imports (from src.live.*, src.core.*, etc.) continue
to work unchanged.
"""
from src.nyx.runtime import NYXRuntime, CentralOrchestrator

__all__ = ['NYXRuntime', 'CentralOrchestrator']
