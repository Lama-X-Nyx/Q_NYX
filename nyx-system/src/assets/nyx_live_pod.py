"""
NYXLivePod — SignalPod wrapper around `NYXLiveDecider` for HubSpokeRunner.

Task (b.B) layer 4.

This is the thinnest possible adapter: `HubSpokeRunner` calls
`pod.on_bar(bar)` per the SignalPod Protocol, and this pod delegates
to `decider.on_15m_bar(bar)`. The decider is multi-TF internally —
the runner doesn't need to know.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.assets.signal import Signal
from src.ml.nyx_live_decider import NYXLiveDecider


class NYXLivePod:
    """SignalPod adapter for NYXLiveDecider."""

    def __init__(
        self,
        symbol: str,
        artifact_dir: Path,
        cluster_group: str = 'majors',
        **decider_kwargs: Any,
    ):
        self.symbol = symbol
        self.cluster_group = cluster_group
        self.decider = NYXLiveDecider(
            symbol=symbol,
            artifact_dir=Path(artifact_dir),
            cluster_group=cluster_group,
            **decider_kwargs,
        )

    # ------------------------------------------------------------------
    def on_bar(self, bar: Dict[str, Any]) -> Signal:
        """Delegate 15m-bar decision to the live decider."""
        return self.decider.on_15m_bar(bar)
