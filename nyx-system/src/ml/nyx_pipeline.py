"""
Deprecation shim — the canonical engine is `NYXEngine` at
`src/core/nyx_engine.py` (Ticket 04).

This module previously held `class NYXPipeline`. The class was renamed
to `NYXEngine` and moved to `src/core/nyx_engine.py`. This file is now
a thin re-export so the ~45 existing callers that still do

    from src.ml.nyx_pipeline import NYXPipeline

keep working. Each caller will be migrated to the new path in follow-up
tickets. New code must import directly :

    from src.core.nyx_engine import NYXEngine

This shim must stay tiny — `test_canonical_entrypoint.py` enforces
size + no-class-redeclaration. Do not add logic here.
"""
from __future__ import annotations

from src.core.nyx_engine import NYXEngine as NYXPipeline


__all__ = ['NYXPipeline']
