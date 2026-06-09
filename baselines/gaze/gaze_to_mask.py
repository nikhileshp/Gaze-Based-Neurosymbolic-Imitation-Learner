"""Backwards-compatible re-export.

``GazeToMask`` now lives in :mod:`grail.gaze.gaze_to_mask` (single source of
truth). This module is kept so existing imports such as
``from baselines.gaze.gaze_to_mask import GazeToMask`` keep working.
"""

from grail.gaze.gaze_to_mask import GazeToMask

__all__ = ["GazeToMask"]
