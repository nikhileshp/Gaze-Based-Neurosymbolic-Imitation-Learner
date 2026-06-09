"""Backwards-compatible re-export shim.

The GABRIL gaze helpers (``GazeToMask``, ``load_dataset``, ``evaluate``,
``set_seed_everywhere``, ``plot_gaze_and_obs`` and the ``MAX_EPISODES`` tables)
were previously duplicated verbatim here and in the repo-root ``gabril_utils.py``.
They now live in the single source of truth :mod:`grail.gaze.gabril`.
"""

from grail.gaze.gabril import (
    GazeToMask,
    MAX_EPISODES,
    MAX_EPISODES_ATARI_HEAD,
    evaluate,
    load_dataset,
    plot_gaze_and_obs,
    set_seed_everywhere,
)

__all__ = [
    "GazeToMask",
    "MAX_EPISODES",
    "MAX_EPISODES_ATARI_HEAD",
    "evaluate",
    "load_dataset",
    "plot_gaze_and_obs",
    "set_seed_everywhere",
]
