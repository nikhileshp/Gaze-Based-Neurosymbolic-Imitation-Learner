"""GRAIL — Gaze-based neuRosymbolic imitAtion Learner.

Shared, importable library consolidating code that was previously copy-pasted
across the repository root, ``scripts/`` and ``baselines/``. Import via the
package path, e.g. ``from grail.gaze import GazeToMask`` (the repository is run
with its root on ``sys.path``, like the existing ``scripts.*`` / ``baselines.*``
packages).
"""

__all__ = ["gaze"]
