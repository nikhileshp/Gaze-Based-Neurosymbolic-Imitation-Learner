"""Differentiable valuation functions for Space Invaders relations.

Same interface/conventions as in/envs/seaquest/valuation.py: each function takes
object logic-state rows of the form ``[present, x, y, w, h, ...]`` (coordinates in
the native 160x210 Atari frame) and returns a soft probability via
``bool_to_probs`` (True -> 0.99, False -> 0.01) or a smoothly-decaying score.

Object types: oplayer, oalien, oshield, obullet, osatellite.

The matching declarations live in
``logic/space_invaders_root/neural_preds.txt``. See
``data/gaze_formulation_viz/si_predicates.py`` for a boolean (numpy) twin used to
overlay the true predicates on the trajectory video.
"""

import torch as th

from nsfr.utils.common import bool_to_probs

# alignment / proximity thresholds in pixels (160x210 frame)
ALIGN_PX = 5.0       # column alignment tolerance (player ~7px wide, alien ~8px)
SLIGHT_PX = 12.0     # 0<|dx|<=this -> slightly left/right (co-occurs with aligned when <ALIGN)
CLOSE_BULLET_PX = 24.0
THREAT_ALIGN_PX = 8.0


def _cx(o: th.Tensor) -> th.Tensor:
    return o[..., 1] + o[..., 3] / 2.0


def _cy(o: th.Tensor) -> th.Tensor:
    return o[..., 2] + o[..., 4] / 2.0


# --- unary existence ---------------------------------------------------------
def visible_alien(obj: th.Tensor) -> th.Tensor:
    return bool_to_probs(obj[..., 0] == 1)


def visible_bullet(obj: th.Tensor) -> th.Tensor:
    return bool_to_probs(obj[..., 0] == 1)


def visible_satellite(obj: th.Tensor) -> th.Tensor:
    return bool_to_probs(obj[..., 0] == 1)


# --- player vs alien horizontal relations ------------------------------------
def left_of_alien(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Player is entirely left of the alien."""
    exists = obj[..., 0] == 1
    result = exists & (player[..., 1] + player[..., 3] < obj[..., 1])
    return bool_to_probs(result)


def right_of_alien(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Player is entirely right of the alien."""
    exists = obj[..., 0] == 1
    result = exists & (player[..., 1] > obj[..., 1] + obj[..., 3])
    return bool_to_probs(result)


def aligned_with_alien(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Player is column-aligned with the alien (can shoot it). Smoothly decays
    from 0.99 at perfect alignment to ~0 beyond ALIGN_PX. May co-occur with
    slightly_left/right_of_alien when the midpoints are close but not identical."""
    exists = obj[..., 0] == 1
    dx = th.abs(_cx(player) - _cx(obj))
    score = th.clip(1.0 - dx / ALIGN_PX, 0.0, 1.0)
    return score * bool_to_probs(exists)


def slightly_left_of_alien(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Player is slightly LEFT of the alien: 0 < (alien_cx - player_cx) <= SLIGHT_PX."""
    exists = obj[..., 0] == 1
    dx = _cx(obj) - _cx(player)  # >0 means alien is to the right of the player
    result = exists & (dx > 0) & (dx <= SLIGHT_PX)
    return bool_to_probs(result)


def slightly_right_of_alien(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Player is slightly RIGHT of the alien: 0 < (player_cx - alien_cx) <= SLIGHT_PX."""
    exists = obj[..., 0] == 1
    dx = _cx(player) - _cx(obj)
    result = exists & (dx > 0) & (dx <= SLIGHT_PX)
    return bool_to_probs(result)


def aligned_with_satellite(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Player is column-aligned with the bonus satellite/saucer."""
    exists = obj[..., 0] == 1
    dx = th.abs(_cx(player) - _cx(obj))
    score = th.clip(1.0 - dx / ALIGN_PX, 0.0, 1.0)
    return score * bool_to_probs(exists)


# --- threat relations --------------------------------------------------------
def close_by_bullet(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Bullet is spatially close to the player (Manhattan), smoothly decaying."""
    exists = obj[..., 0] == 1
    dist = th.abs(_cx(player) - _cx(obj)) + th.abs(_cy(player) - _cy(obj))
    score = th.clip(1.0 - dist / CLOSE_BULLET_PX, 0.0, 1.0)
    return score * bool_to_probs(exists)


def bullet_above_player(obj: th.Tensor, player: th.Tensor) -> th.Tensor:
    """Bullet is above the player and column-aligned with it (descending threat)."""
    exists = obj[..., 0] == 1
    above = _cy(obj) < _cy(player)
    aligned = th.abs(_cx(obj) - _cx(player)) < THREAT_ALIGN_PX
    return bool_to_probs(exists & above & aligned)


def bullet_left_of_player(obj: th.Tensor, player: th.Tensor) -> th.Tensor:
    """Bullet is entirely left of the player."""
    exists = obj[..., 0] == 1
    result = exists & (obj[..., 1] + obj[..., 3] < player[..., 1])
    return bool_to_probs(result)


def bullet_right_of_player(obj: th.Tensor, player: th.Tensor) -> th.Tensor:
    """Bullet is entirely right of the player."""
    exists = obj[..., 0] == 1
    result = exists & (obj[..., 1] > player[..., 1] + player[..., 3])
    return bool_to_probs(result)


def bullet_aligned_player(obj: th.Tensor, player: th.Tensor) -> th.Tensor:
    """Bullet is column-aligned with the player (any vertical position)."""
    exists = obj[..., 0] == 1
    aligned = th.abs(_cx(obj) - _cx(player)) < THREAT_ALIGN_PX
    return bool_to_probs(exists & aligned)


def bullet_threatens_shield(obj: th.Tensor, shield: th.Tensor) -> th.Tensor:
    """Bullet is above a shield and column-aligned with it."""
    exists = (obj[..., 0] == 1) & (shield[..., 0] == 1)
    above = _cy(obj) < _cy(shield)
    aligned = th.abs(_cx(obj) - _cx(shield)) < THREAT_ALIGN_PX
    return bool_to_probs(exists & above & aligned)


def bullet_above_alien(obj: th.Tensor, alien: th.Tensor) -> th.Tensor:
    """Bullet is above an alien and column-aligned (a player shot rising at it)."""
    exists = (obj[..., 0] == 1) & (alien[..., 0] == 1)
    above = _cy(obj) < _cy(alien)
    aligned = th.abs(_cx(obj) - _cx(alien)) < THREAT_ALIGN_PX
    return bool_to_probs(exists & above & aligned)


# --- defensive positioning ---------------------------------------------------
def behind_shield(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Player is protected by the shield: the player's MIDPOINT (center x) lies within
    the shield's x-range, so a vertical bullet in the player's column is blocked.
    Mere x-overlap is not enough — the player can still be hit if its center column is
    not under the shield."""
    exists = obj[..., 0] == 1
    pcx = _cx(player)
    s_left, s_right = obj[..., 1], obj[..., 1] + obj[..., 3]
    under = (pcx >= s_left) & (pcx <= s_right)
    return bool_to_probs(exists & under)
