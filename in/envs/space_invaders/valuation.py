"""Differentiable valuation functions for Space Invaders relations.

Uniform-`object` scheme (consistent with freeway/seaquest/getout): every object
row is ``[present, x, y, w, h, type_id]`` in the native 160x210 Atari frame, with
``type_id`` matching the ``type:`` order in
``logic/space_invaders_root/consts.txt`` (player,alien,shield,bullet,satellite).

Each relational predicate is declared over ``object,object`` and gates internally
on the involved objects' types, so it only fires for the intended category pair
(e.g. ``aligned_with_alien`` is 0.01 unless arg0 is the player and arg1 an alien).
Booleans go through ``bool_to_probs`` (True -> 0.99, False -> 0.01); proximity
predicates return a smoothly-decaying score.

Geometry adapted from the original draft in this repo's ``main`` branch.
"""

import torch as th

from nsfr.utils.common import bool_to_probs

# type_id values — keep in sync with consts.txt `type:` line and env.py TYPE_MAP.
PLAYER, ALIEN, SHIELD, BULLET, SATELLITE = 0, 1, 2, 3, 4

# alignment / proximity thresholds in pixels (160x210 frame)
ALIGN_PX = 5.0          # column-alignment tolerance (player ~7px, alien ~8px wide)
SLIGHT_PX = 12.0        # 0<|dx|<=this -> slightly left/right
CLOSE_BULLET_PX = 24.0  # bullet proximity (Manhattan) decay scale
THREAT_ALIGN_PX = 8.0   # vertical-threat column-alignment tolerance


def _cx(o: th.Tensor) -> th.Tensor:
    return o[..., 1] + o[..., 3] / 2.0


def _cy(o: th.Tensor) -> th.Tensor:
    return o[..., 2] + o[..., 4] / 2.0


def _is(o: th.Tensor, type_id: int) -> th.Tensor:
    """Present AND of the given category."""
    return (o[..., 0] == 1) & (o[..., 5] == type_id)


# --- generic type / existence ------------------------------------------------
def type(obj: th.Tensor, type_oh: th.Tensor) -> th.Tensor:
    """obj exists and its type_id equals the one-hot type argument's index."""
    target = type_oh.argmax(dim=-1)
    return bool_to_probs((obj[..., 0] == 1) & (obj[..., 5].long() == target))


def in_image(zs: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Object is present in the frame (used by the placeholder action clauses)."""
    return bool_to_probs(obj[..., 0] == 1)


# --- unary existence ---------------------------------------------------------
def visible_alien(obj: th.Tensor) -> th.Tensor:
    return bool_to_probs(_is(obj, ALIEN))


def visible_bullet(obj: th.Tensor) -> th.Tensor:
    return bool_to_probs(_is(obj, BULLET))


def visible_satellite(obj: th.Tensor) -> th.Tensor:
    return bool_to_probs(_is(obj, SATELLITE))


# --- player vs alien horizontal relations ------------------------------------
def left_of_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    """Player is entirely left of the alien."""
    gate = _is(player, PLAYER) & _is(alien, ALIEN)
    return bool_to_probs(gate & (player[..., 1] + player[..., 3] < alien[..., 1]))


def right_of_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    """Player is entirely right of the alien."""
    gate = _is(player, PLAYER) & _is(alien, ALIEN)
    return bool_to_probs(gate & (player[..., 1] > alien[..., 1] + alien[..., 3]))


def aligned_with_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    """Player is column-aligned with the alien (can shoot it). Smoothly decays
    from 0.99 at perfect alignment to ~0 beyond ALIGN_PX."""
    gate = _is(player, PLAYER) & _is(alien, ALIEN)
    dx = th.abs(_cx(player) - _cx(alien))
    score = th.clip(1.0 - dx / ALIGN_PX, 0.0, 1.0)
    return score * bool_to_probs(gate)


def slightly_left_of_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    """Player slightly LEFT of the alien: 0 < (alien_cx - player_cx) <= SLIGHT_PX."""
    gate = _is(player, PLAYER) & _is(alien, ALIEN)
    dx = _cx(alien) - _cx(player)
    return bool_to_probs(gate & (dx > 0) & (dx <= SLIGHT_PX))


def slightly_right_of_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    """Player slightly RIGHT of the alien: 0 < (player_cx - alien_cx) <= SLIGHT_PX."""
    gate = _is(player, PLAYER) & _is(alien, ALIEN)
    dx = _cx(player) - _cx(alien)
    return bool_to_probs(gate & (dx > 0) & (dx <= SLIGHT_PX))


def aligned_with_satellite(player: th.Tensor, sat: th.Tensor) -> th.Tensor:
    """Player is column-aligned with the bonus satellite/saucer."""
    gate = _is(player, PLAYER) & _is(sat, SATELLITE)
    dx = th.abs(_cx(player) - _cx(sat))
    score = th.clip(1.0 - dx / ALIGN_PX, 0.0, 1.0)
    return score * bool_to_probs(gate)


# --- threat relations --------------------------------------------------------
def close_by_bullet(player: th.Tensor, bullet: th.Tensor) -> th.Tensor:
    """Bullet is spatially close to the player (Manhattan), smoothly decaying."""
    gate = _is(player, PLAYER) & _is(bullet, BULLET)
    dist = th.abs(_cx(player) - _cx(bullet)) + th.abs(_cy(player) - _cy(bullet))
    score = th.clip(1.0 - dist / CLOSE_BULLET_PX, 0.0, 1.0)
    return score * bool_to_probs(gate)


def bullet_above_player(bullet: th.Tensor, player: th.Tensor) -> th.Tensor:
    """Bullet is above the player and column-aligned (descending threat)."""
    gate = _is(bullet, BULLET) & _is(player, PLAYER)
    above = _cy(bullet) < _cy(player)
    aligned = th.abs(_cx(bullet) - _cx(player)) < THREAT_ALIGN_PX
    return bool_to_probs(gate & above & aligned)


def bullet_left_of_player(bullet: th.Tensor, player: th.Tensor) -> th.Tensor:
    """Bullet is entirely left of the player."""
    gate = _is(bullet, BULLET) & _is(player, PLAYER)
    return bool_to_probs(gate & (bullet[..., 1] + bullet[..., 3] < player[..., 1]))


def bullet_right_of_player(bullet: th.Tensor, player: th.Tensor) -> th.Tensor:
    """Bullet is entirely right of the player."""
    gate = _is(bullet, BULLET) & _is(player, PLAYER)
    return bool_to_probs(gate & (bullet[..., 1] > player[..., 1] + player[..., 3]))


def bullet_aligned_player(bullet: th.Tensor, player: th.Tensor) -> th.Tensor:
    """Bullet is column-aligned with the player (any vertical position)."""
    gate = _is(bullet, BULLET) & _is(player, PLAYER)
    aligned = th.abs(_cx(bullet) - _cx(player)) < THREAT_ALIGN_PX
    return bool_to_probs(gate & aligned)


def bullet_threatens_shield(bullet: th.Tensor, shield: th.Tensor) -> th.Tensor:
    """Bullet is above a shield and column-aligned with it."""
    gate = _is(bullet, BULLET) & _is(shield, SHIELD)
    above = _cy(bullet) < _cy(shield)
    aligned = th.abs(_cx(bullet) - _cx(shield)) < THREAT_ALIGN_PX
    return bool_to_probs(gate & above & aligned)


def bullet_above_alien(bullet: th.Tensor, alien: th.Tensor) -> th.Tensor:
    """Bullet is above an alien and column-aligned (a player shot rising at it)."""
    gate = _is(bullet, BULLET) & _is(alien, ALIEN)
    above = _cy(bullet) < _cy(alien)
    aligned = th.abs(_cx(bullet) - _cx(alien)) < THREAT_ALIGN_PX
    return bool_to_probs(gate & above & aligned)


# --- defensive positioning ---------------------------------------------------
def behind_shield(player: th.Tensor, shield: th.Tensor) -> th.Tensor:
    """Player is protected: the player's center-x lies within the shield's x-range,
    so a vertical bullet in the player's column is blocked."""
    gate = _is(player, PLAYER) & _is(shield, SHIELD)
    pcx = _cx(player)
    under = (pcx >= shield[..., 1]) & (pcx <= shield[..., 1] + shield[..., 3])
    return bool_to_probs(gate & under)
