import torch
from nsfr.utils.common import bool_to_probs
 
 
def _get_x_y(z: torch.Tensor):
    # Freeway layout in OCAtari: [vis, chicken, car, x, y]
    # Indices 3 and 4 are consistently x and y.
    # If gaze is appended (6 features), it is at index 5.
    return z[..., 3], z[..., 4]
 
 
def type(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    z_type = z[:, 1:3]  
    prob = (a * z_type).sum(dim=1)
    return prob
 
 
def visible(z: torch.Tensor) -> torch.Tensor:
    # Robust visibility: true if any type flag is set
    # This ignores index 0 which might be unreliable in some datasets
    is_present = (z[..., 1:3].sum(dim=-1) > 0.5)
    return bool_to_probs(is_present)
 
 
def _is_player(z: torch.Tensor) -> torch.Tensor:
    # Chicken is identified by the second feature (index 1) in the logic state
    return z[..., 1] > 0.5
 
 
def closeby(z_1: torch.Tensor, z_2: torch.Tensor) -> torch.Tensor:
    x1, y1 = _get_x_y(z_1)
    x2, y2 = _get_x_y(z_2)
 
    dis_x = abs(x1 - x2) / 171
    dis_y = abs(y1 - y2) / 171
 
    # Continuous Gaussian decay
    prob = torch.exp(- (dis_x**2 / (2 * 0.2**2)) - (dis_y**2 / (2 * 0.05**2)))
 
    # Only return probability if z_2 is the player
    return prob * _is_player(z_2).float()
 
 
# Horizontal collision threshold (pixels). Calibrated from the dataset: when the
# human chose `noop`, the nearest car one lane up was within ~12px horizontally in
# 90% of frames; when going `up` only ~15% of frames had a car that close.
X_CLOSE_PX = 12


def x_close(z_1: torch.Tensor, z_2: torch.Tensor):
    """Horizontal proximity of a car (z_1) to the player's column (z_2).

    Unlike `closeby` (an anisotropic Gaussian whose tight y-bandwidth, sigma_y~8.55px,
    conflicts with the >=9px inter-lane spacing that `above_row` requires and caps the
    noop rules at ~0.5), this deliberately ignores the vertical gap. The lane (vertical)
    relation is handled separately by `above_row`/`same_row`; this predicate only asks
    "is the car in my column?". Near-binary (0.99/0.01) so a true trigger reaches ~1.
    Gated to the player like `closeby`.
    """
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    result = (torch.abs(x1 - x2) < X_CLOSE_PX) & _is_player(z_2)
    return bool_to_probs(result)


# Freeway lane y -> car speed (px/frame), measured from OCAtari dx. Used to make the
# "danger" lookahead distance proportional to how fast the car is closing in — i.e. a
# fast car triggers a wait from further away. This recovers time-to-arrival from the
# lane (y) alone, no velocity feature in the state.
_LANE_Y = [27, 43, 59, 75, 91, 107, 123, 139, 155, 171]
_LANE_SPEED = [0.8, 1.0, 1.33, 2.0, 4.0, 4.0, 2.0, 1.0, 1.0, 0.8]
# How many frames ahead to anticipate an approaching car (lookahead = X_CLOSE_PX + frames*speed).
# Env-overridable for tuning (APPROACH_FRAMES=4 ...). 0 => no approach extension (x_close only).
import os as _os
APPROACH_FRAMES = float(_os.environ.get("APPROACH_FRAMES", 2))


def _lane_speed(y: torch.Tensor) -> torch.Tensor:
    centers = torch.tensor(_LANE_Y, dtype=y.dtype, device=y.device)
    speeds = torch.tensor(_LANE_SPEED, dtype=y.dtype, device=y.device)
    idx = (y.unsqueeze(-1) - centers).abs().argmin(dim=-1)
    return speeds[idx]


# Width (px) of the "in front" zone: a continuous Gaussian over horizontal distance, so
# the car is still "in front" (noop keeps firing) until it has clearly passed the column,
# instead of a hard cutoff that flips to 0 the instant the center crosses 12px. Tunable.
CAR_INFRONT_SIGMA = float(_os.environ.get("CAR_INFRONT_SIGMA", 14.0))


def car_in_front(z_1: torch.Tensor, z_2: torch.Tensor):
    """Continuous probability that a car (z_1) is in front of the player's (z_2) column.
    ~1 when horizontally aligned, decaying smoothly with |Δx| (sigma=CAR_INFRONT_SIGMA px),
    so noop stays high while the car is still passing and only fades once it is clear."""
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    dx = x1 - x2
    prob = torch.exp(-(dx ** 2) / (2 * CAR_INFRONT_SIGMA ** 2))
    return prob * _is_player(z_2).float()


def path_clear_infront(z: torch.Tensor, all_objects: torch.Tensor = None):
    """Continuous probability that NO car is in front in the lane above the player (z).
    Noisy-AND over cars: prod_i (1 - car_in_front_i). Scene-level gate for `up` so a far
    car cannot green-light going up while another car is still passing the column."""
    px, py = _get_x_y(z)
    if all_objects is None:
        return torch.zeros_like(px)
    is_car = all_objects[..., 2] > 0.5
    vis = all_objects[..., 0] > 0.5
    cx = all_objects[..., 3]; cy = all_objects[..., 4]
    dy = py.unsqueeze(-1) - cy
    above = (dy >= 9) & (dy < 23)
    dx = cx - px.unsqueeze(-1)
    infront = torch.exp(-(dx ** 2) / (2 * CAR_INFRONT_SIGMA ** 2))
    infront = infront * (above & is_car & vis).float()    # only cars in the lane above
    clear = torch.prod(1.0 - infront, dim=-1)              # P(no car in front)
    return clear * _is_player(z).float()


# car_blocking tuning: how aligned (x) and how close-below (y) before the chicken waits.
# BLOCK_DY      = stop this many px below the car's lane (smaller => advance closer).
# BLOCK_SIGMA_X = horizontal block width on the APPROACHING side (anticipate the car).
# BLOCK_SIGMA_RECEDE = width on the RECEDING side (smaller => release / go up earlier once
#                 the car has passed the column and is moving away).
BLOCK_SIGMA_X = float(_os.environ.get("BLOCK_SIGMA_X", 12.0))
BLOCK_SIGMA_RECEDE = float(_os.environ.get("BLOCK_SIGMA_RECEDE", 7.0))
BLOCK_DY = float(_os.environ.get("BLOCK_DY", 15.0))
BLOCK_DY_SHARP = float(_os.environ.get("BLOCK_DY_SHARP", 3.0))


def _block_align(cx, cy, px):
    """Asymmetric horizontal block: wide on the side the car comes FROM (approaching),
    narrow on the side it moves TO (receding) so the chicken releases up earlier as the
    car passes. Direction is fixed by lane: top half (y<100) moves left, bottom moves right."""
    direction = torch.where(cy < 100, -1.0, 1.0)        # car travel direction in x
    travel = direction * (cx - px)                       # >0: car has passed (receding); <0: approaching
    sigma_sq = torch.where(travel > 0,
                           torch.full_like(travel, BLOCK_SIGMA_RECEDE ** 2),
                           torch.full_like(travel, BLOCK_SIGMA_X ** 2))
    return torch.exp(-((cx - px) ** 2) / (2 * sigma_sq))


def car_blocking(z_1: torch.Tensor, z_2: torch.Tensor):
    """Continuous: a car (z_1) is blocking the player's (z_2) immediate path up — high only
    when horizontally aligned AND the chicken is *just below* its lane. The horizontal term is
    asymmetric (see _block_align), so the chicken holds while the car approaches but starts up
    earlier as the car passes. BLOCK_DY sets the stop distance below the car."""
    x1, y1 = _get_x_y(z_1)
    x2, y2 = _get_x_y(z_2)
    dy = y2 - y1                                   # > 0 when the car is above the player
    align = _block_align(x1, y1, x2)
    near = torch.sigmoid((BLOCK_DY - dy) / BLOCK_DY_SHARP) * (dy > 3).float()  # high only just below
    return align * near * _is_player(z_2).float()


def path_clear_blocking(z: torch.Tensor, all_objects: torch.Tensor = None):
    """Continuous P(no car blocking the immediate path up) = prod_i (1 - car_blocking_i).
    Scene-level gate for `up`: climb until a car is aligned-and-just-above, then hold."""
    px, py = _get_x_y(z)
    if all_objects is None:
        return torch.zeros_like(px)
    is_car = all_objects[..., 2] > 0.5
    vis = all_objects[..., 0] > 0.5
    cx = all_objects[..., 3]; cy = all_objects[..., 4]
    dy = py.unsqueeze(-1) - cy
    align = _block_align(cx, cy, px.unsqueeze(-1))
    near = torch.sigmoid((BLOCK_DY - dy) / BLOCK_DY_SHARP) * (dy > 3).float()
    block = align * near * (is_car & vis).float()
    clear = torch.prod(1.0 - block, dim=-1)
    return clear * _is_player(z).float()


# car_speed_fast thresholds (on measured |dx| in px/frame). Env-tunable.
SPEED_FAST_THRESH = float(_os.environ.get("SPEED_FAST_THRESH", 0.6))
SPEED_FAST_SHARP = float(_os.environ.get("SPEED_FAST_SHARP", 0.2))


def car_speed_fast(z: torch.Tensor):
    """Graded probability that a car is FAST — higher for faster cars.

    Uses the MEASURED horizontal velocity dx (state feature index 5, displacement over a
    multi-frame window — Atari is higher-order Markov, so speed needs >1 frame). This
    generalizes to changed car speeds (HackAtari), unlike a lane->speed lookup. Falls back
    to the lane-speed table for the legacy 5-feature state (no velocity column).
    """
    if z.shape[-1] >= 6:
        dx = z[..., 5].abs()
        return torch.sigmoid((dx - SPEED_FAST_THRESH) / SPEED_FAST_SHARP)
    _, y = _get_x_y(z)
    speed = _lane_speed(y)
    return ((speed - 1.0) / 3.0).clamp(0.01, 0.99)


def _is_dangerous(cx, cy, px, py):
    """Shared danger test: a car (cx,cy) threatens the player's (px,py) column in the
    lane above if it is x-close (in the column now, either direction) OR approaching and
    within a speed-scaled lookahead (still in front / about to arrive)."""
    dy = py - cy                                   # >0 when car is above the player
    above = (dy >= 9) & (dy < 23)
    dxabs = (cx - px).abs()
    xclose = dxabs < X_CLOSE_PX                     # in the column right now (any direction)
    top = cy < 100                                  # top lanes move left, bottom move right
    approaching = (top & (cx > px)) | (~top & (cx < px))   # moving toward the column
    lookahead = X_CLOSE_PX + APPROACH_FRAMES * _lane_speed(cy)
    appr_close = approaching & (dxabs < lookahead)  # still in front, will arrive soon
    return above & (xclose | appr_close)


def car_dangerous(z_1: torch.Tensor, z_2: torch.Tensor):
    """The car (z_1) threatens the player's (z_2) column in the lane above — either it is
    in the column now (x_close) or it is approaching within a speed-scaled lookahead.
    This replaces the tight symmetric not_x_close so `up` no longer fires while a car is
    still closing in from the side."""
    x1, y1 = _get_x_y(z_1)
    x2, y2 = _get_x_y(z_2)
    return bool_to_probs(_is_dangerous(x1, y1, x2, y2) & _is_player(z_2))


def path_clear_above(z: torch.Tensor, all_objects: torch.Tensor = None):
    """No visible car is dangerous in the lane above the player (z). Scene-level gate for
    `up`: go up only if NOTHING threatens the column (fixes the per-car bug where a far car
    let up_safe fire while a close car should force noop)."""
    px, py = _get_x_y(z)
    if all_objects is None:
        return bool_to_probs(torch.zeros_like(px, dtype=torch.bool))
    is_car = all_objects[..., 2] > 0.5
    vis = all_objects[..., 0] > 0.5
    cx = all_objects[..., 3]; cy = all_objects[..., 4]
    danger = _is_dangerous(cx, cy, px.unsqueeze(-1), py.unsqueeze(-1)) & is_car & vis
    return bool_to_probs((~danger.any(dim=-1)) & _is_player(z))


# ── Atomic perception predicates (rules compose the semantics) ───────────────
# Horizontal-collision threshold, derived from data: P(human noop) crosses 0.5 at
# |dx| ~= 10 px (sharp drop over ~6-18 px). car_close_x is continuous around it.
CLOSE_X_PX = float(_os.environ.get("CLOSE_X_PX", 10.0))
CLOSE_X_SHARP = float(_os.environ.get("CLOSE_X_SHARP", 3.0))


def car_to_left(z_1: torch.Tensor, z_2: torch.Tensor):
    """Car (z_1) is to the LEFT of the player (z_2)."""
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    return bool_to_probs((x1 < x2) & _is_player(z_2))


def car_to_right(z_1: torch.Tensor, z_2: torch.Tensor):
    """Car (z_1) is to the RIGHT of the player (z_2)."""
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    return bool_to_probs((x1 > x2) & _is_player(z_2))


def car_above(z_1: torch.Tensor, z_2: torch.Tensor):
    """Car (z_1) is in the lane just ABOVE the player (z_2) (Δy ∈ [9,23) px)."""
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    dy = y2 - y1                                   # > 0 when the car is above
    return bool_to_probs((dy >= 9) & (dy < 23) & _is_player(z_2))


def car_below(z_1: torch.Tensor, z_2: torch.Tensor):
    """Car (z_1) is in the lane just BELOW the player (z_2) (Δy ∈ (-23,-9])."""
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    dy = y2 - y1
    return bool_to_probs((dy <= -9) & (dy > -23) & _is_player(z_2))


def car_close_x(z_1: torch.Tensor, z_2: torch.Tensor):
    """Continuous horizontal closeness of car (z_1) to the player's (z_2) column.
    ~1 when aligned, crossing 0.5 at |dx|≈CLOSE_X_PX (data-derived ≈10 px)."""
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    dx = (x1 - x2).abs()
    return torch.sigmoid((CLOSE_X_PX - dx) / CLOSE_X_SHARP) * _is_player(z_2).float()


def car_far_x(z_1: torch.Tensor, z_2: torch.Tensor):
    """Continuous complement of car_close_x: ~1 when the car (z_1) is NOT in the player's
    (z_2) column (|dx| well past CLOSE_X_PX). Used to gate `up` ('car above but not in my
    column -> safe to go'), symmetric with car_close_x gating `noop`."""
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    dx = (x1 - x2).abs()
    return torch.sigmoid((dx - CLOSE_X_PX) / CLOSE_X_SHARP) * _is_player(z_2).float()


def car_speed(z: torch.Tensor):
    """Graded speed of a car from its measured horizontal velocity dx (state index 5,
    displacement over a 3-4 frame window — Atari is higher-order Markov). Higher => faster.
    Falls back to the lane-speed table for the legacy 5-feature state (no velocity column)."""
    if z.shape[-1] >= 6:
        dx = z[..., 5].abs()
        return torch.sigmoid((dx - SPEED_FAST_THRESH) / SPEED_FAST_SHARP)
    _, y = _get_x_y(z)
    return ((_lane_speed(y) - 1.0) / 3.0).clamp(0.01, 0.99)


def car_same_row(z_1: torch.Tensor, z_2: torch.Tensor):
    """Car (z_1) is in the SAME row/lane as the player (z_2) (|Δy| < 9 px)."""
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    return bool_to_probs((torch.abs(y2 - y1) < 9) & _is_player(z_2))


def _direction(z):
    """Car horizontal travel direction: sign of measured velocity dx (6-feat), else lane
    (top half y<100 moves left=-1, bottom moves right=+1)."""
    if z.shape[-1] >= 6:
        return torch.sign(z[..., 5])
    _, y = _get_x_y(z)
    return torch.where(y < 100, torch.full_like(y, -1.0), torch.full_like(y, 1.0))


def car_approaching(z_1: torch.Tensor, z_2: torch.Tensor):
    """Car (z_1) is moving TOWARD the player's (z_2) column (will enter it). Uses measured
    velocity direction (3-4 frame dx) when available, else the lane->direction prior."""
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    d = _direction(z_1)
    appr = ((x1 > x2) & (d < 0)) | ((x1 < x2) & (d > 0))   # right-of & moving left, or left-of & moving right
    return bool_to_probs(appr & _is_player(z_2))


def car_receding(z_1: torch.Tensor, z_2: torch.Tensor):
    """Car (z_1) has passed the player's (z_2) column and is moving AWAY (gap opening)."""
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    d = _direction(z_1)
    rec = ((x1 > x2) & (d > 0)) | ((x1 < x2) & (d < 0))    # right-of & moving right, or left-of & moving left
    return bool_to_probs(rec & _is_player(z_2))


def not_x_close(z_1: torch.Tensor, z_2: torch.Tensor):
    """Negation of x_close: the car (z_1) is NOT in the player's (z_2) column.

    Used to make `up` and `noop` rules mutually exclusive (up fires when the lane
    above is safe, i.e. the car is x-far; noop when x-close). Under GRAIL's max
    aggregation this lets noop win on danger frames without any unconditional fact.
    """
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    result = (torch.abs(x1 - x2) >= X_CLOSE_PX) & _is_player(z_2)
    return bool_to_probs(result)


def lane_above_clear(z: torch.Tensor, all_objects: torch.Tensor = None):
    """No visible car occupies the lane immediately above the player (z).

    Aggregate predicate over the full object set (like Asterix's `no_object`),
    enabling an `up`-progress rule on the ~15.6% of frames where the lane above is
    empty (no car to condition the safe-up rules on). Near-binary.
    """
    _, py = _get_x_y(z)
    if all_objects is None:
        return bool_to_probs(torch.zeros_like(py, dtype=torch.bool))
    is_car = all_objects[..., 2] > 0.5            # (B, N_OBJ)
    vis = all_objects[..., 0] > 0.5
    cy = all_objects[..., 4]
    dy = py.unsqueeze(-1) - cy                     # player_y - car_y
    in_lane_above = (dy >= 9) & (dy < 23) & is_car & vis
    any_above = in_lane_above.any(dim=-1)
    return bool_to_probs((~any_above) & _is_player(z))


def on_left(z_1: torch.Tensor, z_2: torch.Tensor):
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    diff = x2 - x1
    result = (diff > 0) & _is_player(z_2)
    return bool_to_probs(result)
 
 
def on_right(z_1: torch.Tensor, z_2: torch.Tensor):
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    diff = x2 - x1
    result = (diff < 0) & _is_player(z_2)
    return bool_to_probs(result)
 
def in_front(z_1: torch.Tensor, z_2: torch.Tensor):
    x1, y1 = _get_x_y(z_1)
    x2, y2 = _get_x_y(z_2)
    diff = y2 - y1
    diff_x = torch.abs(x1 - x2)
    # Redefined: X-alignment for any object vertically above the player
    result = (diff > 0) & _is_player(z_2) & (diff_x < 12)
    return bool_to_probs(result)

def same_row(z_1: torch.Tensor, z_2: torch.Tensor):
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    diff = abs(y2 - y1)
    result = (diff <=8) & _is_player(z_2)
    return bool_to_probs(result)
 
 
def above_row(z_1: torch.Tensor, z_2: torch.Tensor):
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    
    diff = y2 - y1
    # z_1 is "above" z_2 if z_1 has smaller Y
    result = ((diff < 23) & (diff >= 9)) & _is_player(z_2)
    return bool_to_probs(result)
 
 
def below_row(z_1: torch.Tensor, z_2: torch.Tensor):
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    
    diff = y2 - y1
    # z_1 is "below" z_2 if z_1 has larger Y
    result = ((diff <= -4) & (diff > -23)) & _is_player(z_2)
    return bool_to_probs(result)
 
 
# Continuous above/below: probability decays with vertical distance, so a car DIRECTLY
# above (small Δy) scores higher than one several lanes up. Falloff midpoint ~one lane.
ABOVE_MID = float(_os.environ.get("ABOVE_MID", 16.0))
ABOVE_SHARP = float(_os.environ.get("ABOVE_SHARP", 6.0))


def above(z_1: torch.Tensor, z_2: torch.Tensor):
    """Continuous 'car (z_1) is above player (z_2)': ~1 directly above, decaying with the
    vertical gap (so the nearest above-lane car dominates a max over cars)."""
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    dy = y2 - y1                                        # > 0 when the car is above
    prob = torch.sigmoid((ABOVE_MID - dy) / ABOVE_SHARP) * (dy > 4).float()
    return prob * _is_player(z_2).float()


def below(z_1: torch.Tensor, z_2: torch.Tensor):
    """Continuous 'car (z_1) is below player (z_2)': ~1 directly below, decaying downward."""
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    dyb = y1 - y2                                       # > 0 when the car is below
    prob = torch.sigmoid((ABOVE_MID - dyb) / ABOVE_SHARP) * (dyb > 4).float()
    return prob * _is_player(z_2).float()
 
def top5car(z_1: torch.Tensor):
    _, y = _get_x_y(z_1)
    # y < 100 corresponds to top half
    result = bool_to_probs(y < 100)
    return result
 
 
def bottom5car(z_1: torch.Tensor):
    _, y = _get_x_y(z_1)
    # y > 100 corresponds to bottom half in Atari (0 is top)
    result = bool_to_probs(y > 100)
    return result
 
def topfastcar(z_1: torch.Tensor):
    _, y = _get_x_y(z_1)
    # Lane Y=107 is fast (following top5car y > 100 convention)
    result = bool_to_probs(abs(y - 107) < 5)
    return result
 
def bottomfastcar(z_1: torch.Tensor):
    _, y = _get_x_y(z_1)
    # Lane Y=91 is fast (following bottom5car y < 100 convention)
    result = bool_to_probs(abs(y - 91) < 5)
    return result


def above_row_free(z_1: torch.Tensor, z_2: torch.Tensor):
    x1, y1 = _get_x_y(z_1)
    x2, y2 = _get_x_y(z_2)
    
    diff = y2 - y1
    # z_1 is "above" z_2 if z_1 has smaller Y
    # Excluding same lane (diff < 9)
    # The closer x1 is to x2 the lower should be the probability 
    x_diff = abs(x1-x2)
    
    val = (x_diff-9)/2
    # Clip val between 0 and 1
    val = torch.clamp(val, 0, 1)
    result = ((diff < 23) & (diff >= 9)) & _is_player(z_2)
    return val * bool_to_probs(result)