import torch as th

from nsfr.utils.common import bool_to_probs


GAZE_LOWER_BOUND = 0.1  # Non-attended but present objects get at least this probability
LOWER_BOUND = 0.5
HIGHER_BOUND = 0.99

def visible_missile(obj: th.Tensor, gaze: th.Tensor = None, all_objects: th.Tensor = None) -> th.Tensor:
    """Probability that a missile is 'visible' (attended to).
    When gaze is provided together with all_objects, uses normalized gaze attention.
    Otherwise falls back to binary presence."""
    result = obj[..., 0] == 1
    val = bool_to_probs(result)
    if gaze is not None and all_objects is not None:
        gaze_val = gaze_object_attention_normalized(obj, gaze, all_objects)
        val = th.where(result, gaze_val, val)
    return val


def visible_enemy(obj: th.Tensor, gaze: th.Tensor = None, all_objects: th.Tensor = None) -> th.Tensor:
    """Probability that an enemy is 'visible' (attended to).
    When gaze is provided together with all_objects, uses normalized gaze attention.
    Otherwise falls back to binary presence."""
    result = obj[..., 0] == 1
    val = bool_to_probs(result)
    if gaze is not None and all_objects is not None:
        gaze_val = gaze_object_attention_normalized(obj, gaze, all_objects)
        val = th.where(result, gaze_val, val)
    return val


def visible_diver(obj: th.Tensor, gaze: th.Tensor = None, all_objects: th.Tensor = None) -> th.Tensor:
    """Probability that a diver is 'visible' (attended to).
    When gaze is provided together with all_objects, uses normalized gaze attention.
    Otherwise falls back to binary presence."""
    result = obj[..., 0] == 1
    val = bool_to_probs(result)
    if gaze is not None and all_objects is not None:
        gaze_val = gaze_object_attention_normalized(obj, gaze, all_objects)
        val = th.where(result, gaze_val, val)
    return val


def gaze_bbox_sum(obj: th.Tensor, gaze: th.Tensor) -> th.Tensor:
    """
    Compute the sum of gaze heatmap values within each object's bounding box.
    Uses a Summed Area Table (integral image) for speed.

    Args:
        obj:  (B, N_FEATURES)  — logic state row: [present, cx, cy, w, h, orient, type_id]
              Coordinates are in original game space (160×210).
        gaze: (B, 84, 84)      — normalized gaze heatmap (values sum to 1 per frame).

    Returns:
        raw_sum: (B,)  — sum of gaze mass inside the object's bbox (0 for absent objects).
    """
    B = obj.shape[0]
    device = obj.device

    # Scale from game space (160×210) to heatmap space (84×84)
    sx = 84.0 / 160.0
    sy = 84.0 / 210.0

    # Object centre coordinates → top-left corner for the bbox
    cx = obj[:, 1].float()
    cy = obj[:, 2].float()
    w  = obj[:, 3].float()
    h  = obj[:, 4].float()

    # Convert to heatmap pixel coordinates
    x  = ((cx - w / 2) * sx).long()
    y  = ((cy - h / 2) * sy).long()
    dw = (w * sx).long().clamp(min=1)
    dh = (h * sy).long().clamp(min=1)

    x1 = x.clamp(0, 84)
    y1 = y.clamp(0, 84)
    x2 = (x + dw).clamp(0, 84)
    y2 = (y + dh).clamp(0, 84)

    # Build integral image (SAT) — shape (B, 85, 85)
    if gaze.shape[-1] == 85:
        integral = gaze
    else:
        padded   = th.nn.functional.pad(gaze, (1, 0, 1, 0))
        integral = padded.cumsum(dim=1).cumsum(dim=2)

    b_idx = th.arange(B, device=device)

    # Four-corner lookup
    val_br = integral[b_idx, y2, x2]
    val_tl = integral[b_idx, y1, x1]
    val_tr = integral[b_idx, y1, x2]
    val_bl = integral[b_idx, y2, x1]

    raw_sum = (val_br - val_tr - val_bl + val_tl).clamp(min=0.0)

    # Zero out absent objects
    present = (obj[:, 0] > 0.5).float()
    return raw_sum * present


def gaze_object_attention_normalized(
    obj: th.Tensor,
    gaze: th.Tensor,
    all_objects: th.Tensor,
) -> th.Tensor:
    """
    Compute the gaze-attention probability for `obj`, normalized over all present
    objects in `all_objects`.

    The normalization ensures probabilities sum to 1 across all objects on screen,
    then each individual value is clipped to [GAZE_LOWER_BOUND, 1.0] so that even
    unattended objects remain 'seen' with at least GAZE_LOWER_BOUND probability.

    Args:
        obj:         (B, N_FEATURES)   — the specific object to compute visibility for.
        gaze:        (B, 84, 84)       — normalized gaze heatmap.
        all_objects: (B, N_OBJ, N_FEATURES) — full logic state for all objects in the scene.

    Returns:
        attention: (B,)  in [GAZE_LOWER_BOUND, 1.0]
    """
    B, N, _ = all_objects.shape
    device = obj.device

    # Precompute the integral image once for the whole batch
    if gaze.shape[-1] == 85:
        integral = gaze
    else:
        padded   = th.nn.functional.pad(gaze, (1, 0, 1, 0))
        integral = padded.cumsum(dim=1).cumsum(dim=2)

    # Sum gaze over every object slot — shape (B, N)
    # Flatten all_objects to (B*N, features), compute sums, reshape back
    flat_objs = all_objects.view(B * N, -1)                    # (B*N, F)
    flat_gaze = integral.unsqueeze(1).expand(-1, N, -1, -1)    # (B, N, 85, 85)
    flat_gaze = flat_gaze.reshape(B * N, 85, 85)               # (B*N, 85, 85)

    # We need gaze_bbox_sum to work on (B*N) items with matching integral images
    # Re-implement the SAT lookup inline to avoid re-padding
    sx = 84.0 / 160.0
    sy = 84.0 / 210.0

    cx  = flat_objs[:, 1].float()
    cy  = flat_objs[:, 2].float()
    w_f = flat_objs[:, 3].float()
    h_f = flat_objs[:, 4].float()

    x  = ((cx - w_f / 2) * sx).long()
    y  = ((cy - h_f / 2) * sy).long()
    dw = (w_f * sx).long().clamp(min=1)
    dh = (h_f * sy).long().clamp(min=1)

    x1 = x.clamp(0, 84);  y1 = y.clamp(0, 84)
    x2 = (x + dw).clamp(0, 84);  y2 = (y + dh).clamp(0, 84)

    bn_idx = th.arange(B * N, device=device)
    sums = (flat_gaze[bn_idx, y2, x2]
            - flat_gaze[bn_idx, y1, x2]
            - flat_gaze[bn_idx, y2, x1]
            + flat_gaze[bn_idx, y1, x1]).clamp(min=0.0)        # (B*N)

    # Zero out absent objects
    present_all = (flat_objs[:, 0] > 0.5).float()             # (B*N)
    sums = sums * present_all
    sums = sums.view(B, N)                                     # (B, N)

    # Normalize across all objects per frame
    total = sums.sum(dim=1, keepdim=True).clamp(min=1e-8)     # (B, 1)
    normalized = sums / total                                  # (B, N) summing to ≤1

    # Find which slot corresponds to `obj` by matching the first feature (present flag)
    # and coordinates — use sum of bbox features as a fingerprint
    obj_fp  = obj[:, 1:5].float()                             # (B, 4)  cx,cy,w,h
    all_fp  = all_objects[:, :, 1:5].float()                  # (B, N, 4)
    diff    = (all_fp - obj_fp.unsqueeze(1)).abs().sum(dim=-1) # (B, N)
    slot    = diff.argmin(dim=1)                               # (B,)

    b_idx   = th.arange(B, device=device)
    obj_attention = normalized[b_idx, slot]                    # (B,)

    # Clip to lower bound so absent objects stay suppressed, but present ones
    # always get at least GAZE_LOWER_BOUND
    is_present = (obj[:, 0] > 0.5).float()
    attention  = th.clamp(obj_attention, min=GAZE_LOWER_BOUND) * is_present + \
                 bool_to_probs(obj[:, 0] == 1) * (1 - is_present)

    return attention


def facing_left(player: th.Tensor) -> th.Tensor:
    # Orientation is at index 5
    result = player[..., 5] == 12
    return bool_to_probs(result)


def facing_right(player: th.Tensor) -> th.Tensor:
    # Orientation is at index 5
    result = player[..., 5] == 4
    return bool_to_probs(result)


def _vertical_iou(player: th.Tensor, obj: th.Tensor, h1: float, h2: float) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    
    y1_midpoint = player_y + h1/2
    y2_min = obj_y
    y2_max = obj_y + h2
    
    # Vectorized logic
    inside = (y1_midpoint > y2_min) & (y1_midpoint < y2_max)
    
    # Case: Below range (midpoint < min)
    diff_below = (player_y + h1) - y2_min
    val_below = th.clip(diff_below / h1, 0, 1)
    
    # Case: Above range (midpoint >= max)
    diff_above = y2_max - player_y
    val_above = th.clip(diff_above / h1, 0, 1)
    
    # If inside -> 1.0
    # Else if below -> val_below
    # Else -> val_above
    result = th.where(inside, th.tensor(1.0, device=player.device),
                      th.where(y1_midpoint < y2_min, val_below, val_above))
    
    return result


# Should be 0.99 if the midpoint of player is withing the bounding box of object
def same_depth_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1
    # Player (11) vs Enemy (10)
    iou = _vertical_iou(player, obj, 11, 10)
    return iou * bool_to_probs(obj_exists)


def same_depth_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1
    # Player (11) vs Diver (11)
    iou = _vertical_iou(player, obj, 11, 11)
    return iou * bool_to_probs(obj_exists)


def same_depth_missile(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1
    # Player (11) vs Missile (4)
    iou = _vertical_iou(player, obj, 11, 4)
    return iou * bool_to_probs(obj_exists)


def deeper_than_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'deeper than' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    result = obj_exists & (player_y > obj_y) & (same_depth_enemy(player, obj) < HIGHER_BOUND)
    # prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1)
    
    return bool_to_probs(result)


def deeper_than_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'deeper than' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    player_y = player[..., 2]
    obj_y = obj[..., 2]
  
    result = obj_exists & (player_y > obj_y) & (same_depth_diver(player, obj) < HIGHER_BOUND)
    # prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1) 

    return bool_to_probs(result)

# If there is an enemy below the player, then the player is higher than the enemy. Based on the distance from the enemy, the probability increased from the LOWER_BOUND_THRESHOLD
def higher_than_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'higher than' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    player_y = player[..., 2]
    # print("Player", player)
    # print("Object", obj)
    obj_y = obj[..., 2]
    result = obj_exists & (player_y < obj_y) & (same_depth_enemy(player, obj) < HIGHER_BOUND)
    
    # prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1) 

    # print("Result", result, "Object y", obj_y, "Player y", player_y, "prox", prox)
    return bool_to_probs(result)


def higher_than_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'higher than' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    
    # Calculate vertical difference (obj_y - player_y)
    # Since y increases downwards, higher means smaller y
    
    
    # Check if higher than threshold (11px)
    result = obj_exists & (player_y < obj_y) & (same_depth_diver(player, obj) < HIGHER_BOUND)

    
    # Old Logic: Increases with distance
    # prox = th.clip((result * (obj_y-player_y-11)/11), 0, 1)
    
    # New Logic: Decays with distance
    # Starts high near threshold (11px) and decays as distance increases
    # e.g. at 11px diff -> 1.0, at 51px diff -> 0.0
    # prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1) 
    return bool_to_probs(result)


def close_by_missile(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    proximity = _close_by(player, obj)
    # Only return proximity if object exists, else 0
    return proximity * bool_to_probs(obj_exists)

def not_close_by_missile(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    proximity = _close_by(player, obj)
    # Only return proximity if object exists, else 0
    return (1-proximity) * bool_to_probs(obj_exists)


def close_by_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    proximity = _close_by(player, obj)
    # Only return proximity if object exists, else 0
    return proximity * bool_to_probs(obj_exists)

def not_close_by_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    proximity = _close_by(player, obj)
    # Only return proximity if object exists, else 0
    return (1-proximity) * bool_to_probs(obj_exists)


def close_by_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    # proximity = _close_by(player, obj)
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    proximity = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1)
    # Only return proximity if object exists, else 0
    return proximity * bool_to_probs(obj_exists)


def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    player_y = player[..., 2]
    obj_x = obj[..., 1]
    obj_y = obj[..., 2]
    result = th.clip((128 - abs(player_x - obj_x) - abs(player_y - obj_y)) / 128, 0, 1)
    #use a threshold of 15 px and return 1 if the distance is less than 15 px else 0
    # bool_val = abs(player_x - obj_x) + abs(player_y - obj_y) < 50
    return result


def left_of_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'left of' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    player_x = player[..., 1]
    player_width = player[..., 3]
    obj_x = obj[..., 1]

    result = obj_exists & (player_x + player_width < obj_x)
    return bool_to_probs(result)


def left_of_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'left of' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    player_x = player[..., 1]
    player_width = player[..., 3]
    obj_x = obj[..., 1]
    result = obj_exists & (player_x + player_width < obj_x)
    return bool_to_probs(result)


def right_of_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'right of' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    obj_width= obj[..., 3]
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    result = obj_exists & (player_x > obj_x+obj_width)
    return bool_to_probs(result)


def right_of_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'right of' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    obj_width= obj[..., 3]
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    result = obj_exists & (player_x > obj_x+obj_width)
    return bool_to_probs(result)


def oxygen_low(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar width is below 16 pixels (approximately 25% oxygen remaining)."""
    oxygen_width = oxygen_bar[..., 3]  # Width in pixels (index 3)
    result = oxygen_width < 16
    
    # DEBUG: Print first few calls
    return bool_to_probs(result)


def in_image(zs: th.Tensor, obj: th.Tensor) -> th.Tensor:
    # Check if object is visible (index 0 is 1)
    return bool_to_probs(obj[..., 0] == 1)


# ADDED Predicates

def on_left(obj1: th.Tensor, obj2: th.Tensor) -> th.Tensor:
    vis = (obj1[..., 0] == 1) & (obj2[..., 0] == 1)
    return bool_to_probs(vis & (obj1[..., 1] < obj2[..., 1]))


def on_right(obj1: th.Tensor, obj2: th.Tensor) -> th.Tensor:
    vis = (obj1[..., 0] == 1) & (obj2[..., 0] == 1)
    return bool_to_probs(vis & (obj1[..., 1] > obj2[..., 1]))


def on_top(obj1: th.Tensor, obj2: th.Tensor) -> th.Tensor:
    # A is above B (smaller Y)
    vis = (obj1[..., 0] == 1) & (obj2[..., 0] == 1)
    return bool_to_probs(vis & (obj1[..., 2] < obj2[..., 2]))


def at_bottom(obj1: th.Tensor, obj2: th.Tensor) -> th.Tensor:
    # A is at bottom of screen. Ignoring obj2.
    vis = obj1[..., 0] == 1
    # Check if Y > 170 (approx bottom)
    return bool_to_probs(vis & (obj1[..., 2] > 170))


def closeby(obj1: th.Tensor, obj2: th.Tensor) -> th.Tensor:
    # Use existing helper
    obj1_exists = obj1[..., 0] == 1
    obj2_exists = obj2[..., 0] == 1
    proximity = _close_by(obj1, obj2) 
    return proximity * bool_to_probs(obj1_exists & obj2_exists)


def type(obj: th.Tensor, type_oh: th.Tensor) -> th.Tensor:
    # Check type equality
    # obj has type_id at index 6
    obj_type_id = obj[..., 6].long()
    
    # type_oh is one-hot vector, get index
    target_type_id = type_oh.argmax(dim=-1)
    
    # Check if object exists
    vis = obj[..., 0] == 1
    
    match = (obj_type_id == target_type_id)
    return bool_to_probs(vis & match)


# NEW PREDICATES

def divers_collected_full(obj: th.Tensor) -> th.Tensor:
    """True if the 6th collected diver exists (implying full capacity)."""
    # This predicate should be bound to the 6th collected diver slot (obj45) in neural_preds.txt
    return bool_to_probs(obj[..., 0] == 1)

def oxygen_critical(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar width is below 5 pixels (critical)."""
    oxygen_width = oxygen_bar[..., 3] # Width in pixels (index 3)
    result = oxygen_width < 5
    return bool_to_probs(result)

def surface_submarine(obj: th.Tensor) -> th.Tensor:
    """True if object is the Surface Submarine."""
    # Surface Submarine is usually located at the very top of the screen (y < 40).
    vis = obj[..., 0] == 1
    y = obj[..., 2]
    is_top = y < 40
    return bool_to_probs(vis & is_top)

def is_collected_diver(obj: th.Tensor) -> th.Tensor:
    """True if object is a collected diver."""
    # Collected divers are shown at the bottom of the screen (y > 160).
    vis = obj[..., 0] == 1
    y = obj[..., 2]
    is_bottom = y > 160
    return bool_to_probs(vis & is_bottom)


    # In `env.py`, they are converted to `[1, x, y, 0, type]`.
    # If I rename `divers_collected_full` to `all_divers_collected(player)`? No.
    #
    # I will implement `oxygen_critical` and `surface_submarine` first.
    pass

def above_water(player: th.Tensor) -> th.Tensor:
    """True if player is above water (at surface, y < 55)."""
    # Uses same threshold as surface_submarine
    vis = player[..., 0] == 1
    y = player[..., 2]
    is_surface = y < 55
    return bool_to_probs(vis & is_surface)


def below_water(player: th.Tensor) -> th.Tensor:
    """True if player is below water (at surface, y > 55)."""
    vis = player[..., 0] == 1
    y = player[..., 2]
    is_surface = y > 55
    return bool_to_probs(vis & is_surface)


def above_surface(player: th.Tensor, surface: th.Tensor) -> th.Tensor:
    """True if player is above the surface."""
    player_vis = player[..., 0] == 1
    surface_vis = surface[..., 0] == 1
    player_y = player[..., 2]
    surface_y = surface[..., 2]
    
    result = player_vis & surface_vis & (player_y < surface_y)
    return bool_to_probs(result)


def below_surface(player: th.Tensor, surface: th.Tensor) -> th.Tensor:
    """True if player is below the surface."""
    player_vis = player[..., 0] == 1
    surface_vis = surface[..., 0] == 1
    player_y = player[..., 2]
    surface_y = surface[..., 2]
    
    result = player_vis & surface_vis & (player_y > surface_y)
    return bool_to_probs(result)
