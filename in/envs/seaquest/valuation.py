import torch as th

from nsfr.utils.common import bool_to_probs


LOWER_BOUND = 0.60
HIGHER_BOUND = 0.99
def visible_missile(obj: th.Tensor) -> th.Tensor:
    result = obj[..., 0] == 1
    return bool_to_probs(result)


def visible_enemy(obj: th.Tensor) -> th.Tensor:
    result = obj[..., 0] == 1
    return bool_to_probs(result)


def visible_diver(obj: th.Tensor) -> th.Tensor:
    result = obj[..., 0] == 1
    return bool_to_probs(result)


def facing_left(player: th.Tensor) -> th.Tensor:
    result = player[..., 3] == 12
    return bool_to_probs(result)


def facing_right(player: th.Tensor) -> th.Tensor:
    result = player[..., 3] == 4
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
    prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1)
    
    return bool_to_probs(result)


def deeper_than_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'deeper than' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    player_y = player[..., 2]
    obj_y = obj[..., 2]
  
    result = obj_exists & (player_y > obj_y) & (same_depth_diver(player, obj) < HIGHER_BOUND)
    prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1) 

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
    
    prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1) 

    # print("Result", result, "Object y", obj_y, "Player y", player_y, "prox", prox)
    return prox*bool_to_probs(result)


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
    prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1) 
    return bool_to_probs(result)


def close_by_missile(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    proximity = _close_by(player, obj)
    # Only return proximity if object exists, else 0
    return proximity * bool_to_probs(obj_exists)


def close_by_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    proximity = _close_by(player, obj)
    # Only return proximity if object exists, else 0
    return proximity * bool_to_probs(obj_exists)


def close_by_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    proximity = _close_by(player, obj)
    # Only return proximity if object exists, else 0
    return proximity * bool_to_probs(obj_exists)


def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    player_y = player[..., 2]
    obj_x = obj[..., 1]
    obj_y = obj[..., 2]
    result = th.clip((128 - abs(player_x - obj_x) - abs(player_y - obj_y)) / 128, 0, 1)
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
    oxygen_width = oxygen_bar[..., 1]  # Width in pixels
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
    # obj has type_id at index 4
    obj_type_id = obj[..., 4].long()
    
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
    oxygen_width = oxygen_bar[..., 1]
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
