import torch as th

from nsfr.utils.common import bool_to_probs
HIGHER_BOUND=0.98

def visible_missile(missile: th.Tensor) -> th.Tensor:
    """Probability that a missile is 'visible' (present).
    Gaze-based attention is now handled by the core ValuationModule."""
    return bool_to_probs(missile[..., 0] == 1)


def visible_enemy(enemy: th.Tensor) -> th.Tensor:
    """Probability that an enemy is 'visible' (present).
    Gaze-based attention is now handled by the core ValuationModule."""
    return bool_to_probs(enemy[..., 0] == 1)


def visible_diver(diver: th.Tensor) -> th.Tensor:
    """Probability that a diver is 'visible' (present).
    Gaze-based attention is now handled by the core ValuationModule."""
    return bool_to_probs(diver[..., 0] == 1)


def directly_above_enemy(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    obj_exists = enemy[..., 0] == 1  # Check if object exists/visible
    player_y = player[..., 2]
    obj_y = enemy[..., 2]
    result = obj_exists & (player_y < obj_y) 
    overlap = _horizontal_iou(player, enemy, 11, 10)
    return bool_to_probs(result) * overlap

def not_directly_above_enemy(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    obj_exists = enemy[..., 0] == 1
    return bool_to_probs(obj_exists) * (1.0 - directly_above_enemy(player, enemy))

def directly_below_enemy(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    obj_exists = enemy[..., 0] == 1  # Check if object exists/visible
    player_y = player[..., 2]
    obj_y = enemy[..., 2]
    result = obj_exists & (player_y > obj_y) 
    overlap = _horizontal_iou(player, enemy, 11, 10)
    return bool_to_probs(result) * overlap


def not_directly_below_enemy(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    obj_exists = enemy[..., 0] == 1
    return bool_to_probs(obj_exists) * (1.0 - directly_below_enemy(player, enemy))

def facing_left(player: th.Tensor) -> th.Tensor:
    # Orientation is at index 5
    result = player[..., 5] == 12
    return bool_to_probs(result)


def facing_right(player: th.Tensor) -> th.Tensor:
    # Orientation is at index 5
    result = player[..., 5] == 4
    return bool_to_probs(result)

def enemy_facing_left(enemy: th.Tensor) -> th.Tensor:
    # Orientation is at index 5
    result = enemy[..., 5] == 12
    return bool_to_probs(result)

def enemy_facing_right(enemy: th.Tensor) -> th.Tensor:
    # Orientation is at index 5
    result = enemy[..., 5] == 4
    return bool_to_probs(result)

def _vertical_iou(player: th.Tensor, obj: th.Tensor, h1: float, h2: float) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    
    y1_midpoint = player_y + 2*h1/3
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


def _fireable_iou(player: th.Tensor, obj: th.Tensor, h1: float, h2: float) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    
    y1_midpoint = player_y + 2*h1/3
    y2_min = obj_y
    y2_max = obj_y + h2
    y2_midpoint = obj_y + h2/2
    y2_min = y2_midpoint - h2/4
    y2_max = y2_midpoint + h2/4
    
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

def _horizontal_iou(player: th.Tensor, obj: th.Tensor, w1: float, w2: float) -> th.Tensor:
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    
    x1_midpoint = player_x + w1/2
    x2_min = obj_x - w1/2
    x2_max = obj_x + w2 + w1
    
    # Vectorized logic
    inside = (x1_midpoint > x2_min) & (x1_midpoint < x2_max)
    
    # Case: Below range (midpoint < min)
    diff_right = (player_x + w1) - x2_min
    val_right= th.clip(diff_right / w1, 0, 1)
    
    # Case: Above range (midpoint >= max)
    diff_left = x2_max - player_x
    val_left = th.clip(diff_left / w1, 0, 1)
    
    # If inside -> 1.0
    # Else if below -> val_below
    # Else -> val_above
    result = th.where(inside, th.tensor(1.0, device=player.device),
                      th.where(x1_midpoint < x2_min, val_right*2, val_left*2))
    
    return result

# Should be 0.99 if the midpoint of player is withing the bounding box of object
def same_depth_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1
    # Player (11) vs Enemy (10)
    iou = _vertical_iou(player, obj, 11, 10)
    return iou * bool_to_probs(obj_exists)

def fireable_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1
    # Player (11) vs Enemy (10)
    iou = _fireable_iou(player, obj, 11, 10)
    return iou * bool_to_probs(obj_exists)


def atleast_one_diver_collected(dummy_player, all_objects: th.Tensor = None) -> th.Tensor:
    """True if at least one collected diver is visible (y > 160)."""
    if all_objects is None:
        return th.tensor([0.01], device=dummy_player.device)
    
    vis = all_objects[..., 0] == 1
    y = all_objects[..., 2]
    is_collected = vis & (y > 160)
    
    any_collected = th.any(is_collected, dim=1)
    return bool_to_probs(any_collected)

def same_depth_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1
    # Player (11) vs Diver (11)
    iou = _vertical_iou(player, obj, 11, 11)
    return iou * bool_to_probs(obj_exists)

# def above_missile(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
#     obj_exists = obj[..., 0] == 1
#     # Player (11) vs Missile (4)
#     iou = _vertical_iou(player, obj, 11, 4)
#     return iou * bool_to_probs(obj_exists)

# def below_missile(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
#     obj_exists = obj[..., 0] == 1
#     # Player (11) vs Missile (4)
#     iou = _vertical_iou(player, obj, 11, 4)
#     return iou * bool_to_probs(obj_exists)

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
    non_overlap = 1 - _vertical_iou(player, obj, 11, 10)
    # prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1)
    
    return bool_to_probs(result) * non_overlap


def deeper_than_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'deeper than' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    player_y = player[..., 2]
    obj_y = obj[..., 2]
  
    result = obj_exists & (player_y > obj_y) & (same_depth_diver(player, obj) < HIGHER_BOUND)
    non_overlap = 1 - _vertical_iou(player, obj, 11, 11)
    # prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1) 

    return bool_to_probs(result) * non_overlap

# If there is an enemy below the player, then the player is higher than the enemy. Based on the distance from the enemy, the probability increased from the LOWER_BOUND_THRESHOLD
def higher_than_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'higher than' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    player_y = player[..., 2]
    # print("Player", player)
    # print("Object", obj)
    obj_y = obj[..., 2]
    result = obj_exists & (player_y < obj_y)
    non_overlap = 1 - _vertical_iou(player, obj, 11, 10)
    
    # prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1) 

    # print("Result", result, "Object y", obj_y, "Player y", player_y, "prox", prox)
    return bool_to_probs(result) * non_overlap


def higher_than_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'higher than' the object."""
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    
    # Calculate vertical difference (obj_y - player_y)
    # Since y increases downwards, higher means smaller y
    # Check if higher than threshold (11px)
    result = obj_exists & (player_y < obj_y) & (same_depth_diver(player, obj) < HIGHER_BOUND)
    non_overlap = 1 - _vertical_iou(player, obj, 11, 10)
    # Old Logic: Increases with distance
    # prox = th.clip((result * (obj_y-player_y-11)/11), 0, 1)
    
    # New Logic: Decays with distance
    # Starts high near threshold (11px) and decays as distance increases
    # e.g. at 11px diff -> 1.0, at 51px diff -> 0.0
    # prox = th.clip((obj_y-player_y)/(100), LOWER_BOUND, 1) 
    return bool_to_probs(result) * non_overlap


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

def left_of_missile(player: th.Tensor, missile: th.Tensor) -> th.Tensor:
    """True iff the player is to the left of the missile."""
    obj_exists = missile[..., 0] == 1
    return bool_to_probs(obj_exists & (player[..., 1] < missile[..., 1]))

def right_of_missile(player: th.Tensor, missile: th.Tensor) -> th.Tensor:
    """True iff the player is to the right of the missile."""
    obj_exists = missile[..., 0] == 1
    return bool_to_probs(obj_exists & (player[..., 1] > missile[..., 1]))

def higher_than_missile(player: th.Tensor, missile: th.Tensor) -> th.Tensor:
    """True iff the player is vertically higher than the missile."""
    obj_exists = missile[..., 0] == 1
    return bool_to_probs(obj_exists & (player[..., 2] < missile[..., 2]))

def deeper_than_missile(player: th.Tensor, missile: th.Tensor) -> th.Tensor:
    """True iff the player is vertically deeper than the missile."""
    obj_exists = missile[..., 0] == 1
    return bool_to_probs(obj_exists & (player[..., 2] > missile[..., 2]))


def close_by_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    proximity = _close_by(player, obj)
    # Only return proximity if object exists, else 0
    return proximity * bool_to_probs(obj_exists)

def very_close_by_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is very close to the enemy based on edge proximity."""
    obj_exists = obj[..., 0] == 1
    player_x, player_y = player[..., 1], player[..., 2]
    player_w, player_h = player[..., 3], player[..., 4]
    obj_x, obj_y = obj[..., 1], obj[..., 2]
    obj_w, obj_h = obj[..., 3], obj[..., 4]
    obj_orient = obj[..., 5]

    # Conditions from user:
    # 1. enemy facing right (4) and enemy_x+enemy_width is within 5 pixels of player_x
    cond1 = (obj_orient == 4) & (th.abs(player_x - (obj_x + obj_w)) < 5)
    # 2. enemy facing left (12) and enemy_x is within 5 pixels of player_x + player_width
    cond2 = (obj_orient == 12) & (th.abs(obj_x - (player_x + player_w)) < 5)
    # 3. enemy_y is within 5 pixels of player_y + player_height
    cond3 = th.abs(obj_y - (player_y + player_h)) < 5
    # 4. enemy_y + enemy_height is within 5 pixels of player_y
    cond4 = th.abs(player_y - (obj_y + obj_h)) < 5

    combined = cond1 | cond2 | cond3 | cond4
    return bool_to_probs(obj_exists & combined)

def closest_enemy(player: th.Tensor, enemy: th.Tensor, all_objects: th.Tensor = None) -> th.Tensor:
    if all_objects is None:
        return visible_enemy(enemy)
        
    # player: (B*N, F), enemy: (B*N, F), all_objects: (B*N, N_OBJ, F)
    # Compute distance from player to target enemy
    target_dist = th.abs(player[..., 1] - enemy[..., 1]) + th.abs(player[..., 2] - enemy[..., 2])
    
    # Compute distances from player to all objects
    player_expanded = player.unsqueeze(1)
    all_dists = th.abs(player_expanded[..., 1] - all_objects[..., 1]) + th.abs(player_expanded[..., 2] - all_objects[..., 2])
    
    # Identify enemies: type_id 0 (at index 6) and visible (at index 0)
    is_enemy = (all_objects[..., 6] == 0) & (all_objects[..., 0] == 1)
    
    # Mask non-enemies with large distance
    enemy_dists = th.where(is_enemy, all_dists, th.tensor(1000.0, device=all_objects.device))
    
    # Find minimum distance to any enemy
    min_dist, _ = th.min(enemy_dists, dim=1)
    
    # Check if target enemy is the closest (using small epsilon for float comparison)
    is_closest = (target_dist <= min_dist + 1e-3) & (enemy[..., 0] == 1)
    return bool_to_probs(is_closest)

def not_close_by_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    proximity = _close_by(player, obj)
    # Only return proximity if object exists, else 0
    return (1-proximity) * bool_to_probs(obj_exists)



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
    result = th.clip((300 - abs(player_x - obj_x) - abs(player_y - obj_y)) / 300, 0, 1)
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

def oxygen_full(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar width is below 16 pixels (approximately 25% oxygen remaining)."""
    oxygen_width = oxygen_bar[..., 3]  # Width in pixels (index 3)
    result = oxygen_width >= 48
    
    # DEBUG: Print first few calls
    return bool_to_probs(result)

def oxygen_not_full(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar width is below 16 pixels (approximately 25% oxygen remaining)."""
    oxygen_width = oxygen_bar[..., 3]  # Width in pixels (index 3)
    result = oxygen_width < 48
    
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
    # pass

def no_object(dummy_player, all_objects: th.Tensor = None) -> th.Tensor:
    """True if there are no enemies (type 0) and no divers (type 1) visible in the scene."""
    if all_objects is None:
        return th.tensor([0.01], device=dummy_player.device)
    
    # Identify enemies and divers: visible (index 0) and type_id in {0, 1}
    vis = all_objects[..., 0] == 1
    type_ids = all_objects[..., 6]
    is_target = vis & ((type_ids == 0) | (type_ids == 1))
    
    # Any target object exists in the scene?
    any_target = th.any(is_target, dim=1)
    
    # Return probability: True if NOT any_target
    return bool_to_probs(~any_target)

def above_water(player: th.Tensor) -> th.Tensor:
    """True if player is above water (at surface, y < 55)."""
    # Uses same threshold as surface_submarine
    vis = player[..., 0] == 1
    y = player[..., 2]
    is_surface = y < 50
    return bool_to_probs(vis & is_surface)


def below_water(player: th.Tensor) -> th.Tensor:
    """True if player is below water (at surface, y > 55)."""
    vis = player[..., 0] == 1
    y = player[..., 2]
    is_surface = y > 50
    return bool_to_probs(vis & is_surface)

def oxygen_not_low(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar width is greater than 16 pixels (approximately 25% oxygen remaining)."""
    oxygen_width = oxygen_bar[..., 3]  # Width in pixels (index 3)
    result = oxygen_width >= 16
    
    return bool_to_probs(result)

def player_left_side(player: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    result = player_x <= 50
    return bool_to_probs(result)

def player_right_side(player: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    result = player_x >= 125
    return bool_to_probs(result)


def above_surface(player: th.Tensor, surface: th.Tensor) -> th.Tensor:
    """True if player is above the surface."""
    player_vis = player[..., 0] == 1
    surface_vis = surface[..., 0] == 1
    player_y = player[..., 2]
    surface_y = surface[..., 2]
    
    result = player_vis & surface_vis & (player_y + 5 < surface_y)
    return bool_to_probs(result)


def below_surface(player: th.Tensor, surface: th.Tensor) -> th.Tensor:
    """True if player is below the surface."""
    player_vis = player[..., 0] == 1
    surface_vis = surface[..., 0] == 1
    player_y = player[..., 2]
    surface_y = surface[..., 2]
    
    result = player_vis & surface_vis & (player_y > surface_y)
    return bool_to_probs(result)


def no_divers_collected(dummy_player, all_objects: th.Tensor = None) -> th.Tensor:
    """True if no collected divers are visible (y > 160)."""
    if all_objects is None:
        return th.tensor([0.99], device=dummy_player.device)
    
    vis = all_objects[..., 0] == 1
    y = all_objects[..., 2]
    # Collected divers are at the bottom (y > 160)
    is_collected = vis & (y > 160)
    
    # If ANY collected diver is visible, it's NOT empty.
    any_collected = th.any(is_collected, dim=1)
    
    return bool_to_probs(~any_collected)
