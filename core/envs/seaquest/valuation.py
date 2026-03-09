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
    is_above = player_y < obj_y
    
    player_left = player[..., 1]
    player_right = player[..., 1] + player[..., 3]
    enemy_left = enemy[..., 1]
    enemy_right = enemy[..., 1] + enemy[..., 3]
    
    dist_left = th.clamp(enemy_left - player_right, min=0.0)
    dist_right = th.clamp(player_left - enemy_right, min=0.0)
    dist = th.maximum(dist_left, dist_right)
    
    overlap_prob = th.clamp(0.99 * (1.0 - dist / 3.0), min=0.0, max=0.99)
    result = obj_exists & is_above
    return bool_to_probs(result) * overlap_prob

def not_directly_above_enemy(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    obj_exists = enemy[..., 0] == 1
    return bool_to_probs(obj_exists) * (1.0 - directly_above_enemy(player, enemy))

def directly_below_enemy(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    obj_exists = enemy[..., 0] == 1  # Check if object exists/visible
    player_y = player[..., 2]
    obj_y = enemy[..., 2]
    is_below = player_y > obj_y
    
    player_left = player[..., 1]
    player_right = player[..., 1] + player[..., 3]
    enemy_left = enemy[..., 1]
    enemy_right = enemy[..., 1] + enemy[..., 3]
    
    dist_left = th.clamp(enemy_left - player_right, min=0.0)
    dist_right = th.clamp(player_left - enemy_right, min=0.0)
    dist = th.maximum(dist_left, dist_right)
    
    overlap_prob = th.clamp(0.99 * (1.0 - dist / 3.0), min=0.0, max=0.99)
    result = obj_exists & is_below
    return bool_to_probs(result) * overlap_prob


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
    
    y1_midpoint = player_y + 3*h1/4
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
    
    y1_midpoint = player_y + 3*h1/4
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
    type_ids = all_objects[..., 6]
    # CollectedDiver is type 6. Strictly filter by type to avoid catching OxygenBar/Surface.
    is_collected = vis & (y > 160) & (type_ids == 6)
    
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
    return bool_to_probs(result)


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
    obj_y = obj[..., 2]
    result = obj_exists & (player_y < obj_y)
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

    # Calculate horizontal distances to edges based on facing direction
    # Enemy facing right (4): check distance from enemy right edge to player left edge
    dist1 = th.abs(player_x - (obj_x + obj_w))
    # Enemy facing left (12): check distance from enemy left edge to player right edge
    dist2 = th.abs(obj_x - (player_x + player_w))
    
    # Use the appropriate distance based on orientation
    dist = th.where(obj_orient == 4, dist1, dist2)
    
    # Calculate vertical overlap (relaxing the strict 4-pixel tolerance to a gentle probability curve)
    vert_dist = th.abs((player_y + player_h/2) - (obj_y + obj_h/2))
    vert_prob = th.clip(1.0 - (vert_dist / 14.0), 0.0, 1.0)
    
    # Calculate continuous proximity probability
    # If distance is 0, prob is 1.0. Decays to 0 at distance 16.
    max_dist = 16.0
    horiz_prob = th.clip(1.0 - (dist / max_dist), 0.0, 1.0)

    result = obj_exists & (horiz_prob > 0) & (vert_prob > 0)
    return horiz_prob * vert_prob * bool_to_probs(result)

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
def closest_diver(player: th.Tensor, diver: th.Tensor, all_objects: th.Tensor = None) -> th.Tensor:
    if all_objects is None:
        return visible_diver(diver)
        
    # player: (B*N, F), diver: (B*N, F), all_objects: (B*N, N_OBJ, F)
    target_dist = th.abs(player[..., 1] - diver[..., 1]) + th.abs(player[..., 2] - diver[..., 2])
    player_expanded = player.unsqueeze(1)
    all_dists = th.abs(player_expanded[..., 1] - all_objects[..., 1]) + th.abs(player_expanded[..., 2] - all_objects[..., 2])
    
    # Identify divers: type_id 1 and visible
    is_diver = (all_objects[..., 6] == 1) & (all_objects[..., 0] == 1)
    
    # Mask non-divers with large distance
    diver_dists = th.where(is_diver, all_dists, th.tensor(1000.0, device=all_objects.device))
    min_dist, _ = th.min(diver_dists, dim=1)
    is_closest = (target_dist <= min_dist + 1e-3) & (diver[..., 0] == 1)
    return bool_to_probs(is_closest)

def closest_missile(player: th.Tensor, missile: th.Tensor, all_objects: th.Tensor = None) -> th.Tensor:
    if all_objects is None:
        return visible_missile(missile)
        
    target_dist = th.abs(player[..., 1] - missile[..., 1]) + th.abs(player[..., 2] - missile[..., 2])
    player_expanded = player.unsqueeze(1)
    all_dists = th.abs(player_expanded[..., 1] - all_objects[..., 1]) + th.abs(player_expanded[..., 2] - all_objects[..., 2])
    
    # Identify missiles: type_id 5 and visible
    is_missile = (all_objects[..., 6] == 5) & (all_objects[..., 0] == 1)
    
    missile_dists = th.where(is_missile, all_dists, th.tensor(1000.0, device=all_objects.device))
    min_dist, _ = th.min(missile_dists, dim=1)
    is_closest = (target_dist <= min_dist + 1e-3) & (missile[..., 0] == 1)
    return bool_to_probs(is_closest)

def not_close_by_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1  # Check if object exists/visible
    proximity = _close_by(player, obj)
    # Only return proximity if object exists, else 0
    return (1-proximity) * bool_to_probs(obj_exists)

def horizontally_far_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Continuous probability of being horizontally far from the enemy. 
    Returns 0.0 when same X, up to 1.0 when X distance > 50px."""
    obj_exists = obj[..., 0] == 1
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    
    dist_x = th.abs(player_x - obj_x)
    
    prob = th.clip(dist_x / 50.0, 0.0, 1.0)
    return prob * bool_to_probs(obj_exists)

def horizontally_close_enemy(player: th.Tensor, obj: th,Tensor) -> th.Tensor:
    obj_exists = obj[..., 0] == 1
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    
    dist_x = th.abs(player_x - obj_x)
    
    prob = th.clip(1 - (dist_x / 50.0), 0.0, 1.0)
    return prob * bool_to_probs(obj_exists)
    
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
    vis = oxygen_bar[..., 0] == 1
    oxygen_width = oxygen_bar[..., 3]  # Width in pixels (index 3)
    return th.where(vis, th.clip(1.0 - (oxygen_width / 16.0), 0.01, 0.99), th.tensor(0.01, device=oxygen_bar.device))

def oxygen_full(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar width is at least 48 pixels."""
    vis = oxygen_bar[..., 0] == 1
    oxygen_width = oxygen_bar[..., 3]  # Width in pixels (index 3)
    return th.where(vis, th.clip(oxygen_width / 48.0, 0.01, 0.99), th.tensor(0.01, device=oxygen_bar.device))

def oxygen_not_full(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar width is < 48 pixels."""
    vis = oxygen_bar[..., 0] == 1
    oxygen_width = oxygen_bar[..., 3]  # Width in pixels (index 3)
    return th.where(vis, th.clip(1.0 - (oxygen_width / 48.0), 0.01, 0.99), th.tensor(0.01, device=oxygen_bar.device))

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

# Global memory to prevent sprite flickering from dropping the "full" status
_divers_full_latch = False
_frames_above_water = 0

def divers_collected_full(dummy_player, all_objects: th.Tensor = None) -> th.Tensor:
    """True if 6 divers are collected. Uses a strict global latch to prevent flickering."""
    global _divers_full_latch, _frames_above_water
    device = dummy_player.device
    
    if all_objects is None:
        return th.tensor([0.01], device=device)
        
    vis = all_objects[..., 0] == 1
    type_ids = all_objects[..., 6]
    is_collected = vis & (type_ids == 6)
    
    num_collected = th.sum(is_collected, dim=1) # Shape: (B,)
    
    # Check if player is above water (use the first batch element for the global latch)
    # y < 48 means above water (matching above_water predicate).
    player_y = dummy_player[0, 2]
    
    # Update global latch with hysteresis (5 frames)
    if player_y < 48:
        _frames_above_water += 1
        if _frames_above_water > 5:
            _divers_full_latch = False
    else:
        _frames_above_water = 0
        if num_collected[0].item() >= 6:
            _divers_full_latch = True
            
    # Broadcast the global latch to the batch dimension
    batch_size = num_collected.shape[0]
    result = th.full((batch_size,), _divers_full_latch, dtype=th.bool, device=device)
    
    return bool_to_probs(result)

def divers_collected_not_full(dummy_player, all_objects: th.Tensor = None) -> th.Tensor:
    """True if fewer than 6 divers are collected. Inverse of full."""
    # We use the full probability and invert it exactly to maintain consistency
    prob_full = divers_collected_full(dummy_player, all_objects)
    return 1.0 - prob_full + 0.02 # Offset to map 0.99 -> 0.01 and 0.01 -> 0.99


def oxygen_critical(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar width is below 5 pixels (critical)."""
    vis = oxygen_bar[..., 0] == 1
    oxygen_width = oxygen_bar[..., 3] # Width in pixels (index 3)
    return th.where(vis, th.clip(1.0 - (oxygen_width / 5.0), 0.01, 0.99), th.tensor(0.01, device=oxygen_bar.device))

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

def few_objects(dummy_player, all_objects: th.Tensor = None) -> th.Tensor:
    """Probabilistic version of no_object: 1/(n+1) where n is num of targets."""
    if all_objects is None:
        return th.tensor([0.99], device=dummy_player.device)
    
    # Identify enemies and divers: visible (index 0) and type_id in {0, 1}
    vis = all_objects[..., 0] == 1
    type_ids = all_objects[..., 6]
    is_target = vis & ((type_ids == 0) | (type_ids == 1))
    
    # Count target objects: (B,)
    n = th.sum(is_target.float(), dim=1)
    
    # Probability = 1 / (n^2 + 1)
    return (1.0 / (n**2 + 1.0)).to(th.float32)

def no_object(dummy_player, all_objects: th.Tensor = None) -> th.Tensor:
    """Probabilistic version of no_object: 1/(n+1) where n is num of targets."""
    if all_objects is None:
        return th.tensor([0.99], device=dummy_player.device)
    
    # Identify enemies and divers: visible (index 0) and type_id in {0, 1}
    vis = all_objects[..., 0] == 1
    type_ids = all_objects[..., 6]
    is_target = vis & ((type_ids == 0) | (type_ids == 1))
    
    # Count target objects: (B,)
    n = th.sum(is_target.float(), dim=1)
    
    # If there are no objects, return 1.0 use bool_to_probs
    return bool_to_probs(n == 0)


def above_water(player: th.Tensor) -> th.Tensor:
    """True if player is above water (at surface, y < 55)."""
    # Uses same threshold as surface_submarine
    vis = player[..., 0] == 1
    y = player[..., 2]
    is_surface = y < 48
    return bool_to_probs(vis & is_surface)


def below_water(player: th.Tensor) -> th.Tensor:
    """True if player is below water (at surface, y > 55)."""
    vis = player[..., 0] == 1
    y = player[..., 2]
    is_surface = y >= 48
    return bool_to_probs(vis & is_surface)

def oxygen_not_low(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar width is greater than 16 pixels (approximately 25% oxygen remaining)."""
    vis = oxygen_bar[..., 0] == 1
    oxygen_width = oxygen_bar[..., 3]  # Width in pixels (index 3)
    return th.where(vis, th.clip(oxygen_width / 16.0, 0.01, 0.99), th.tensor(0.01, device=oxygen_bar.device))

def player_left_side(player: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    # Linear decay: 1.0 at x=0, 0.5 at x=80, 0.01 at x=160
    return th.clip(1.0 - (player_x / 160.0), 0.01, 0.99)

def player_right_side(player: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    # Linear increase: 0.01 at x=0, 0.5 at x=80, 1.0 at x=160
    return th.clip(player_x / 160.0, 0.01, 0.99)

def player_down_side(player: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    player_y = th.where(player_y < 50, 50, player_y)
    player_y = th.where(player_y > 110, 110, player_y)
    # Linear decay: 1.0 at y=0, 0.5 at y=130, 0.01 at y=260
    return th.clip(1.0 - (player_y-50 / 1.0), 0.01, 0.99)

def player_up_side(player: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    player_y = th.where(player_y < 50, 50, player_y)
    player_y = th.where(player_y > 90, 90, player_y)
    # Linear increase: 0.01 at y=0, 0.5 at y=130, 1.0 at y=260
    return th.clip((player_y-50) / 110.0, 0.01, 0.99)


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
def danger_diver(diver: th.Tensor, all_objects: th.Tensor = None) -> th.Tensor:
    """True if there is an enemy in a 5 pixel radius to diver (using centers)."""
    if all_objects is None:
        return th.tensor([0.01], device=diver.device)
    
    # Identify enemies: type_id 0 and visible
    is_enemy = (all_objects[..., 6] == 0) & (all_objects[..., 0] == 1)
    
    # Compute centers
    diver_cx = diver[..., 1] + diver[..., 3] / 2.0
    diver_cy = diver[..., 2] + diver[..., 4] / 2.0
    
    obj_cx = all_objects[..., 1] + all_objects[..., 3] / 2.0
    obj_cy = all_objects[..., 2] + all_objects[..., 4] / 2.0
    
    # Euclidean distance between centers
    dx = diver_cx.unsqueeze(1) - obj_cx
    dy = diver_cy.unsqueeze(1) - obj_cy
    dist_sq = dx**2 + dy**2
    
    # Radius 5 pixels -> dist_sq < 25 (slightly more lenient than 3 to account for jitter)
    is_near_enemy = is_enemy & (dist_sq < 25)
    
    # If ANY enemy is near this diver
    any_danger = th.any(is_near_enemy, dim=1)
    
    return bool_to_probs(any_danger)

def not_danger_diver(diver: th.Tensor, all_objects: th.Tensor = None) -> th.Tensor:
    """Negation of danger_diver."""
    return 1.0 - danger_diver(diver, all_objects)
