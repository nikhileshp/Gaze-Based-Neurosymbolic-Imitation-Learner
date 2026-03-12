from nsfr.utils.common import bool_to_probs
<<<<<<< HEAD


def obj_type(z, a):
    z_type = z[:, 0:4]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
    prob = (a * z_type).sum(dim=1)
    return prob


def closeby(z_1, z_2):
=======
import torch as th

def type(z, a):
    z_type = z[:, 0:4]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
    prob = (a * z_type).sum(dim=1)
    return bool_to_probs(prob > 0.5)


def is_player(z):
    return (z[:, 0] > 0.5)

def is_present(z):
    return (z[:, 0:4].sum(dim=1) > 0.5)

def closest(z_1, z_2):
    """Returns bool_to_probs(True) iff z_2 is the closest present object to z_1 (player).

    Computes L1 distance between the player (z_1) and the candidate object (z_2).
    Finds the global minimum distance among all present objects in the batch,
    then returns True only for the pair whose distance equals that minimum.
    """
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)

    c_1 = z_1[:, -2:]   # player coords  (batch, 2)
    c_2 = z_2[:, -2:]   # candidate coords (batch, 2)

    dist = (c_1 - c_2).abs().sum(dim=1)   # L1 distance per pair

    # Replace non-present objects with inf so they don't win
    inf_dist = th.full_like(dist, float('inf'))
    present_dists = th.where(obj_mask, dist, inf_dist)
    min_dist = present_dists.min()

    # True iff this pair is the (unique or tied) nearest present object
    is_closest = player_mask & obj_mask & (dist <= min_dist + 0.5)
    return bool_to_probs(is_closest)

def notcloseby(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    
>>>>>>> a84adb97b4d89759b1f0ff6ec4808e0af9298681
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    dis_x = abs(c_1[:, 0] - c_2[:, 0]) / 171
    dis_y = abs(c_1[:, 1] - c_2[:, 1]) / 171

<<<<<<< HEAD
    result = bool_to_probs((dis_x < 2.5) & (dis_y <= 0.1))

    return result


def on_left(z_1, z_2):
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    diff = c_2 - c_1
    result = bool_to_probs(diff > 0)
    return result


def on_right(z_1, z_2):
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    diff = c_2 - c_1
    result = bool_to_probs(diff < 0)
    return result


def same_row(z_1, z_2):
    c_1 = z_1[:, -1]
    c_2 = z_2[:, -1]
    diff = abs(c_2 - c_1)
    result = bool_to_probs(diff < 6)
    return result


def above_row(z_1, z_2):
    c_1 = z_1[:, -1]
    c_2 = z_2[:, -1]
    diff = c_1 - c_2
    result1 = bool_to_probs(diff < 23)
    result2 = bool_to_probs(diff > 4)
    return result1 * result2


def below_row(z_1, z_2):
    c_1 = z_1[:, -1]
    c_2 = z_2[:, -1]
    diff = c_2 - c_1
    result1 = bool_to_probs(diff < 23)
    result2 = bool_to_probs(diff > 4)
    return result1 * result2


def on_even(z_1):
    y = z_1[:, -1]
    result = bool_to_probs((y - 26) % 32 > 10)
    return result


def on_odd(z_1):
    y = z_1[:, -1]
    result = bool_to_probs((y - 26) % 32 < 10)
    return result


def at_top(z_1):
    y = z_1[:, -1]
    result = bool_to_probs(y > 87)
    return result


def at_bottom(z_1):
    y = z_1[:, -1]
    result = bool_to_probs(y < 87)
    return result


def at_left(z_1):
    x = z_1[:, -2]
    result = bool_to_probs(x < 80)
    return result


def at_right(z_1):
    x = z_1[:, -2]
    result = bool_to_probs(x > 80)
    return result
=======
    result = player_mask & obj_mask & (dis_x < 1) & (dis_y <= 1) # Adjusted threshold
    proximity = _close_by(z_1, z_2)
    return (1-proximity) * bool_to_probs(result)

def closeby(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    dis_x = abs(c_1[:, 0] - c_2[:, 0]) / 171
    dis_y = abs(c_1[:, 1] - c_2[:, 1]) / 171

    result = player_mask & obj_mask & (dis_x < 1) & (dis_y <= 1) # Adjusted threshold
    proximity = _close_by(z_1, z_2)
    return proximity * bool_to_probs(result)


def on_left(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    result = player_mask & obj_mask & (c_1 < c_2)
    return bool_to_probs(result)


def on_right(z_1, z_2):
    player_mask = is_player(z_1)
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    result = player_mask & (c_1 > c_2)
    return bool_to_probs(result)


def same_row(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    c_1 = z_1[:, -1]
    c_2 = z_2[:, -1]
    diff = abs(c_2 - c_1)
    result = player_mask & obj_mask & (diff < 6)
    return bool_to_probs(result)


def above_row(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    c_1 = z_1[:, -1]
    c_2 = z_2[:, -1]
    diff = c_2 - c_1 # Player above Object -> Obj.y > Player.y
    result = player_mask & obj_mask & (diff > 4) & (diff < 23)
    return bool_to_probs(result)


def below_row(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    c_1 = z_1[:, -1]
    c_2 = z_2[:, -1]
    diff = c_1 - c_2 # Player below Object -> Player.y > Obj.y
    result = player_mask & obj_mask & (diff > 4) & (diff < 23)
    return bool_to_probs(result)


def on_even(z_1):
    obj_mask = is_present(z_1)
    y = z_1[:, -1]
    result = obj_mask & ((y - 26) % 32 > 10)
    return bool_to_probs(result)


def on_odd(z_1):
    obj_mask = is_present(z_1)
    y = z_1[:, -1]
    result = obj_mask & ((y - 26) % 32 < 10)
    return bool_to_probs(result)


def at_top(z_1):
    player_mask = is_player(z_1)
    y = z_1[:, -1]
    result = player_mask & (y < 90)
    return bool_to_probs(result)


def at_bottom(z_1):
    player_mask = is_player(z_1)
    y = z_1[:, -1]
    result = player_mask & (y > 90)
    return bool_to_probs(result)


def at_left(z_1):
    player_mask = is_player(z_1)
    x = z_1[:, -2]
    result = player_mask & (x < 60)
    return bool_to_probs(result)
 
 
def at_right(z_1):
    player_mask = is_player(z_1)
    x = z_1[:, -2]
    result = player_mask & (x > 100)
    return bool_to_probs(result)


def visible(z_1, gaze=None):
    result = z_1[:, 0:4].sum(dim=1) > 0.5
    val = bool_to_probs(result)
    if gaze is not None and len(gaze.shape) > 2:
        gaze_val = _get_gaze_value(z_1, gaze)
        val = th.where(result, gaze_val, val)
    return val

def _get_gaze_value(obj: th.Tensor, gaze: th.Tensor, height: int = 10) -> th.Tensor:
    """
    Calculate average gaze intensity within the object's bounding box.
    Vectorized implementation using integral images for speed.
    obj: (batch, features) [vis, x, y, w, ...]
    gaze: (batch, 84, 84)
    height: approximate height of object (since obj might not have it)
    """
    batch_size = obj.shape[0]
    device = obj.device

    # Scaling factors (160x210 -> 84x84)
    sx = 84.0 / 160.0
    sy = 84.0 / 210.0
    
    # Coordinates (Vectorized)
    # Asterix features: [P, E, B, R, ?, X, Y] -> indices 5, 6
    x = (obj[:, 5] * sx).long()
    y = (obj[:, 6] * sy).long()
    w = (th.ones_like(obj[:, 5]) * 8 * sx).long() # Default width 8
    h = (th.ones_like(obj[:, 6]) * 11 * sy).long() # Default height 11
    
    # Clip coordinates to valid range [0, 84]
    # We use 0-84 because for integral image, index 84 corresponds to sum of all 0-83
    x1 = x.clamp(0, 84)
    y1 = y.clamp(0, 84)
    x2 = (x + w).clamp(0, 84)
    y2 = (y + h).clamp(0, 84)
    
    # Calculate area (clamp min=1 to avoid division by zero)
    area = ((x2 - x1) * (y2 - y1)).float().clamp(min=1.0)
    
    # Compute Integral Image (Summed Area Table)
    # Pad left and top with 0 for easy indexing (0,0 corresponds to sum=0)
    # Result shape: (batch, 85, 85)
    # gaze is (batch, 84, 84)
    
    # Compute Integral Image (Summed Area Table)
    # If gaze is already integral (85x85), use it. Else compute.
    if gaze.shape[-1] == 85:
        integral = gaze
    else:
        gaze_padded = th.nn.functional.pad(gaze, (1, 0, 1, 0)) # Pad left and top
        integral = gaze_padded.cumsum(dim=1).cumsum(dim=2)
    
    # Gather values at corners using batch indices
    # We need to index (b, y, x)
    b_idx = th.arange(batch_size, device=device)
    
    # x1, y1, x2, y2 are definitely in [0, 84] range, valid for indexing [0, 85] size
    
    # Bottom-Right (y2, x2)
    val_br = integral[b_idx, y2, x2]
    # Top-Left (y1, x1)
    val_tl = integral[b_idx, y1, x1]
    # Top-Right (y1, x2)
    val_tr = integral[b_idx, y1, x2]
    # Bottom-Left (y2, x1)
    val_bl = integral[b_idx, y2, x1]
    
    total_val = val_br - val_tr - val_bl + val_tl
    
    avg_val = total_val / area
    
    # The heatmap is softmax-normalized: all 84*84=7056 pixels sum to 1.0.
    # So uniform density = 1/7056 per pixel. Raw avg_val is always ~0.0001,
    # which would zero out every visible object if used directly.
    #
    # Solution: compute an attention_ratio = (object density) / (uniform density).
    # - ratio == 1  ->  object gets exactly its fair share of gaze  ->  keep ~0.99
    # - ratio >> 1  ->  object is actively gazed at                 ->  keep 0.99
    # - ratio << 1  ->  object is not being looked at               ->  suppress
    uniform_density = 1.0 / (84.0 * 84.0)  # ~0.000142
    attention_ratio = avg_val / uniform_density  # dimensionless, ~1.0 for uniform gaze
    
    # Scale to [0.01, 0.99] probability range
    gaze_prob = th.clamp(0.99 * attention_ratio, 0.5, 0.99)
    
    # Mask out invisible objects (any type bit 0-3 <= 0.5)
    vis_mask = (obj[:, 0:4].sum(dim=1) > 0.5).float()
    
    return gaze_prob * vis_mask

def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    # Feature layout: [player_flag, enemy_flag, bonus_flag, reward_flag, ?, x, y]
    # Indices -2 and -1 are x and y (equivalently, indices 5 and 6 for n_features=7)
    player_x = player[..., -2]
    player_y = player[..., -1]
    obj_x = obj[..., -2]
    obj_y = obj[..., -1]
    result = th.clip((128 - abs(player_x - obj_x) - abs(player_y - obj_y)) / 171, 0, 1)
    return result
>>>>>>> a84adb97b4d89759b1f0ff6ec4808e0af9298681
