from nsfr.utils.common import bool_to_probs


def obj_type(z, a):
    z_type = z[:, 0:4]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
    prob = (a * z_type).sum(dim=1)
    return prob


def closeby(z_1, z_2):
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    dis_x = abs(c_1[:, 0] - c_2[:, 0]) / 171
    dis_y = abs(c_1[:, 1] - c_2[:, 1]) / 171

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

def visible(z_1, gaze):
    result = z_1[..., 0] == 1
    val = bool_to_probs(result)
    if gaze is not None and len(gaze.shape) >2:
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
    x = (obj[:, 1] * sx).long()
    y = (obj[:, 2] * sy).long()
    w = (obj[:, 3] * sx).long()
    h = (obj[:, 4] * sy).long()
    
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
    
    # Mask out invisible objects (vis <= 0.5)
    vis_mask = (obj[:, 0] > 0.5).float()
    
    return gaze_prob * vis_mask