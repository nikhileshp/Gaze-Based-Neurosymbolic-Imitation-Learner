from nsfr.utils.common import bool_to_probs
import torch as th

X_IDX = 5
Y_IDX = 6

def type(z, a):
    z_type = z[:, 1:5]
    prob = (a * z_type).sum(dim=1)
    return bool_to_probs(prob > 0.5)

def is_player(z):
    return (z[:, 1] > 0.5)

def is_present(z):
    return (z[:, 0] > 0.5)

def closest(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)

    c_1 = z_1[:, X_IDX:Y_IDX+1]
    c_2 = z_2[:, X_IDX:Y_IDX+1]

    dist = (c_1 - c_2).abs().sum(dim=1)

    inf_dist = th.full_like(dist, float('inf'))
    present_dists = th.where(obj_mask, dist, inf_dist)
    min_dist = present_dists.min()

    is_closest = player_mask & obj_mask & (dist <= min_dist + 0.5)
    return bool_to_probs(is_closest)

def notcloseby(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)

    c_1 = z_1[:, X_IDX:Y_IDX+1]
    c_2 = z_2[:, X_IDX:Y_IDX+1]

    dis_x = abs(c_1[:, 0] - c_2[:, 0]) / 171
    dis_y = abs(c_1[:, 1] - c_2[:, 1]) / 171

    result = player_mask & obj_mask & (dis_x < 1) & (dis_y <= 1)
    proximity = _close_by(z_1, z_2)
    return (1 - proximity) * bool_to_probs(result)

def closeby(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)

    c_1 = z_1[:, X_IDX:Y_IDX+1]
    c_2 = z_2[:, X_IDX:Y_IDX+1]

    dis_x = abs(c_1[:, 0] - c_2[:, 0]) / 171
    dis_y = abs(c_1[:, 1] - c_2[:, 1]) / 171

    result = player_mask & obj_mask & (dis_x < 1) & (dis_y <= 1)
    proximity = _close_by(z_1, z_2)
    return proximity * bool_to_probs(result)

def on_left(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    c_1 = z_1[:, X_IDX]
    c_2 = z_2[:, X_IDX]
    result = player_mask & obj_mask & (c_1 < c_2)
    return bool_to_probs(result)

def on_right(z_1, z_2):
    player_mask = is_player(z_1)
    c_1 = z_1[:, X_IDX]
    c_2 = z_2[:, X_IDX]
    result = player_mask & (c_1 > c_2)
    return bool_to_probs(result)

def same_row(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    c_1 = z_1[:, Y_IDX]
    c_2 = z_2[:, Y_IDX]
    diff = abs(c_2 - c_1)
    result = player_mask & obj_mask & (diff < 6)
    return bool_to_probs(result)

def above_row(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    c_1 = z_1[:, Y_IDX]
    c_2 = z_2[:, Y_IDX]
    diff = c_2 - c_1
    result = player_mask & obj_mask & (diff > 4) & (diff < 23)
    return bool_to_probs(result)

def above(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    c_1 = z_1[:, Y_IDX]
    c_2 = z_2[:, Y_IDX]
    diff = c_2 - c_1
    result = player_mask & obj_mask & (diff > 4)
    return bool_to_probs(result)

def below_row(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    c_1 = z_1[:, Y_IDX]
    c_2 = z_2[:, Y_IDX]
    diff = c_1 - c_2
    result = player_mask & obj_mask & (diff > 4) & (diff < 23)
    return bool_to_probs(result)

def below(z_1, z_2):
    player_mask = is_player(z_1)
    obj_mask = is_present(z_2)
    c_1 = z_1[:, Y_IDX]
    c_2 = z_2[:, Y_IDX]
    diff = c_1 - c_2
    result = player_mask & obj_mask & (diff > 4)
    return bool_to_probs(result)

def on_even(z_1):
    obj_mask = is_present(z_1)
    y = z_1[:, Y_IDX]
    result = obj_mask & ((y - 26) % 32 > 10)
    return bool_to_probs(result)

def on_odd(z_1):
    obj_mask = is_present(z_1)
    y = z_1[:, Y_IDX]
    result = obj_mask & ((y - 26) % 32 < 10)
    return bool_to_probs(result)

def at_top(z_1):
    player_mask = is_player(z_1)
    y = z_1[:, Y_IDX]
    result = player_mask & (y < 90)
    return bool_to_probs(result)

def at_bottom(z_1):
    player_mask = is_player(z_1)
    y = z_1[:, Y_IDX]
    result = player_mask & (y > 90)
    return bool_to_probs(result)

def at_left(z_1):
    player_mask = is_player(z_1)
    x = z_1[:, X_IDX]
    result = player_mask & (x < 60)
    return bool_to_probs(result)

def at_right(z_1):
    player_mask = is_player(z_1)
    x = z_1[:, X_IDX]
    result = player_mask & (x > 100)
    return bool_to_probs(result)

def visible(z_1, gaze=None):
    result = z_1[:, 0] > 0.5
    val = bool_to_probs(result)
    if gaze is not None and len(gaze.shape) > 2:
        gaze_val = _get_gaze_value(z_1, gaze)
        val = th.where(result, gaze_val, val)
    return val

def _get_gaze_value(obj: th.Tensor, gaze: th.Tensor, height: int = 10) -> th.Tensor:
    batch_size = obj.shape[0]
    device = obj.device

    sx = 84.0 / 160.0
    sy = 84.0 / 210.0

    x = (obj[:, X_IDX] * sx).long()
    y = (obj[:, Y_IDX] * sy).long()
    w = (th.ones_like(obj[:, X_IDX]) * 8 * sx).long()
    h = (th.ones_like(obj[:, Y_IDX]) * 11 * sy).long()

    x1 = x.clamp(0, 84)
    y1 = y.clamp(0, 84)
    x2 = (x + w).clamp(0, 84)
    y2 = (y + h).clamp(0, 84)

    area = ((x2 - x1) * (y2 - y1)).float().clamp(min=1.0)

    if gaze.shape[-1] == 85:
        integral = gaze
    else:
        gaze_padded = th.nn.functional.pad(gaze, (1, 0, 1, 0))
        integral = gaze_padded.cumsum(dim=1).cumsum(dim=2)

    b_idx = th.arange(batch_size, device=device)
    val_br = integral[b_idx, y2, x2]
    val_tl = integral[b_idx, y1, x1]
    val_tr = integral[b_idx, y1, x2]
    val_bl = integral[b_idx, y2, x1]

    total_val = val_br - val_tr - val_bl + val_tl
    avg_val = total_val / area

    uniform_density = 1.0 / (84.0 * 84.0)
    attention_ratio = avg_val / uniform_density

    gaze_prob = th.clamp(0.99 * attention_ratio, 0.5, 0.99)
    vis_mask = (obj[:, 0] > 0.5).float()
    return gaze_prob * vis_mask

def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., X_IDX]
    player_y = player[..., Y_IDX]
    obj_x = obj[..., X_IDX]
    obj_y = obj[..., Y_IDX]
    result = th.clip((128 - abs(player_x - obj_x) - abs(player_y - obj_y)) / 171, 0, 1)
    return result

def bool_to_probs(bool_tensor: th.Tensor) -> th.Tensor:
    return th.where(bool_tensor, 
                    th.tensor(0.99, device=bool_tensor.device), 
                    th.tensor(0.01, device=bool_tensor.device))