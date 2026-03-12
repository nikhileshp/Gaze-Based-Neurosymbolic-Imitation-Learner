import torch
from nsfr.utils.common import bool_to_probs


def _get_x_y(z: torch.Tensor):
    # Determine feature count to handle both 5 and 6 feature datasets
    num_feat = z.shape[-1]
    if num_feat == 6:
        # Layout: [?, chicken, car, ?, x, y]
        return z[..., 4], z[..., 5]
    else:
        # Layout: [vis, chicken, car, x, y]
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


def closeby(z_1: torch.Tensor, z_2: torch.Tensor) -> torch.Tensor:
    x1, y1 = _get_x_y(z_1)
    x2, y2 = _get_x_y(z_2)

    dis_x = abs(x1 - x2) / 171
    dis_y = abs(y1 - y2) / 171

    # Continuous Gaussian decay
    # sigma_y=0.05 targets ~8-10 pixels (half lane)
    # sigma_x=0.5 targets ~85 pixels (half screen)
    prob = torch.exp(- (dis_x**2 / (2 * 0.5**2)) - (dis_y**2 / (2 * 0.05**2)))

    return prob


def on_left(z_1: torch.Tensor, z_2: torch.Tensor):
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    diff = x2 - x1
    result = bool_to_probs(diff > 0)
    return result


def on_right(z_1: torch.Tensor, z_2: torch.Tensor):
    x1, _ = _get_x_y(z_1)
    x2, _ = _get_x_y(z_2)
    diff = x2 - x1
    result = bool_to_probs(diff < 0)
    return result


def same_row(z_1: torch.Tensor, z_2: torch.Tensor):
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    diff = abs(y2 - y1)
    result = bool_to_probs(diff < 6)
    return result


def above_row(z_1: torch.Tensor, z_2: torch.Tensor):
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    
    diff = y2 - y1
    result1 = bool_to_probs(diff < 23)
    result2 = bool_to_probs(diff > 4)
    return result1 * result2

def above(z_1: torch.Tensor, z_2: torch.Tensor):
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    diff = y2 - y1
    # z_1 is "above" z_2 if z_1 has smaller Y
    # Range looking for the lane immediately above
    result = bool_to_probs((diff < 23) & (diff > 4))
    return result

def below(z_1: torch.Tensor, z_2: torch.Tensor):
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    diff = y2 - y1
    # z_1 is "below" z_2 if z_1 has larger Y
    # Range looking for the lane immediately below
    result = bool_to_probs((diff < 4) & (diff > -23))
    return result

def top5car(z_1: torch.Tensor):
    _, y = _get_x_y(z_1)
    # y > 100 corresponds to bottom half in Atari (0 is top)
    result = bool_to_probs(y > 100)
    return result


def bottom5car(z_1: torch.Tensor):
    _, y = _get_x_y(z_1)
    # y < 100 corresponds to top half
    result = bool_to_probs(y < 100)
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
