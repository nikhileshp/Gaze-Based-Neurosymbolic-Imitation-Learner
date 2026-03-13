import torch
from nsfr.utils.common import bool_to_probs


def _get_x_y(z: torch.Tensor):


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


def same_row(z_1: torch.Tensor, z_2: torch.Tensor):
    # Overlap logic for collision detection (height ~9 pixels)
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    diff = abs(y1 - y2)
    # Requirement: z1 is a Car (index 2) and z2 is the Player
    result = (diff < 9) & (z_1[..., 2] > 0.5) & _is_player(z_2)
    return bool_to_probs(result)


def above_row(z_1: torch.Tensor, z_2: torch.Tensor):
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    
    diff = y2 - y1
    # z_1 is "above" z_2 if z_1 has smaller Y
    # Excluding same lane (diff < 9)
    result = ((diff < 23) & (diff >= 9)) & _is_player(z_2)
    return bool_to_probs(result)


def below_row(z_1: torch.Tensor, z_2: torch.Tensor):
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    
    diff = y2 - y1
    # z_1 is "below" z_2 if z_1 has larger Y
    # Excluding same lane (diff > -9)
    result = ((diff <= -9) & (diff > -23)) & _is_player(z_2)
    return bool_to_probs(result)


def above(z_1: torch.Tensor, z_2: torch.Tensor):
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    diff = y2 - y1
    # Anywhere above z_2 (excluding same lane)
    result = (diff >= 9) & _is_player(z_2)
    return bool_to_probs(result)


def below(z_1: torch.Tensor, z_2: torch.Tensor):
    _, y1 = _get_x_y(z_1)
    _, y2 = _get_x_y(z_2)
    diff = y2 - y1
    # Anywhere below z_2 (excluding same lane)
    result = (diff <= -9) & _is_player(z_2)
    return bool_to_probs(result)

def top5car(z_1: torch.Tensor):
    _, y = _get_x_y(z_1)
    # y < 100 corresponds to top half
    # Only true for cars
    result = (y < 100) & (z_1[..., 2] > 0.5)
    return bool_to_probs(result)


def bottom5car(z_1: torch.Tensor):
    _, y = _get_x_y(z_1)
    # y > 100 corresponds to bottom half in Atari (0 is top)
    # Only true for cars
    result = (y > 100) & (z_1[..., 2] > 0.5)
    return bool_to_probs(result)

def topfastcar(z_1: torch.Tensor):
    _, y = _get_x_y(z_1)
    # Lane Y=107 is fast (following top5car y > 100 convention)
    # Only true for cars
    result = (abs(y - 107) < 5) & (z_1[..., 2] > 0.5)
    return bool_to_probs(result)

def bottomfastcar(z_1: torch.Tensor):
    _, y = _get_x_y(z_1)
    # Lane Y=91 is fast (following bottom5car y < 100 convention)
    # Only true for cars
    result = (abs(y - 91) < 5) & (z_1[..., 2] > 0.5)
    return bool_to_probs(result)
