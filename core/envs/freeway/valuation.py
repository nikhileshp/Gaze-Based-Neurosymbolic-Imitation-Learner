import torch
from nsfr.utils.common import bool_to_probs


def type(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    z_type = z[:, 1:3]  # Shifted from 0:2 to accommodate visibility at index 0
    prob = (a * z_type).sum(dim=1)
    return prob


def visible(z: torch.Tensor) -> torch.Tensor:
    return bool_to_probs(z[..., 0] == 1)


def closeby(z_1: torch.Tensor, z_2: torch.Tensor) -> torch.Tensor:
    c_1 = z_1[:, 3:5] # Shifted from -2:
    c_2 = z_2[:, 3:5]

    dis_x = abs(c_1[:, 0] - c_2[:, 0]) / 171
    dis_y = abs(c_1[:, 1] - c_2[:, 1]) / 171

    result = bool_to_probs((dis_x < 2.5) & (dis_y <= 0.1))

    return result


def on_left(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, 3] # Shifted from -2
    c_2 = z_2[:, 3]
    diff = c_2 - c_1
    result = bool_to_probs(diff > 0)
    return result


def on_right(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, 3]
    c_2 = z_2[:, 3]
    diff = c_2 - c_1
    result = bool_to_probs(diff < 0)
    return result


def same_row(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, 4] # Shifted from -1
    c_2 = z_2[:, 4]
    diff = abs(c_2 - c_1)
    result = bool_to_probs(diff < 6)
    return result


def above_row(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, 4]
    c_2 = z_2[:, 4]
    
    diff = c_2 - c_1
    result1 = bool_to_probs(diff < 23)
    result2 = bool_to_probs(diff > 4)
    return result1 * result2

def above(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, 4]
    c_2 = z_2[:, 4]
    diff = c_2 - c_1
    # z_1 is "above" z_2 if z_1 has smaller Y
    # Range looking for the lane immediately above
    result = bool_to_probs((diff < 23) & (diff > 4))
    return result

def below(z_1: torch.Tensor, z_2: torch.Tensor):
    c_1 = z_1[:, 4]
    c_2 = z_2[:, 4]
    diff = c_2 - c_1
    # z_1 is "below" z_2 if z_1 has larger Y
    # Range looking for the lane immediately below
    result = bool_to_probs((diff < 4) & (diff > -23))
    return result

def top5car(z_1: torch.Tensor):
    y = z_1[:, 4]
    result = bool_to_probs(y > 100)
    return result


def bottom5car(z_1: torch.Tensor):
    y = z_1[:, 4]
    result = bool_to_probs(y < 100)
    return result

def topfastcar(z_1: torch.Tensor):
    # Lane Y=107 is fast (following top5car y > 100 convention)
    y = z_1[:, 4]
    result = bool_to_probs(abs(y - 107) < 5)
    return result

def bottomfastcar(z_1: torch.Tensor):
    # Lane Y=91 is fast (following bottom5car y < 100 convention)
    y = z_1[:, 4]
    result = bool_to_probs(abs(y - 91) < 5)
    return result
