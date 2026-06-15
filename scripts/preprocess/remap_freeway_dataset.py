#!/usr/bin/env python
"""Remap a legacy Freeway dataset .pt to the current env logic-state format.

The old converter wrote Freeway logic states as 6 features per object:
    [is_chicken, is_car, 0, 0, x, y]
but the current env (`core/envs/freeway/env.py`) and the valuation functions
expect 5 features:
    [visible, is_chicken, is_car, x, y]            (visible = is_chicken OR is_car)

Objects are kept as (player chicken at obj0, cars at obj1..) — the legacy data
already drops the static decoy chicken, so only a feature reorder is needed. The
object dimension is padded/truncated to 12 slots to match the env. All other keys
(observations, gaze_image, actions, episode_number, ...) are passed through.

Usage:
    python scripts/preprocess/remap_freeway_dataset.py \
        --input  data/freeway/Freeway_full_data_6_episodes_10p0_sigma_win_10_obj_12.pt \
        --output data/freeway/Freeway_logicfmt5_obj12.pt
"""
import argparse

import numpy as np
import torch

N_OBJECTS = 12


def remap_logic_state(ls: np.ndarray) -> np.ndarray:
    """(N, O, 6) legacy layout -> (N, 12, 5) current env layout."""
    if ls.ndim != 3:
        raise ValueError(f"logic_state must be 3D (N, O, F); got shape {ls.shape}")
    n, o, f = ls.shape
    if f == 5:
        raise ValueError("logic_state already has 5 features; nothing to remap.")
    if f != 6:
        raise ValueError(f"expected 6 features in legacy layout, got {f}")

    is_chicken = ls[..., 0]
    is_car = ls[..., 1]
    x = ls[..., 4]
    y = ls[..., 5]
    visible = ((is_chicken + is_car) > 0.5).astype(ls.dtype)

    out = np.stack([visible, is_chicken, is_car, x, y], axis=-1)  # (N, O, 5)

    if o < N_OBJECTS:
        pad = np.zeros((n, N_OBJECTS - o, 5), dtype=out.dtype)
        out = np.concatenate([out, pad], axis=1)
    elif o > N_OBJECTS:
        out = out[:, :N_OBJECTS]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="legacy Freeway .pt dataset")
    ap.add_argument("--output", required=True, help="path for the remapped .pt")
    args = ap.parse_args()

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    ls = np.asarray(data["logic_state"])
    print(f"input  logic_state: {ls.shape}  (legacy [is_chicken,is_car,0,0,x,y])")

    data["logic_state"] = remap_logic_state(ls)
    print(f"output logic_state: {data['logic_state'].shape}  (env [vis,is_chicken,is_car,x,y])")

    torch.save(data, args.output)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
