"""Augment the freeway logic-state dataset with per-object horizontal velocity (dx).

Atari Freeway is higher-order Markov: a single frame can't reveal car speed. We add a
6th feature dx = horizontal displacement of each car vs the previous frame (within the
same episode), matching cars by nearest-x in the same lane and correcting for screen
wrap-around. The chicken (obj0) gets dx=0. This lets velocity-based neural predicates
work even when car speeds change at test time (HackAtari), instead of relying on the
fixed lane->speed mapping.

In:  data/freeway/Freeway_logicfmt5_obj12.pt   logic_state (N,12,5) [vis,is_chk,is_car,x,y]
Out: data/freeway/Freeway_logicfmt6_vel.pt     logic_state (N,12,6) [...,x,y,dx]
"""
import numpy as np
import torch

SRC = "data/freeway/Freeway_logicfmt5_obj12.pt"
DST = "data/freeway/Freeway_logicfmt6_vel.pt"
SCREEN_W = 160
LANE_TOL = 5      # px: same slot only if it stayed in the same lane
WRAP = SCREEN_W
K = 8             # velocity window (frames): cars move sub-pixel/frame, so diff over K


def main() -> None:
    d = torch.load(SRC, map_location="cpu", weights_only=False)
    ls = np.asarray(d["logic_state"]).astype(np.float32)          # (N,12,5)
    ep = np.asarray(d["episode_number"]).reshape(-1)
    N, O, F = ls.shape
    out = np.concatenate([ls, np.zeros((N, O, 1), np.float32)], axis=2)  # (N,12,6), dx=0

    # Velocity = same-slot displacement over the last K frames / K (slots are stable in
    # this dataset). Same lane only (guards slot reuse), wrap-around corrected.
    for t in range(N):
        tk = t - K
        if tk < 0 or ep[t] != ep[tk]:
            continue
        for o in range(1, O):                                     # cars only (obj0 chicken: dx=0)
            if ls[t, o, 0] > 0.5 and ls[tk, o, 0] > 0.5 and ls[t, o, 2] > 0.5 \
               and abs(ls[t, o, 4] - ls[tk, o, 4]) < LANE_TOL:
                dx = ls[t, o, 3] - ls[tk, o, 3]
                if dx > WRAP / 2:  dx -= WRAP
                if dx < -WRAP / 2: dx += WRAP
                out[t, o, 5] = dx / K                              # px per frame

    d["logic_state"] = out
    torch.save(d, DST)
    # sanity
    dxv = out[:, 1:, 5]
    vis = out[:, 1:, 2] > 0.5
    print(f"wrote {DST}  shape={out.shape}")
    print(f"car dx: mean|dx|={np.abs(dxv[vis]).mean():.2f}  max|dx|={np.abs(dxv[vis]).max():.1f}  "
          f"frac nonzero={np.mean(dxv[vis]!=0):.2f}")
    # per-lane mean dx (should recover the known fixed speeds + signs)
    cy = out[:, 1:, 4]
    for lane in [27, 43, 59, 75, 91, 107, 123, 139, 155, 171]:
        m = vis & (np.abs(cy - lane) < 5) & (dxv != 0)
        if m.sum() > 50:
            print(f"  lane y={lane:3d}: mean dx={dxv[m].mean():+.2f}  (n={int(m.sum())})")


if __name__ == "__main__":
    main()
