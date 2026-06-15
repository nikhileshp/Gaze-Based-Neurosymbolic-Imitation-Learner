"""Where does conditional_xclose disagree with the human? Bucket the errors by scene
to decide which rules/predicates would close the gap to BC.

Run: python scripts/diagnostics/error_analysis_freeway.py
"""
import sys

import numpy as np
import torch

from core.nsfr.agents.imitation_agent import ImitationAgent

DATASET = "data/freeway/Freeway_logicfmt5_obj12.pt"
N = 30000
RULES = sys.argv[1] if len(sys.argv) > 1 else "conditional_xclose"


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(DATASET, map_location="cpu", weights_only=False)
    ls = torch.as_tensor(np.asarray(d["logic_state"])).float()      # (N,12,5)
    act = torch.as_tensor(np.asarray(d["actions"])).long().reshape(-1)
    n = min(N, ls.size(0))
    ls, act = ls[:n], act[:n]

    agent = ImitationAgent("freeway", RULES, dev, action_head="max")
    preds = []
    with torch.no_grad():
        for i in range(0, n, 4096):
            _, sc = agent.predict(ls[i:i+4096].to(dev), None)
            preds.append(sc.argmax(1).cpu())
    pred = torch.cat(preds)

    # scene geometry
    P = ls[:, 0, :]; cars = ls[:, 1:, :]
    px, py = P[:, 3], P[:, 4]
    cx, cy = cars[:, :, 3], cars[:, :, 4]; vis = cars[:, :, 0] > 0.5
    dy = py[:, None] - cy; dxabs = (cx - px[:, None]).abs()
    xclose = dxabs < 12
    above1 = (dy >= 9) & (dy < 23) & vis            # lane immediately above
    above2 = (dy >= 23) & (dy < 40) & vis           # two lanes above
    samerow = (dy.abs() < 9) & vis                  # player's row
    below1 = (dy <= -9) & (dy > -23) & vis          # lane below
    top = cy < 100                                   # moving left (else right)
    car_right = cx > px[:, None]; car_left = cx < px[:, None]
    approaching = (top & car_right) | (~top & car_left)
    fast = (cy.sub(91).abs() < 5) | (cy.sub(107).abs() < 5)   # the two fast lanes

    def any_(m): return m.any(1)

    print(f"samples={n}  acc={ (pred==act).float().mean():.4f}")
    print("confusion (rows=human, cols=pred)  [noop,up,down]:")
    for h in range(3):
        row = [int(((act==h)&(pred==p)).sum()) for p in range(3)]
        print(f"  human={h}: {row}")

    miss = (act == 0) & (pred == 1)     # human noop, model went UP (missed dodge)
    fstop = (act == 1) & (pred == 0)    # human up, model NOOP'd (false stop)
    print(f"\nMISSED DODGE (human=noop, pred=up): {int(miss.sum())} frames "
          f"({int(miss.sum())/max(int((act==0).sum()),1)*100:.1f}% of noop)")
    print(f"  has car in lane above (any) : {any_(above1)[miss].float().mean():.3f}")
    print(f"  has x-close car lane above  : {any_(above1&xclose)[miss].float().mean():.3f}")
    print(f"  has x-close car TWO lanes up: {any_(above2&xclose)[miss].float().mean():.3f}  <- two-lane lookahead")
    print(f"  has x-close car SAME row    : {any_(samerow&xclose)[miss].float().mean():.3f}  <- same-row threat")
    print(f"  has x-close car lane BELOW  : {any_(below1&xclose)[miss].float().mean():.3f}  <- car catching up")
    print(f"  involves a FAST-lane x-close car (above1/2): {any_((above1|above2)&xclose&fast)[miss].float().mean():.3f}")
    print(f"  NO x-close car in above1/above2/samerow/below1: "
          f"{(~any_(((above1|above2|samerow|below1)&xclose)))[miss].float().mean():.3f}  <- rules truly blind")

    print(f"\nFALSE STOP (human=up, pred=noop): {int(fstop.sum())} frames "
          f"({int(fstop.sum())/max(int((act==1).sum()),1)*100:.1f}% of up)")
    fired = above1 & xclose
    print(f"  the x-close above car is RECEDING: {(any_(fired & (~approaching)) & ~any_(fired & approaching))[fstop].float().mean():.3f}")
    print(f"  the x-close above car is in a FAST lane: {any_(fired & fast)[fstop].float().mean():.3f}")


if __name__ == "__main__":
    main()
