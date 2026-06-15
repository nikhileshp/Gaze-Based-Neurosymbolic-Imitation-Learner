"""Derive the danger/noop neural-predicate FORM from data.

For each frame take the nearest car in the lane just above the chicken, with features
(|dx| horizontal offset, dy vertical gap, |speed|), and the human action (noop vs up).
Then read off where the human waits vs goes:
  - a 2-D P(noop) table over (|dx|, speed)  -> the danger region / predicate support
  - a logistic fit (action ~ |dx|, dy, speed) -> sign & scale of each feature
  - a shallow decision tree -> human-readable thresholds = the predicate
This tells us what car_dangerous(...) SHOULD compute, with data-derived constants.
"""
import numpy as np
import torch

d = torch.load("data/freeway/Freeway_logicfmt6_vel.pt", map_location="cpu", weights_only=False)
ls = np.asarray(d["logic_state"]).astype(np.float32)        # (N,12,6) [vis,is_chk,is_car,x,y,dx]
act = np.asarray(d["actions"]).reshape(-1)
P = ls[:, 0]; cars = ls[:, 1:]
px, py = P[:, 3], P[:, 4]
cx, cy, cvx = cars[:, :, 3], cars[:, :, 4], cars[:, :, 5]
vis = cars[:, :, 2] > 0.5
dy = py[:, None] - cy                                        # >0: car above chicken
above = (dy >= 9) & (dy < 23) & vis                          # lane just above
dxabs = np.abs(cx - px[:, None])

# nearest above-lane car per frame = smallest |dx|
big = np.where(above, dxabs, 1e9)
j = big.argmin(1)
has = above.any(1)
fr = np.arange(len(act))
feat_dx = dxabs[fr, j][has]
feat_dy = dy[fr, j][has]
feat_sp = np.abs(cvx[fr, j][has])
a = act[has]
keep = a != 2                                                # drop rare 'down'
X = np.stack([feat_dx[keep], feat_dy[keep], feat_sp[keep]], 1)
y = (a[keep] == 0).astype(int)                               # 1 = noop (wait), 0 = up
print(f"frames with above-lane car: {len(y)}  noop-rate={y.mean():.3f}\n")

print("P(noop) over |dx| x speed  (rows=|dx| px, cols=speed px/frame):")
dxe = [0, 6, 12, 18, 24, 40]; spe = [0, 0.3, 0.7, 1.2, 5]
hdr = "  |dx|\\spd " + "".join(f"{spe[c]:>6}-{spe[c+1]:<4}" for c in range(len(spe)-1))
print(hdr)
for r in range(len(dxe)-1):
    row=f"  {dxe[r]:>2}-{dxe[r+1]:<3} "
    for c in range(len(spe)-1):
        m=(X[:,0]>=dxe[r])&(X[:,0]<dxe[r+1])&(X[:,2]>=spe[c])&(X[:,2]<spe[c+1])
        row += f"  {y[m].mean():.2f}({int(m.sum())//1000}k)" if m.sum()>30 else "   --      "
    print(row)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    lr = LogisticRegression(max_iter=1000).fit(Xs, y)
    print("\nlogistic (standardized) coef [|dx|, dy, speed]:", np.round(lr.coef_[0], 3),
          " (neg |dx| => closer->noop; +speed => faster->noop)")
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500).fit(X, y)
    print("\ndecision tree (the predicate, readable):")
    print(export_text(dt, feature_names=["abs_dx", "dy", "speed"], show_weights=True))
except Exception as e:
    print("\n(sklearn unavailable:", e, ")")
