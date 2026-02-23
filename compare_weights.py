import torch
import numpy as np

no_gaze = torch.load('out/imitation/seaquest_new_il_epoch_10_lr_0.1_no_gaze.pth', map_location='cpu')
with_gazemap = torch.load('out/imitation/seaquest_new_il_epoch_10_lr_0.1_with_gazemap_values.pth', map_location='cpu')

ng = no_gaze['im.W'].float()
wg = with_gazemap['im.W'].float()

ng_soft = torch.softmax(ng, dim=1)
wg_soft = torch.softmax(wg, dim=1)

print('im.W shape:', tuple(ng.shape), '  (num_rules x num_clauses)')
print()
print('Rule  | no_gaze best_clause (prob) | gazemap best_clause (prob) | Same? | max_delta')
print('-' * 80)
for i in range(ng.shape[0]):
    ng_am = ng_soft[i].argmax().item()
    wg_am = wg_soft[i].argmax().item()
    diff = (ng_soft[i] - wg_soft[i]).abs().max().item()
    same = ng_am == wg_am
    print(f'  {i:2d}  | clause {ng_am:3d} ({ng_soft[i,ng_am]:.4f})         | clause {wg_am:3d} ({wg_soft[i,wg_am]:.4f})         | {"YES" if same else "NO ":3}   | {diff:.5f}')

print()
print('=== Raw im.W per row ===')
print('Row | no_gaze raw W                           | gazemap raw W')
for i in range(ng.shape[0]):
    top3_ng = ng[i].topk(3)
    top3_wg = wg[i].topk(3)
    print(f'  {i:2d} | top3 idx={top3_ng.indices.tolist()} vals={[round(v,2) for v in top3_ng.values.tolist()]}  | top3 idx={top3_wg.indices.tolist()} vals={[round(v,2) for v in top3_wg.values.tolist()]}')
