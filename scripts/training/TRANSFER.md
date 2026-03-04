# Code Transfer Reference: `train_bc.py` → `train_bc_pt.py`

This document tracks the planned migration of gaze baseline methods and features
from the original `train_bc.py` (GABRIL-style, complex) into `train_bc_pt.py`
(the canonical, simplified format).

**Strategy:** keep `train_bc_pt.py` as the single source of truth for all BC
baselines. Migrate one feature block at a time, test it, then delete the
corresponding code from `train_bc.py`. Once all features are ported,
`train_bc.py` can be removed.

---

## Status

| Feature | `train_bc_pt.py` | `train_bc.py` | Priority |
|---|---|---|---|
| Plain BC (no gaze) | ✅ done | ✅ | — |
| AGIL (dual encoder) | ✅ done | ✅ | — |
| Gaze Mask | ✅ done | ✅ | — |
| **ViSaRL** | ❌ missing | ✅ | High |
| **Reg (Regularisation)** | ❌ missing | ✅ | Medium |
| **Contrastive** | ❌ missing | ✅ | Medium |
| **GRIL** | ❌ missing | ✅ | Low |
| Frame stacking (`--stack`) | ❌ missing | ✅ | Medium |
| `--datapath` alias | ❌ missing | ✅ | Low |
| `--train_type` (confounded/normal) | ❌ missing | ✅ | Low |

---

## Migration Checklist

### Step 1 — ViSaRL (High Priority)

**What it does in `train_bc.py`:**  
ViSaRL applies a spatial soft-attention mask derived from the gaze heatmap to
the frame *before* encoding. Differs from `Mask` in that the gaze is used as a
continuous weight, not a hard binary mask.

**Where to find it:** `train_bc.py` → `train()` → `if args.gaze_method == 'ViSaRL':` block  
Inside `evaluate_bc_gabril()` and the training loop.

**Steps to port:**
1. Add `'ViSaRL'` to the `choices` list in `get_args()` in `train_bc_pt.py`.
2. Copy the ViSaRL forward-pass logic from `train_bc.py`'s training loop into
   `train_bc_pt.py`'s training loop (alongside the existing `Mask`/`AGIL` branches).
3. Add the same branch to `evaluate_bc()` in `train_bc_pt.py`.
4. Test: `python scripts/training/train_bc_pt.py --dataset <file>.pt --gaze_method ViSaRL --epochs 1`

---

### Step 2 — Reg (Gaze Regularisation) (Medium Priority)

**What it does in `train_bc.py`:**  
Adds a regularisation term to the loss that penalises the model for ignoring the
gaze-attended region. The gaze mask is used to compute a secondary supervision
signal.

**Where to find it:** `train_bc.py` → `train()` → `if args.gaze_method == 'Reg':` block

**Steps to port:**
1. Add `'Reg'` to `choices` in `get_args()`.
2. Copy the auxiliary loss computation into the training loop in `train_bc_pt.py`
   (after the main `criterion(logits, aa)` call).
3. Add the eval branch (Reg uses plain encoding at test time — same as BC).
4. Test: `... --gaze_method Reg --epochs 1`

---

### Step 3 — Contrastive (Medium Priority)

**What it does in `train_bc.py`:**  
Uses contrastive learning to encourage the encoder to produce similar
representations for frames with similar gaze patterns.

**Where to find it:** `train_bc.py` → the `ContrastiveLoss` class and its use in
the training loop.

**Steps to port:**
1. Copy the `ContrastiveLoss` class to the top of `train_bc_pt.py` (or into
   `core/utils/linear_models.py` if reusable).
2. Add `'Contrastive'` to `choices`.
3. Wire up the contrastive loss branch in the training loop.
4. Test: `... --gaze_method Contrastive --epochs 1`

---

### Step 4 — Frame Stacking (`--stack`) (Medium Priority)

**What it does in `train_bc.py`:**  
Stacks N consecutive frames into a single input tensor `(B, N, H, W)` to give
the model temporal context.

**Where to find it:** `train_bc.py` → `StackedPtDataset` / rolling buffer inside
`evaluate_bc_gabril()`.

**Steps to port:**
1. Add `--stack` argument to `get_args()` in `train_bc_pt.py`.
2. Adapt `load_pt_dataset()` call (or wrap the returned tensors) to build stacked
   frames. A helper `build_stacked_frames(obs, stack=4)` is cleanest.
3. Change `Encoder(1, ...)` to `Encoder(args.stack, ...)` when `args.stack > 1`.
4. Add the rolling buffer to `evaluate_bc()` (same pattern already present for
   the gaze predictor buffer).
5. Test: `... --stack 4 --epochs 1`

---

### Step 5 — GRIL (Low Priority)

**What it does in `train_bc.py`:**  
GRIL is a gaze-regularisation + inverse-RL hybrid. Complex training loop.

**Where to find it:** `train_bc.py` → `if args.gaze_method == 'GRIL':` block

**Steps to port:**
1. After all other methods are stable, copy the GRIL block.
2. May require additional model components — review carefully before porting.

---

### Step 6 — Cleanup

Once all methods above are ported and tested:
1. Delete `train_bc.py`.
2. Remove the `grail-train-bc` entry point from `setup.py` (or rename it to point
   to `train_bc_pt:main`).
3. Update `scripts/README.md` to remove the `train_bc.py` row.

---

## Key Differences to Keep in Mind

| Item | `train_bc.py` style | `train_bc_pt.py` style |
|---|---|---|
| Dataset arg | `--datapath` | `--dataset` |
| Data loader | `PtDataset` / `ExpertDataset` | `load_pt_dataset()` |
| Env name casing | `"Seaquest"` (capital) | `"seaquest"` (lowercase) |
| Eval function | `evaluate_bc_gabril()` | `evaluate_bc()` |
| Imports | From `core/utils/data_utils` | From `core/utils/utils` |

Always keep `train_bc_pt.py`'s conventions when porting.
