# NSFR Imitation Learning — Optimization Notes

## Overview

This document summarizes a full performance investigation and optimization session for the NSFR-based imitation learning training pipeline (`train_il.py`, `imitation_agent.py`, `infer.py`) for the Seaquest environment.

**Starting point:** ~1.96 it/s, ~1 hour per epoch  
**End result:** ~1.76 it/s at batch=512 across 4 GPUs = ~902 samples/sec, ~3.7 min per epoch  
**Total speedup: ~13.5×**

---

## Step 1 — Profiling First

Before making any changes, a diagnostic profiler was added to measure where time was actually going. The diagnostic breaks the training loop into three independently timed phases: data loading, forward pass, and backward pass.

**Key finding from first diagnostic (batch=32, single GPU):**
```
Data loading:   25.3 ms
Forward pass:   315.1 ms
Backward pass:  176.3 ms
Total:          516.6 ms  →  1.94 it/s
GPU reserved:   11.58 GB
```

Data loading was negligible. The bottleneck was model compute, with GPU memory usage far exceeding what the model weights required — indicating massive intermediate tensor allocations.

---

## Step 2 — Double Forward Pass Eliminated (`imitation_agent.py`)

**Problem:** The training loop called `agent.update()` for the loss (forward pass #1), then immediately called `agent.model()` again inside `torch.no_grad()` to compute training accuracy (forward pass #2). This was 2× the necessary compute per batch.

**Fix:** `update()` was refactored to return `(loss, probs, action_scores)` so the caller can reuse the already-computed tensors for accuracy without a second forward pass.

A new `predict()` method was added for use in validation loops — wraps `torch.no_grad()` and returns `(probs, action_scores)` without computing loss.

A shared `_aggregate_to_action_scores()` helper was extracted to avoid duplicating the predname-to-action aggregation loop across `update()`, `predict()`, and `act()`.

**Result:** Minor improvement in isolation, but eliminated redundant computation and cleaned up architecture significantly.

---

## Step 3 — `ClauseFunction.forward()` Memory Fix (`infer.py`)

**Problem:** The original `ClauseFunction.forward()` used `.repeat()` twice to expand tensors before `torch.gather`:

```python
# ORIGINAL — allocates (B, G, S, L) THREE times
V_tild   = V.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.S, self.L)  # allocation
I_i_tild = I_i.repeat(batch_size, 1, 1, 1)                             # allocation
C = softor(torch.prod(torch.gather(V_tild, 1, I_i_tild), 3), ...)      # allocation
```

`.repeat()` physically copies memory. At `(B=32, G=284, S=625, L=6)`, this allocated ~11GB of intermediate tensors per batch.

**Fix:** Replace `.repeat()` with `.expand()`, which creates a zero-copy view with no memory allocation:

```python
# FIXED — zero-copy views, gather backward remains fast scatter_add
V_tild   = V.unsqueeze(-1).unsqueeze(-1).expand(batch_size, V.size(1), self.S, self.L)
I_i_tild = I_i.unsqueeze(0).expand(batch_size, -1, -1, -1)
C = softor(torch.prod(torch.gather(V_tild, 1, I_i_tild), dim=3), dim=2, gamma=self.gamma)
```

`torch.gather` was kept (not replaced with advanced indexing `V[:, I_i]`) because gather's backward uses `scatter_add` which is fast. Advanced indexing backward was tested and found to be ~10× slower (1965ms vs 158ms).

The same fix was applied to `ClauseBodySumFunction.forward()` which had the identical pattern.

**Result:**
```
Before: GPU reserved 11.58 GB, backward 176ms
After:  GPU reserved 4.14 GB,  backward 158ms
```

---

## Step 4 — `InferModule.r()` einsum Fix (`infer.py`)

**Problem:** The weighted sum over clause results was materializing a full `(m, C, B, G)` intermediate tensor:

```python
# ORIGINAL — allocates (m, C, B, G)
W_tild = W_star.unsqueeze(-1).unsqueeze(-1).expand(m, C, B, G)
C_tild = C.unsqueeze(0).expand(m, C, B, G)
H = torch.sum(W_tild * C_tild, dim=1)
```

**Fix:** Replace with `einsum` which contracts the C dimension directly without materializing the 4D intermediate:

```python
# FIXED — (m, C, B, G) never exists in memory
W_star = torch.softmax(self.W, dim=1)           # (m, C)
H = torch.einsum('mc,cbg->mbg', W_star, C_stack) # (m, B, G)
```

---

## Step 5 — Optimal Batch Size

With memory pressure reduced, batch size was profiled systematically:

| Batch | Samples/sec |
|-------|-------------|
| 32    | 67/s        |
| 64    | 88/s        |
| 128   | 107/s       |
| 256   | OOM         |

**Batch=128 was the single-GPU optimum** at 107 samples/sec (~31 min/epoch).

The scaling confirmed sub-linear time growth with batch size, meaning larger batches were more efficient despite higher per-batch cost.

---

## Step 6 — Multi-GPU with DataParallel (`train_il.py`, `infer.py`, `imitation_agent.py`)

**Approach:** `nn.DataParallel` splits the batch across all visible GPUs. Each GPU processes `batch/N` samples independently in the forward pass. Gradients are summed on GPU 0 after backward.

**Prerequisite fix — `nn.ModuleList`:** DataParallel replicates all registered submodules to each GPU. Plain Python lists (`self.cs = [ClauseFunction(...)]`) are invisible to DataParallel, so `ClauseFunction` instances stayed on GPU 0 and caused device mismatch errors. All lists were converted to `nn.ModuleList`.

**Prerequisite fix — `register_buffer`:** Index tensors (`self.I`, `self.I_i`) were plain tensor attributes. DataParallel only moves `nn.Parameter` and registered buffers to each GPU — plain attributes stay on GPU 0. All index tensors were changed to `register_buffer`.

**`DataParallel` wrapping in `train_il.py`:**
```python
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    agent.model = nn.DataParallel(agent.model)
```

**`unwrapped_model` property in `imitation_agent.py`:** DataParallel wraps the model so `agent.model.get_prednames()` fails — DataParallel doesn't proxy arbitrary methods. A property was added to always return the underlying model:

```python
@property
def unwrapped_model(self):
    if isinstance(self.model, nn.DataParallel):
        return self.model.module
    return self.model
```

All calls to `model.get_prednames()`, `model.atoms`, `model.prednames` were updated to use `self.unwrapped_model`. `save()` was also fixed to save `unwrapped_model.state_dict()` to avoid `module.X` key prefix issues when reloading.

**Note on loss display:** DataParallel's loss is computed on GPU 0's sub-batch only. Divide by `torch.cuda.device_count()` for display only — never divide before calling `.backward()`:

```python
loss.backward()                                              # untouched
loss_val = loss.item() / max(torch.cuda.device_count(), 1)  # display only
```

**Result with batch=512, 4 GPUs:**
```
Samples/sec:  325/s
Epoch time:   ~10 min
```

---

## Step 7 — Diagnostic Upgraded for Multi-GPU

The original diagnostic timed a single cold batch, which was misleading with DataParallel — the first batch pays a one-time model replication cost to each GPU that doesn't represent steady-state performance.

The diagnostic was rewritten to run 30 warm-up batches (discarded) then average over 50 measured batches, reporting real wall-clock throughput:

```
Wall-clock it/s:        X.XX
Samples/sec:            XXX
Est. epoch time (200k): X.X min
```

---

## Step 8 — S=25 Rule Rewrite (Biggest Single Win)

**Problem discovered:** Profiling showed `I.shape = (18, 284, 625, 6)`. S=625 was traced to clause 17:

```
noop_evade(X):-close_by_enemy(P,E),close_by_enemy(P,E2),deeper_than_enemy(P,E2),
               higher_than_enemy(P,E),visible_enemy(E),visible_enemy(E2).
```

This clause ranges over two enemies simultaneously (E and E2), giving S = 25×25 = 625 substitution pairs. Every other clause ranges over a single object (S≤25). The clause was using only 625/177500 (0.4%) of its substitution slots.

**Fix:** Clause 17 was rewritten to range over a single enemy:
```
noop_evade(X):-close_by_enemy(P,E),higher_than_enemy(P,E),visible_enemy(E).
```

This reduced S from 625 to 25 across the entire index tensor.

**Result with batch=512, 4 GPUs:**
```
Before: Samples/sec 325/s, epoch ~10 min, GPU reserved 16.32 GB
After:  Samples/sec 902/s, epoch ~3.7 min, GPU reserved 0.82 GB
```

**This was the single largest optimization of the entire session — a 2.8× further speedup.**

---

## Step 9 — NaN Loss Fix (`imitation_agent.py`)

**Problem:** After all optimizations, NaN losses appeared intermittently. Debug output revealed `probs` occasionally reaching extreme values (e.g. `-312500000`, `2680`) — `softor` produces large negative values when all substitution slots for a clause are empty/zero.

These negative values flowed into `torch.log(action_scores + eps)` where adding a small epsilon to a large negative still gave a negative argument to log, producing NaN.

**Fix in `update()` and `predict()`:**
```python
# Clamp probs from model — softor can produce out-of-range on empty clauses
probs = probs.clamp(0.0, 1.0)
```

**Fix for NLL loss:**
```python
# BEFORE — adding eps to a negative doesn't help
log_action_scores = torch.log(action_scores + eps)

# AFTER — clamp ensures minimum before log
log_action_scores = torch.log(action_scores.clamp(min=eps))
```

Same fix applied in the validation loop in `train_il.py`.

---

## Loss Function Recommendation

**Use NLL (`--loss nll`), not BCE.**

NSFR rule valuations are logically independent soft scores — a rule fires based on its own conditions regardless of other rules. BCE penalizes all non-target action scores toward 0 every batch, which conflicts with NSFR's design where multiple rules can legitimately have non-zero valuations simultaneously (e.g. `up_evade` and `down_evade` both fire near an enemy).

NLL only requires the correct action to be the *highest scoring* — it doesn't penalize other rules for being active. This is more consistent with soft logic semantics and produces more stable training curves.

---

## Summary of All File Changes

### `infer.py`
| Location | Change | Reason |
|---|---|---|
| `ClauseFunction.__init__` | `self.I_i = I[i]` → `register_buffer('I_i', I[i])` | DataParallel GPU placement |
| `ClauseFunction.forward` | `.repeat()` → `.expand()` | Zero-copy, 11GB → 4GB memory |
| `ClauseBodySumFunction.__init__` | `self.I_i = I[i]` → `register_buffer('I_i', I[i])` | DataParallel GPU placement |
| `ClauseBodySumFunction.forward` | `.repeat()` → `.expand()` | Zero-copy memory fix |
| `InferModule.__init__` | `self.I = I` → `register_buffer('I', I)` | DataParallel GPU placement |
| `InferModule.__init__` | `self.cs = [...]` → `nn.ModuleList([...])` | DataParallel submodule registration |
| `InferModule.r()` | `expand + multiply` → `torch.einsum` | Avoids (m,C,B,G) intermediate tensor |
| `ClauseBodyInferModule.__init__` | `self.I = I` → `register_buffer('I', I)` | DataParallel GPU placement |
| `ClauseBodyInferModule.__init__` | `self.cs/cs_bs = [...]` → `nn.ModuleList` | DataParallel submodule registration |
| `ClauseInferModule.__init__` | `self.I/I_bk` → `register_buffer` | DataParallel GPU placement |
| `ClauseInferModule.__init__` | `self.cs/cs_bs/cs_bk = [...]` → `nn.ModuleList` | DataParallel submodule registration |

### `imitation_agent.py`
| Location | Change | Reason |
|---|---|---|
| `update()` | Returns `(loss, probs, action_scores)` | Eliminate double forward pass |
| `update()` | `probs.clamp(0.0, 1.0)` after forward | Fix softor out-of-range values |
| `update()` | `action_scores.clamp(min=eps)` before log | Fix NaN from log(negative) |
| New method `predict()` | Inference-only forward, no grad | Clean val loop, no redundant forward |
| New method `_aggregate_to_action_scores()` | Shared helper for rule→action aggregation | DRY — was duplicated 3 times |
| New property `unwrapped_model` | Returns `model.module` if DataParallel else `model` | DataParallel method access |
| `act()` | Simplified to use `predict()` | Uses shared helper |
| `save()` | `self.model` → `self.unwrapped_model` | Avoid `module.X` key prefix in saved weights |
| `load()` | `self.model` → `self.unwrapped_model` | Consistent with save |

### `train_il.py`
| Location | Change | Reason |
|---|---|---|
| After agent init | `nn.DataParallel(agent.model)` if >1 GPU | Multi-GPU training |
| Training loop | Reuse `action_scores` from `update()` | Eliminate double forward pass |
| Training loop | `loss_val = loss.item() / num_gpus` | Correct loss display with DataParallel |
| Training loop | `clip_grad_norm_(agent.unwrapped_model.parameters())` | Avoid double-counting with DataParallel |
| Validation loop | Replace aggregation block with `agent.predict()` | Uses shared helper, cleaner code |
| Validation loop | `action_scores.clamp(min=eps)` before log | Fix NaN from log(negative) |
| `run_diagnostics()` | 30 warm-up + 50 measured batches, wall-clock timing | Accurate throughput with DataParallel |
| Bug fix | `args.gazemap` → `args.use_gazemap` | AttributeError fix |
| Bug fix | Added `import pandas as pd` | Was missing, crashed at end |
| Added `--diagnose` flag | Triggers diagnostic mode then exits | Easy profiling without code changes |
| Added `--compile` flag | Optional `torch.compile` wrapping | Not recommended, kept for experimentation |
| Added `persistent_workers=True` | DataLoader workers persist between epochs | Avoids respawn overhead |
