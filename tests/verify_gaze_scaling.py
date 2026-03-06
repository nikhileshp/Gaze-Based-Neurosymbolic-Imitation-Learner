import torch
import torch.nn as nn
from core.nsfr.valuation import ValuationModule
from nsfr.utils.logic import get_lang

def test_scaling():
    # Setup dummy language
    lark_path = 'core/nsfr/lark/exp.lark'
    lang_base_path = 'core/envs/seaquest/logic/'
    rules_name = 'new'
    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, rules_name)
    
    # Mock valuation functions
    def visible_enemy(obj): return torch.tensor([1.0])
    def closeby(obj1, obj2): return torch.tensor([1.0])
    
    # Instance with vis_only=True
    vm = ValuationModule(val_fn_path='core/envs/seaquest/valuation.py', lang=lang, device='cpu', 
                         unnormalized=True, visible_preds_only=True)
    
    # Overwrite loaded functions for testing
    vm.val_fns['visible_enemy'] = visible_enemy
    vm.val_fns['closeby'] = closeby
    
    # Dummy object tensors (B=1, N=2, F=8)
    # Feature 0: present, Feature 7: gaze
    zs = torch.zeros((1, 2, 8))
    # Obj0: gaze = 0.2
    zs[0, 0, 0] = 1.0; zs[0, 0, 7] = 0.2
    # Obj1: gaze = 0.8
    zs[0, 1, 0] = 1.0; zs[0, 1, 7] = 0.8
    
    # Case 1: visible_enemy(obj0) -> expects 0.2
    val_vis = vm._call_val_fn('visible_enemy', [zs[0:1, 0]], None)
    print(f"visible_enemy (vis_only=True) result: {val_vis.item():.2f}")
    assert abs(val_vis.item() - 0.2) < 0.01

    # Case 2: closeby(obj0, obj1) -> expects 1.0 (unscaled because vis_only=True)
    val_close = vm._call_val_fn('closeby', [zs[0:1, 0], zs[0:1, 1]], None)
    print(f"closeby (vis_only=True) result: {val_close.item():.2f}")
    assert abs(val_close.item() - 1.0) < 0.01

    # Case 3: visible_preds_only=False -> closeby(obj0, obj1) -> expects max(0.2, 0.8) = 0.8
    vm.visible_preds_only = False
    val_close_scaled = vm._call_val_fn('closeby', [zs[0:1, 0], zs[0:1, 1]], None)
    print(f"closeby (vis_only=False) result: {val_close_scaled.item():.2f}")
    assert abs(val_close_scaled.item() - 0.8) < 0.01

    print("ALL TESTS PASSED")

def test_smoothing():
    print("\n--- Testing Laplacian Smoothing Math ---")
    alpha = 0.1
    # 2 present objects, gaze sums 0.8 and 0.0
    gaze_sums = torch.tensor([[0.8, 0.0]])
    is_present = torch.tensor([[1.0, 1.0]])
    num_present = is_present.sum(dim=1, keepdim=True) # 2.0
    
    total_mass = gaze_sums.sum(dim=1, keepdim=True) # 0.8
    # Formula: (gaze_sums + alpha) / (total_mass + num_present * alpha)
    # Norm: (0.8 + 0.1) / (0.8 + 2 * 0.1) = 0.9 / 1.0 = 0.9
    #       (0.0 + 0.1) / (0.8 + 2 * 0.1) = 0.1 / 1.0 = 0.1
    
    smoothed = (gaze_sums + alpha) / (total_mass + num_present * alpha + 1e-8)
    smoothed = smoothed * is_present
    
    print(f"Gaze Sums: {gaze_sums.tolist()}")
    print(f"Smoothed:  {smoothed.tolist()}")
    
    assert abs(smoothed[0, 0].item() - 0.9) < 0.01
    assert abs(smoothed[0, 1].item() - 0.1) < 0.01
    assert abs(smoothed.sum().item() - 1.0) < 0.01
    print("Smoothing Math Test Passed!")

if __name__ == "__main__":
    test_scaling()
    test_smoothing()
