import torch
import torch.nn as nn
from core.nsfr.valuation import ValuationModule
from nsfr.utils.logic import get_lang

def test_freeway_scaling_and_continuity():
    # Setup dummy language for Freeway
    lark_path = 'core/nsfr/lark/exp.lark'
    lang_base_path = 'core/envs/freeway/logic/'
    rules_name = 'new'
    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, rules_name)
    
    # Instance with unnormalized=True
    vm = ValuationModule(val_fn_path='core/envs/freeway/valuation.py', lang=lang, device='cpu', 
                         unnormalized=True, visible_preds_only=True)
    
    # 1. Test Continuity of closeby
    # Freeway dimensions: breadth=160, height=210 (roughly)
    # Positions are stored in zs at index 3:5 (x, y)
    # Chicken 1: [vis, type1, type2, x=44, y=100, ...]
    # Car 1:     [vis, type1, type2, x=44, y=105, ...]
    
    z_chicken = torch.zeros((1, 5))
    z_car = torch.zeros((1, 5))
    
    # Same lane (close y), same x -> should be high prob
    z_chicken[0, 3] = 44.0; z_chicken[0, 4] = 100.0
    z_car[0, 3] = 44.0; z_car[0, 4] = 101.0
    
    val_close_1 = vm.val_fns['closeby'](z_chicken, z_car)
    print(f"closeby (exact match): {val_close_1.item():.4f}")
    assert 0.8 < val_close_1.item() <= 1.0
    
    # Slightly further in y -> should be lower but non-zero
    z_car[0, 4] = 104.0
    val_close_2 = vm.val_fns['closeby'](z_chicken, z_car)
    print(f"closeby (slightly further y): {val_close_2.item():.4f}")
    assert 0.0 < val_close_2.item() < val_close_1.item()
    
    # 2. Test Clamping with Unnormalized Gaze
    # Create arguments for visible predicate
    # Z shape: (B, F) where last feature is gaze
    # Gaze appended by FactsConverter is at index -1
    z_obj = torch.zeros((1, 6))
    z_obj[0, 0] = 1.0 # visible
    z_obj[0, 3] = 44.0; z_obj[0, 4] = 100.0 # pos
    z_obj[0, 5] = 5.0 # LARGE unnormalized gaze sum
    
    # Call _call_val_fn which handles the scaling and damping
    # args: List of tensors
    # val_fn 'visible' in freeway takes 1 arg (an object)
    val_clamped = vm._call_val_fn('visible', [z_obj], None, is_object_list=[True])
    print(f"visible (unnormalized gaze=5.0) result: {val_clamped.item():.4f}")
    
    # visible() returns 0.99 for present=1
    # scaling: 0.99 * 5.0 = 4.95 -> CLAMPED TO 1.0
    assert abs(val_clamped.item() - 1.0) < 0.01
    
    # Test Scaling < 1.0
    z_obj[0, 5] = 0.5
    val_scaled = vm._call_val_fn('visible', [z_obj], None, is_object_list=[True])
    print(f"visible (unnormalized gaze=0.5) result: {val_scaled.item():.4f}")
    # scaling: 0.99 * 0.5 = 0.495
    assert abs(val_scaled.item() - 0.495) < 0.01

    print("\nALL FREEWAY VALUATION TESTS PASSED")

if __name__ == "__main__":
    test_freeway_scaling_and_continuity()
