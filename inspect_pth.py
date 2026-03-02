import torch
import os

def compare_pths(path1, path2):
    print(f"\nComparing:\n1: {path1}\n2: {path2}")
    
    if not os.path.exists(path1) or not os.path.exists(path2):
        print("One or both files not found.")
        return
    
    data1 = torch.load(path1, map_location="cpu")
    data2 = torch.load(path2, map_location="cpu")
    
    if data1.keys() != data2.keys():
        print(f"Keys mismatch: {data1.keys()} vs {data2.keys()}")
        return
    
    for key in data1:
        v1 = data1[key]
        v2 = data2[key]
        if torch.is_tensor(v1) and torch.is_tensor(v2):
            if v1.shape != v2.shape:
                print(f"Shape mismatch for {key}: {v1.shape} vs {v2.shape}")
                continue
            
            diff = torch.abs(v1 - v2).sum().item()
            if diff > 1e-10:
                print(f"Key '{key}': DIFFER by {diff:.6f} (sum of absolute diff)")
            else:
                print(f"Key '{key}': IDENTICAL")
        else:
            if v1 != v2:
                print(f"Key '{key}': DIFFER (non-tensor)")
            else:
                print(f"Key '{key}': IDENTICAL (non-tensor)")

if __name__ == "__main__":
    base = "out/imitation/"
    compare_pths(base + "seaquest_new_il_no_gaze.pth", 
                 base + "seaquest_new_il_with_gaze_20.0.pth")
    compare_pths(base + "seaquest_new_evade_il_no_gaze.pth", 
                 base + "seaquest_new_evade_il_with_gaze_20.0.pth")
