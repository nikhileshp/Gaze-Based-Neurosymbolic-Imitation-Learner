import torch
import sys

def inspect_pkl():
    path = 'data/seaquest/train_atoms.pkl'
    print(f"Loading {path}...")
    data = torch.load(path)
    
    if 'data' not in data:
        print("Key 'data' not found.")
        return

    items = data['data']
    if len(items) == 0:
        print("Data is empty.")
        return

    item = items[0]
    print("First item keys:", item.keys())
    
    if 'atoms' in item:
        print("Atoms shape:", len(item['atoms']))
        print("Atoms example:", item['atoms'][:5])
        
    if 'state' in item:
        print("State found!")
    else:
        print("State NOT found in item.")

if __name__ == "__main__":
    inspect_pkl()
