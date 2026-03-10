import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Add project root and core to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'core'))

# Mock OCAtari before importing environment
import ocatari.core
ocatari.core.OCAtari = MagicMock()

from core.envs.asterix.env import NSFREnv
from core.nsfr.nsfr import NSFReasoner
from core.nsfr.common import get_nsfr_model
from core.nsfr.fol.logic import NeuralPredicate

class MockEntity:
    def __init__(self, category, xy):
        self.category = category
        self.xy = xy

def test_asterix_end_to_end():
    print("Testing Asterix End-to-End Valuation Pipeline...")
    
    # 1. Setup Environment and Reasoner
    env_name = "asterix"
    rules = "best"
    device = torch.device("cpu")
    
    # Initialize NSFR Model
    # This loads the language, predicates, etc.
    nsfr = get_nsfr_model(env_name, rules, device)
    fc = nsfr.fc
    vm = fc.vm
    
    # 2. Create Mock State
    env = NSFREnv(mode='logic')
    raw_state = [
        MockEntity("Player", (10, 20)),
        MockEntity("Enemy", (30, 40))
    ]
    logic_state = env.extract_logic_state(raw_state)
    Z = torch.tensor(logic_state).unsqueeze(0).float() # (1, N_OBJ, F)
    
    # 3. Simulate Gaze (Heatmap)
    # 84x84 heatmap.sx=84/160, sy=84/210. 
    # Player at (10, 20) -> x=5, y=8.
    gaze = torch.zeros((1, 84, 84))
    gaze[0, 8, 5] = 1.0 # Gaze on player
    
    # 4. Run through FactsConverter
    # G derived from get_lang in get_nsfr_model
    G = nsfr.atoms
    B = nsfr.bk
    
    print("Running FactsConverter.convert()...")
    # This will use the new index-0 visibility check and gaze scaling
    V = fc.convert(Z, G, B, gaze=gaze)
    
    # 5. Verify Valuations
    # Find indices for some predicates
    # e.g., player(obj1), enemy(obj2), visible(obj1), visible(obj2)
    # The atom labels depend on the ruleset. In 'best', it might be type(obj1, player) etc.
    
    player_idx = -1
    enemy_idx = -1
    for i, atom in enumerate(G):
        label = str(atom).lower()
        # Look for any atom that specifies obj1 is a player
        if "obj1" in label and "player" in label:
            player_idx = i
        # Look for any atom that specifies obj2 is an enemy
        if "obj2" in label and "enemy" in label:
            enemy_idx = i
            
    print(f"Indices found: player(obj1)={player_idx}, enemy(obj2)={enemy_idx}")
    
    if player_idx != -1:
        val_p = V[0, player_idx].item()
        print(f"Valuation for player(obj1): {val_p:.4f}")
        assert val_p > 0.5, f"Player valuation should be high, got {val_p}"
        
    if enemy_idx != -1:
        val_e = V[0, enemy_idx].item()
        print(f"Valuation for enemy(obj2): {val_e:.4f}")
        assert val_e > 0.5, f"Enemy valuation should be high, got {val_e}"

    print("✓ End-to-End Asterix Valuation Verification OK")

if __name__ == "__main__":
    test_asterix_end_to_end()
