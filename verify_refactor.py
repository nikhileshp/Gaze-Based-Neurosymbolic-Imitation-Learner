import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from core.envs.asterix.env import NSFREnv
from core.envs.asterix import valuation as asterix_val

class MockEntity:
    def __init__(self, category, xy):
        self.category = category
        self.xy = xy

def test_asterix_refactor():
    print("Testing Asterix Refactor...")
    
    # 1. Test Environment extract_logic_state
    # Mock OCAtari to avoid actual initialization which can hang or fail
    from unittest.mock import MagicMock
    import ocatari.core
    ocatari.core.OCAtari = MagicMock()
    
    env = NSFREnv(mode='logic')
    
    raw_state = [
        MockEntity("Player", (10, 20)),
        MockEntity("Enemy", (30, 40)),
        MockEntity("Cauldron", (50, 60)),
        MockEntity("Reward50", (70, 80))
    ]
    
    logic_state = env.extract_logic_state(raw_state)
    
    # Expected structure: [vis, P, E, C, R, x, y]
    # Player (obj1)
    obj1 = logic_state[1]
    assert obj1[0] == 1, f"Obj1 visibility should be 1, got {obj1[0]}"
    assert obj1[1] == 1, f"Obj1 should be Player, got {obj1[1]}"
    assert obj1[5] == 10 and obj1[6] == 20, f"Obj1 coords wrong, got {obj1[5:7]}"
    
    # Enemy (obj2)
    obj2 = logic_state[2]
    assert obj2[0] == 1, "Obj2 visibility should be 1"
    assert obj2[2] == 1, "Obj2 should be Enemy"
    assert obj2[5] == 30 and obj2[6] == 40, "Obj2 coords wrong"

    # Consumable (obj3)
    obj3 = logic_state[3]
    assert obj3[0] == 1, "Obj3 visibility should be 1"
    assert obj3[3] == 1, "Obj3 should be Consumable"
    
    # Reward (obj4)
    obj4 = logic_state[4]
    assert obj4[0] == 1, "Obj4 visibility should be 1"
    assert obj4[4] == 1, "Obj4 should be Reward"

    print("✓ Environment extract_logic_state OK")

    # 2. Test Valuation Functions
    z = torch.tensor(logic_state[1:5]).float() # (4, 7)
    
    # is_player
    assert asterix_val.is_player(z[0:1]).item() == True, "is_player(player) failed"
    assert asterix_val.is_player(z[1:2]).item() == False, "is_player(enemy) failed"
    
    # is_present
    assert asterix_val.is_present(z[0:1]).item() == True, "is_present(player) failed"
    
    # type
    # [vis, P, E, C, R, x, y]
    # type(z, a) checks z[:, 1:5]
    # a=[1,0,0,0] -> Player
    player_type = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert asterix_val.type(z[0:1], player_type).item() > 0.5, "type(player, player_type) failed"
    assert asterix_val.type(z[1:2], player_type).item() < 0.5, "type(enemy, player_type) failed"
    
    enemy_type = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    assert asterix_val.type(z[1:2], enemy_type).item() > 0.5, "type(enemy, enemy_type) failed"

    print("✓ Valuation functions OK")

if __name__ == "__main__":
    test_asterix_refactor()
