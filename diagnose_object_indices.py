"""
Script to determine actual object index mapping in Seaquest environment.
Run this to find out which indices correspond to which object types.
"""
from nudge.env import NudgeBaseEnv
import torch as th

# Initialize environment
env = NudgeBaseEnv.from_name("seaquest", mode="logic", render_oc_overlay=False)

# Print offset information
print("Object Index Allocation:")
print("="*60)
if hasattr(env, 'obj_offsets'):
    for obj_type, offset in sorted(env.obj_offsets.items(), key=lambda x: x[1]):
        print(f"{obj_type:20s}: starts at index {offset}")

print("\n" + "="*60)
print("Running environment to collect actual indices...")
print("="*60)

# Run for a few steps and collect object indices
env.reset()
seen_indices = {}

for step in range(200):
    action = "noop"
    (logic_state, neural_state), reward, done = env.step(action)
    
    if hasattr(env, 'env') and hasattr(env.env, 'objects'):
        objects = env.env.objects
        for i, obj in enumerate(objects):
            obj_type = obj.__class__.__name__
            if obj_type != 'NoObject':
                if obj_type not in seen_indices:
                    seen_indices[obj_type] = set()
                seen_indices[obj_type].add(i)
    
    if done:
        env.reset()

print("\nObserved Object Type -> Index Mapping:")
print("="*60)
for obj_type in sorted(seen_indices.keys()):
    indices = sorted(seen_indices[obj_type])
    if len(indices) <= 10:
        print(f"{obj_type:20s}: {indices}")
    else:
        print(f"{obj_type:20s}: {indices[:5]} ... {indices[-5:]} ({len(indices)} total)")

print("\n" + "="*60)
print("Recommended consts.txt mapping:")
print("="*60)

# Generate recommended mapping
if 'Player' in seen_indices or 'Submarine' in seen_indices:
    player_indices = seen_indices.get('Player', set()) | seen_indices.get('Submarine', set()) | seen_indices.get('SurfaceSubmarine', set())
    if player_indices:
        player_idx = min(player_indices)
        print(f"oplayer:obj{player_idx}")

if 'Diver' in seen_indices or 'CollectedDiver' in seen_indices:
    diver_indices = sorted(seen_indices.get('Diver', set()) | seen_indices.get('CollectedDiver', set()))
    if diver_indices:
        diver_str = ','.join([f"obj{i}" for i in diver_indices[:10]])  # Limit to first 10
        print(f"odiver:{diver_str}")

if 'Shark' in seen_indices:
    shark_indices = sorted(seen_indices['Shark'])
    if shark_indices:
        shark_str = ','.join([f"obj{i}" for i in shark_indices])
        print(f"oenemy:{shark_str}")

if 'EnemyMissile' in seen_indices or 'PlayerMissile' in seen_indices:
    missile_indices = sorted(seen_indices.get('EnemyMissile', set()) | seen_indices.get('PlayerMissile', set()))
    if missile_indices:
        missile_str = ','.join([f"obj{i}" for i in missile_indices[:10]])
        print(f"omissile:{missile_str}")

if 'OxygenBar' in seen_indices:
    oxygen_indices = sorted(seen_indices['OxygenBar'])
    if oxygen_indices:
        print(f"ooxygenbar:obj{oxygen_indices[0]}")

env.close()
