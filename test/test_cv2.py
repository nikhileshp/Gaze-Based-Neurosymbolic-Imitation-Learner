import cv2
import gym
import numpy as np

print("Testing cv2 import and resize...")
frame = np.zeros((210, 160, 3), dtype=np.uint8)
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
print(f"Resized shape: {resized.shape}")
print("Test complete.")
