"""
Debug script to check if there are black pixels at y=45-55 in Seaquest frames.
"""
import numpy as np
from ocatari.core import OCAtari

# Initialize with vision mode
env = OCAtari("ALE/Seaquest-v5", mode="vision", render_mode="rgb_array")
env.reset()

# Take a few steps
for _ in range(10):
    env.step(0)  # NOOP

# Get the frame
frame = env.getScreenRGB()

print("=== Seaquest Frame Analysis ===")
print(f"Frame shape: {frame.shape}")
print(f"Frame dtype: {frame.dtype}")

# Check surface region (y=45-55)
surface_region = frame[45:56, :, :]
print(f"\nSurface region shape: {surface_region.shape}")

# Count black pixels
black_pixels = np.all(surface_region == [0, 0, 0], axis=-1)
num_black = np.sum(black_pixels)
total_pixels = surface_region.shape[0] * surface_region.shape[1]

print(f"Black pixels in surface region: {num_black} / {total_pixels}")
print(f"Percentage black: {100 * num_black / total_pixels:.2f}%")

# Check row by row
print("\nBlack pixels per row in surface region (y=45-55):")
for i, y in enumerate(range(45, 56)):
    row_black = np.sum(np.all(surface_region[i, :, :] == [0, 0, 0], axis=-1))
    print(f"  y={y}: {row_black}/160 black pixels")

# Sample some pixel values around the surface
print("\nSample pixel values:")
for y in [44, 45, 50, 55, 56]:
    pixel = frame[y, 80]  # middle of screen
    print(f"  y={y}, x=80: RGB={pixel}")

print("\n=== Detection Test ===")
# Manually test the detection logic
import numpy as np
obs = frame
surface_region = obs[45:56, :, :]
black_color = np.array([0, 0, 0])
black_pixels_mask = np.all(surface_region == black_color, axis=-1)

print(f"Any black pixels found: {np.any(black_pixels_mask)}")
print(f"Detection would trigger: {np.any(black_pixels_mask)}")

env.close()
