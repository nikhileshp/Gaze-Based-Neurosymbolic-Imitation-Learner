
try:
    from ocatari.vision.seaquest import objects_from_observation
    print("Ocatari Vision for Seaquest found.")
except ImportError:
    print("Ocatari Vision for Seaquest NOT found.")

try:
    from ocatari.core import OCAtari
    print("OCAtari class found.")
except ImportError:
    print("OCAtari class NOT found.")
