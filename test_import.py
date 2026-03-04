import sys, os
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_scripts_dir)
for _p in [_scripts_dir, _project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from baselines.models.linear_models import Encoder
    print(f"Imported from: {Encoder.__module__}")
    import baselines.models.linear_models
    print(f"File path: {baselines.models.linear_models.__file__}")
except ImportError as e:
    print(f"Import failed: {e}")

try:
    from models.linear_models import Encoder
    print(f"Imported from models: {Encoder.__module__}")
except ImportError as e:
    print(f"Import failed from models: {e}")
