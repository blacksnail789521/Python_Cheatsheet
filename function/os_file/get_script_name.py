import os
from pathlib import Path


script_path = __file__
script_name = os.path.splitext(os.path.basename(script_path))[0]
print(f"script_name: {script_name}")

script_name = Path(script_path).stem
print(f"script_name: {script_name}")