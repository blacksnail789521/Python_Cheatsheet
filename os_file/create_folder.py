import os


exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
output_path = os.path.join(exec_path, "output")

# Create output folder.
os.makedirs(output_path, exist_ok = True)