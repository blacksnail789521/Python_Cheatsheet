import os


current_path = os.path.dirname(os.path.abspath(__file__))
print(os.path.join(current_path))
# Add ".." and os.path.abspath
print(os.path.abspath(os.path.join(current_path, "..")))
print(os.path.abspath(os.path.join(current_path, "..", "..")))