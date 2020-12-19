import os


exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
file_path = os.path.join(exec_path, "check_path_existence.py")

# Check folder.
if os.path.exists(exec_path):
    print("Exist!")
else:
    print("Not exist!")
    
# Check file.
if os.path.exists(file_path):
    print("Exist!")
else:
    print("Not exist!")