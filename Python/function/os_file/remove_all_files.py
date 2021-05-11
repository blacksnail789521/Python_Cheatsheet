import os
import shutil


exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
remove_path = os.path.join(exec_path, "output")

if os.path.exists(remove_path):
    print("Remove!")
    shutil.rmtree(remove_path)

# One line
shutil.rmtree(remove_path, ignore_errors = True)