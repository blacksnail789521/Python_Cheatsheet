import os


exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
for root, dirs, files in os.walk(exec_path):
    for file in files:
        file_path = os.path.join(root, file)
        print(file_path)