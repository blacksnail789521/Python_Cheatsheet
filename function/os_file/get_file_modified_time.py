import os
from datetime import datetime


exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
file_path = os.path.join(exec_path, "get_file_modified_time.py")
modified_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
print(modified_time)