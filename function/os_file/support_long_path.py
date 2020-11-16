import os


def support_long_path(path):
    
    if path.startswith("\\\\"):
        # Server. (ex: \\server_name\D\folder_name)
        path = path.replace("\\\\", "\\")
        path = u"\\\\?\\UNC" + path
    else:
        # Local. (ex: D:\Users\Jason)
        path = "\\\\?\\" + path
    
    return path


exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
exec_path = support_long_path(exec_path)
# Prove that we can use exec_path.
for file_name in os.listdir(exec_path):
    print(file_name)