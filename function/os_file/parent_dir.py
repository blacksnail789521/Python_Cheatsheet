import os


print("\n#####################")
print("### First method  ###")
print("#####################")
print("---------------------")
print("current path:")
exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
print(exec_path)

print("---------------------")
print("parent path:")
exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(exec_path)

print("---------------------")
print("parent's parent path:")
exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
print(exec_path)


print("\n#####################")
print("### Second method ###")
print("#####################")
print("---------------------")
print("current path:")
exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.curdir))
print(exec_path)

print("---------------------")
print("parent path:")
exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
print(exec_path)

print("---------------------")
print("parent's parent path:")
exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
print(exec_path)