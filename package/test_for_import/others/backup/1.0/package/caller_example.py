import importlib
import os

def main():
    # Set python's working directory to module level so that we can call them. (We need to change back after we get the module.)
    folder_name = os.path.basename(os.path.abspath(__file__ + "/../"))
    os.chdir(os.path.dirname(os.path.abspath(__file__ + "/../")))
    EPT_module = importlib.import_module(folder_name + ".EPT")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    EPT_DATA = getattr(EPT_module, "EPT_DATA")
    EPT_DATA(0)

if __name__ == "__main__":
    main()