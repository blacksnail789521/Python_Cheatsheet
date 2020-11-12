import importlib
import os
import sys

def main():
    
    # Set python's working directory to module level so that we can call them. (We need to change back after we get the module.)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    EPT_module = importlib.import_module("EPT")
    EPT_DATA = getattr(EPT_module, "EPT_DATA")
    EPT_DATA(0)
    
    # We need to change back after we get the module.
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

if __name__ == "__main__":
    main()