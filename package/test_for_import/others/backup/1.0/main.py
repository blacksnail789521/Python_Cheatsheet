import importlib

if __name__ == "__main__":
    caller_example_module = importlib.import_module("package.caller_example")
#    main = getattr(caller_example_module, "main")
#    main()
    caller_example_module.main()