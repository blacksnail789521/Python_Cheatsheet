import importlib

def main():
    with open('test_main.txt', 'r', encoding='UTF-8') as file:
        for line in file:
            print(line)

if __name__ == "__main__":
    
    caller_example_module = importlib.import_module("package.package.caller_example")
    caller_example_module.main()
    
    main()