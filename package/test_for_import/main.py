import sys
import os

package_paths = [os.path.join(os.path.abspath(os.path.dirname(__file__)), "package", "package")]
for package_path in package_paths: 
    if package_path not in sys.path: sys.path.insert(0, package_path)

import caller_example


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, "test_main.txt"), "r") as file:
        for line in file:
            print(line)

if __name__ == "__main__":
    
    caller_example.main()
    main()