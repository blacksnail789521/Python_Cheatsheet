exec("""from os.path import join, abspath, dirname\ntry:\n    """ +
     """with open(join(abspath(dirname(__file__)), "package", """ + 
     """ "log_to_file", "import.txt")) as f: exec(f.read())\nexcept:pass""")

import time


def test_function(number):
    print_some_number(number)
    time.sleep(1)
    intensive_function()
    print_some_number(number)
#    divide_by_zero()

def print_some_number(number):
    for i in range(number):
        print(i)

def intensive_function():
    a = 0
    for i in range(10000000):
        a = minus(a) + 1

def minus(a):
    return a - 1

def divide_by_zero():
    1/0