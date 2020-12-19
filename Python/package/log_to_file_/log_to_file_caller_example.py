from log_to_file import log_to_file, print, log_to_file_for_func, log_to_file_for_class

import time
import os


inner_reraise = False # If True, we should NOT have ZeroDivisionError.
reraise = False

enable_file = True
exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
file_path = os.path.join(exec_path, "(logging)_test.log")
overwrite_file = True


""" log_to_file """
"""------------------------------------------------------------------------"""
def some_block_without_decorator_inner(a_inner):
    
    print("inner block before:", a_inner)
    raise Exception("Inner error")
    print("inner block after")


def some_block_without_decorator(a):
    
    print("block before:", a)
    
    time.sleep(3)
    
    log_to_file(lambda : some_block_without_decorator_inner(a), reraise = inner_reraise, \
                enable_file = enable_file, file_path = file_path, overwrite_file = overwrite_file)
    
    time.sleep(3)
    
    test = 1/0
    
    print("block after")
    
    return test
"""------------------------------------------------------------------------"""


""" log_to_file_for_func """
"""------------------------------------------------------------------------"""
@log_to_file_for_func(inner_reraise, enable_file, file_path, overwrite_file)
def some_block_with_decorator_inner(a_inner):
    
    print("inner block before:", a_inner)
    raise Exception("Inner error")
    print("inner block after")


@log_to_file_for_func(reraise, enable_file, file_path, overwrite_file)
def some_block_with_decorator(a):
    
    print("block before:", a)
    
    time.sleep(3)
    
    some_block_with_decorator_inner(a)
    
    time.sleep(3)
    
    test = 1/0
    
    print("block after")
    
    return test
"""------------------------------------------------------------------------"""


""" log_to_file_for_class """
"""------------------------------------------------------------------------"""
class some_block_class():
    
    def __init__(self):
        
        self.reraise = reraise
        
        self.enable_file = enable_file
        self.file_path = file_path
        self.overwrite_file = overwrite_file
        
    
    @log_to_file_for_class(reraise = inner_reraise)
    def some_block_inner(self, a_inner):
        
        print("inner block before:", a_inner)
        raise Exception("Inner error")
        print("inner block after")
    
    
    @log_to_file_for_class()
    def some_block(self, a):
        
        print("block before:", a)
        
        time.sleep(3)
        
        self.some_block_inner(a)
        
        time.sleep(3)
        
        test = 1/0
        
        print("block after")
        
        return test
"""------------------------------------------------------------------------"""


def easy_test(a):
    
    1/0
    
    return a


def main():
    
    print("start")
    
    # Choose either one.
    output = log_to_file(lambda : some_block_without_decorator(1), reraise = reraise, \
                         enable_file = enable_file, file_path = file_path, overwrite_file = overwrite_file)
    # output = some_block_with_decorator(1)
    # output = some_block_class().some_block(1)
    # output = easy_test(1)
    
    time.sleep(3)
    print("##################")
    print(output)
    print("##################")
    
    raise Exception("XD")
    
    print("end")
    
    return output


if __name__ == "__main__":

    result = log_to_file(lambda : main(), outermost = True, reraise = False)