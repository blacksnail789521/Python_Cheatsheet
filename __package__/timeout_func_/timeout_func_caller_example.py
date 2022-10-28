from timeout_func import timeout_func

import time


def countdown(n, enable_divide_by_zero = False, \
              start_message = "start", end_message = "end"):
    
    print(start_message)
    for i in range(n):
        print(n - i)
        time.sleep(1)
        
        if enable_divide_by_zero and i == 1:
            1/0
    print(0)
    print(end_message)
    
    return "Finished!!"


def main():
    
    """ Set parameters. (The top is the default one.) """
    """-----------------------------------------------"""
    enable_timeout_in_this_example = True
    # enable_timeout_in_this_example = False
    
    enable_divide_by_zero = False
    # enable_divide_by_zero = True
    
    raise_Exception = False
    # raise_Exception = True
    
    show_full_func_name = False
    # show_full_func_name = True
    """-----------------------------------------------"""
    
    if enable_timeout_in_this_example:
        timeout_spec = 3
    else:
        timeout_spec = 10
    
    result = timeout_func(lambda : countdown(5, enable_divide_by_zero), timeout_spec, \
                          raise_Exception, show_full_func_name)
    
    time.sleep(0.2)
    
    # If we don't want to raise Exception, we can still distinguish the error from result.
    if isinstance(result, TimeoutError):
        print("We had a timeout error.")
    elif isinstance(result, BaseException):
        print("We had a {error}.".format(error = result.__class__.__name__))
    else:
        print("We don't have any error.")
    
    return result


if __name__ == "__main__":
    
    result = main()