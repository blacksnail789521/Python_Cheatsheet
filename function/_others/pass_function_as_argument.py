def greeting(function, demarcation = True):
    
    print("Greeting!")
    if demarcation:
        print("-----------------")
    result = function()
    
    return result


def say_hi(start_message, end_message = "end"):
    
    print(start_message)
    print("hello there")
    print(end_message)
    
    return "YA"


def test_for_passing_directly():
    
    return greeting(lambda : say_hi("start", end_message = "end"), demarcation = True)


def test_for_decorator_by_func():
    
    def greeting_for_func(demarcation = True):
        
        def decorator(function):
            
            # *args and **kwargs are function's input.
            def wrapper(*args, **kwargs):
                
                result = greeting(lambda : function(*args, **kwargs), demarcation)
                
                return result
            
            return wrapper
        
        return decorator
    
    
    @greeting_for_func(demarcation = True)
    def say_hi_by_func(start_message, end_message = "end"):
        
        return say_hi(start_message, end_message)
    
    
    """ Main function of test_for_decorator_by_func. """
    
    return say_hi_by_func("start", end_message = "end")


def test_for_decorator_by_class():
    
    def greeting_for_class(demarcation = None):
        
        # If demarcation is not given by decorator, we should use either self's value or default value in wrapper.
        
        def decorator(function):
            
            # *args and **kwargs are function's input.
            def wrapper(*args, **kwargs):
                
                # Inside wrapper, we must use "_demarcation" instead of "demarcation" because of UnboundLocalError.
                
                # demarcation.
                if demarcation is None:
                    try:
                        # Use self's value.
                        _demarcation = args[0].demarcation
                        assert isinstance(_demarcation, bool)
                    except:
                        # Default of _demarcation is True.
                        _demarcation = True
                else:
                    _demarcation = demarcation
                
                result = greeting(lambda : function(*args, **kwargs), _demarcation)
                
                return result
            
            return wrapper
        
        return decorator
    
    
    class say_hi_class():
        
        def __init__(self):
            
            self.demarcation = True
        
        
        @greeting_for_class()
        def say_hi_by_class(self, start_message, end_message = "end"):
            
            return say_hi(start_message, end_message)
    
    
    """ Main function of test_for_decorator_by_class. """
    
    return say_hi_class().say_hi_by_class("start", end_message = "end")
    
        
def main():
    
    result = test_for_passing_directly()
    # result = test_for_decorator_by_func()
    # result = test_for_decorator_by_class()
    
    return result


if __name__ == "__main__":

    result = main()