# https://github.com/kata198/func_timeout
from func_timeout import func_timeout, FunctionTimedOut

# exec("""import sys,os\nexec_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))\nif exec_path not in sys.path:sys.path.insert(0,exec_path)""")
# from package.log_to_file_.log_to_file import print

import inspect
import sys


def timeout_func(function, timeout_spec, raise_Exception = False, show_full_func_name = False):
    
    # Get func_name.
    full_func_name = inspect.getsource(function).strip()
    short_func_name = full_func_name.split("(")[1].replace("lambda", "").split(":")[1].strip()
    func_name = short_func_name if show_full_func_name == False else full_func_name
    
    # Call function with timeout mechanism.
    try:
        return func_timeout(timeout_spec, function)
    
    except FunctionTimedOut:
        print_message = f"The function '{func_name}' could not complete within {timeout_spec} seconds and was terminated."
        error_message = print_message
        error_obj_for_raise = TimeoutError
        # If we want to use isinstance(result, TimeoutError), we must raise manually.
        try:
            raise TimeoutError()
        except TimeoutError as e:
            error_obj_for_return = e
    
    except Exception as e:
        print_message = f"The function '{func_name}' had some error itself. Error message:\n{e.__class__.__name__}: {e}"
        error_message = e
        error_obj_for_raise = sys.exc_info()[0]
        error_obj_for_return = e
    
    # Raise exception if we need.
    if raise_Exception == True:
        raise error_obj_for_raise(error_message)
    else:
        return error_obj_for_return