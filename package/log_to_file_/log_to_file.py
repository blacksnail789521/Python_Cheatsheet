# https://github.com/Delgan/loguru
# import locale
# locale.setlocale(locale.LC_ALL, "en")
from loguru import logger

import sys
import os
import inspect


def print(*args, **kwargs):
    
    # If we don't have sep, we set sep to " " by default.
    sep = kwargs.get("sep", " ")
    
    # If we don't have end, we set end to "" by default.
    end = kwargs.get("end", "")
    
    # This line will let "print" log to every added handler.
    logger.opt(depth = 1).info \
        ( sep.join([ str(element) for element in list(args) ]) + end )


@logger.catch(reraise = True)
def call_function_with_catch_and_reraise(function):
    return function()


@logger.catch(default = "ERROR")
def call_function_with_catch_and_default(function):
    return function()


def get_default_format(format_length = {"name": 45, "function": 45, "line": 5}):
    
    default_format = \
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " + \
        "<level>{level: <5}</level> | " + \
        "<cyan>{name: <" + str(format_length["name"]) + "}</cyan> : " + \
        "<cyan>{function: <" + str(format_length["function"]) + "}</cyan> : " + \
        "<cyan>{line: <" + str(format_length["line"]) + "}</cyan> | " + \
        "<level>{message}</level>"
    
    return default_format


def log_to_file_for_func(reraise = True, enable_file = False, file_path = None, overwrite_file = True):
    
    def decorator(function):
        
        # *args and **kwargs are function's input.
        def wrapper(*args, **kwargs):
            
            result = log_to_file(lambda : function(*args, **kwargs), reraise, enable_file, file_path, overwrite_file)
            
            return result
        
        return wrapper
    
    return decorator


def log_to_file_for_class(reraise = "not_given_by_decorator", enable_file = "not_given_by_decorator", \
                          file_path = "not_given_by_decorator", overwrite_file = "not_given_by_decorator"):
    
    def decorator(function):
        
        # *args and **kwargs are function's input.
        def wrapper(*args, **kwargs):
            
            # reraise
            if reraise == "not_given_by_decorator":
                # Default of _reraise is True.
                try:
                    # Try to get from "self".
                    _reraise = args[0].reraise
                except:
                    _reraise = True
            else:
                _reraise = reraise
            
            # enable_file
            if enable_file == "not_given_by_decorator":
                # Default of _enable_file is False.
                try:
                    # Try to get from "self".
                    _enable_file = args[0].enable_file
                except:
                    _enable_file = False
            else:
                _enable_file = enable_file
            
            # file_path
            if file_path == "not_given_by_decorator":
                # Default of _file_path is None.
                try:
                    # Try to get from "self".
                    _file_path = args[0].file_path
                except:
                    _file_path = None
            else:
                _file_path = file_path
            
            # overwrite_file
            if overwrite_file == "not_given_by_decorator":
                # Default of _overwrite_file is True.
                try:
                    # Try to get from "self".
                    _overwrite_file = args[0].overwrite_file
                except:
                    _overwrite_file = True
            else:
                _overwrite_file = overwrite_file
            
            result = log_to_file(lambda : function(*args, **kwargs), _reraise, _enable_file, _file_path, _overwrite_file)
            
            return result
        
        return wrapper
    
    return decorator


def log_to_file(function, reraise = True, enable_file = True, file_path = None, overwrite_file = True, \
                outermost = False, show_demarcation = False, \
                enable_print = True, print_with_logging_style = False):
    
    def exist_new_file_path(new_file_path):
        
        if new_file_path in [ handler._name for handler_id, handler in logger._core.handlers.items() ]:
            return True
        else:
            return False
    
    
    """ Main function of log_to_file. """
    
    if outermost:
        # Initialization.
        logger.remove()
        
        # Set the console's printing.
        if enable_print == False:
            logger.add(sys.stderr, level = "ERROR", colorize = True, diagnose = False, backtrace = False, \
                       format = get_default_format())
        else:
            logger.add(sys.stderr, level = "INFO", colorize = True, diagnose = False, backtrace = False, \
                       format = "{message}" if print_with_logging_style == False else get_default_format())
    else:
        # Store non_disposable_handler_id_list in order to track disposable ones.
        non_disposable_handler_id_list = [ handler_id for handler_id, handler in logger._core.handlers.items() ]
    
    # Set default file_path.
    if file_path is None:
        # The original caller is at 1st level.
        caller_path = os.path.abspath(inspect.stack()[1].filename)
        
        # Get file_dir and file_name.
        file_dir = os.path.dirname(caller_path)
        file_name = os.path.splitext(os.path.basename(caller_path))[0]
        
        # Use file_dir and file_name to create file_path.
        file_path = os.path.join(file_dir, "Logging", f"(logging)_{file_name}.log")
    
    # Set file's logging.
    if enable_file == True and exist_new_file_path(file_path) == False:
        if overwrite_file == True:
            try:
                # Delete the old logging file directly.
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                # Someone is using the file for logging.
                pass
        
        logger.add(file_path, level = "DEBUG", diagnose = True, backtrace = False, format = get_default_format())
    
    if outermost and show_demarcation:
        # For every time we start to log, print the demarcation.
        logger.opt(raw = True).info("#" * 200 + "\n")
    
    # Call function.
    if reraise == True:
        output_of_function = call_function_with_catch_and_reraise(function)
    else:
        output_of_function = call_function_with_catch_and_default(function)
    
    if outermost:
        logger.remove()
    else:
        # Remember to remove disposable_handler. (Reserve non_disposable ones.)
        for handler_id, handler in logger._core.handlers.items():
            if handler_id not in non_disposable_handler_id_list:
                logger.remove(handler_id)
    
    return output_of_function