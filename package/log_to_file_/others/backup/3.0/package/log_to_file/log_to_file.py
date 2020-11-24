import sys
from os.path import join, abspath, dirname, basename
import inspect

package_path = join(dirname(abspath(__file__)))
if package_path not in sys.path: sys.path.insert(0, package_path)

from loguru import logger



def print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "")
    logger.opt(depth = 1).info \
        ( sep.join([ str(element) for element in list(args) ]) + end )


@logger.catch
def call_function(function):
    return function()

# log_opt = 0: Don't log on file, 1: Please log on file.
# print_opt = 0: Don't print on console (except error), 
#             1: Print without logging style, 
#             2:  Print with logging style.
# format_length: set the maximum length for each attribute.

def log_to_file(function, \
                file_name = None, \
                log_opt = 1, \
                print_opt = 1, \
                format_length = {"name": 25, "function": 20, "line": 4}, \
                show_demarcation = True, \
                remove_logger = True):
    
    # Initialization.
    logger.remove()
    default_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " + \
                     "<level>{level: <5}</level> | " + \
                     "<cyan>{name: <" + str(format_length["name"]) + \
                         "}</cyan> : " + \
                     "<cyan>{function: <" + str(format_length["function"]) + \
                         "}</cyan> : " + \
                     "<cyan>{line: <" + str(format_length["line"]) + \
                         "}</cyan> | " + \
                     "<level>{message}</level>"
                     
    # Set the file logging.
    if log_opt == 1:
        caller_path = abspath(inspect.stack()[1].filename)
        caller_dir = dirname(caller_path)
        if file_name is None:
            file_name = basename(caller_path) \
                        .replace(".py", "")
            file_name = "(logging)_" + file_name
        logger.add(join(caller_dir, file_name + ".log"),  \
                   level = "DEBUG", diagnose = True, format = default_format)
    
    # Set the console printing.    
    if print_opt == 0:
        logger.add(sys.stderr, level = "ERROR", colorize = True, \
                   diagnose = False, format = default_format)
    elif print_opt == 1:
        logger.add(sys.stderr, level = "INFO",  colorize = True, \
                   diagnose = False, format = "{message}")
    elif print_opt == 2:
        logger.add(sys.stderr, level = "INFO",  colorize = True, \
                   diagnose = False, format = default_format)
    
    # For every time we start to log, print the demarcation.
    # (If show_demarcation == True).
    if show_demarcation == True:
        logger.opt(raw = True).info("#" * 120 + "\n")
    
    # Call the function with logger.catch decorator.
    everything_you_want_to_see = call_function(function)
    
    # Remove all logger.
    if remove_logger == True:
        logger.remove()
    
    return everything_you_want_to_see