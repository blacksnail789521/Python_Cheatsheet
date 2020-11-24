import sys
from os.path import join, abspath, dirname, basename

package_paths = [join(abspath(dirname(__file__)))]
for package_path in package_paths: 
    if package_path not in sys.path: sys.path.insert(0, package_path)

from loguru import logger

def print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "")
    logger.opt(depth = 1).info \
        ( sep.join([ str(element) for element in list(args) ]) + end )

# log_opt = 0: Don't log on file, 1: Please log on file.
# print_opt = 0: Don't print on console (except error), 
#             1: Print without logging style, 
#             2:  Print with logging style.
# format_length: set the maximum length for each attribute.
def log_to_file(file_name = basename(abspath(__file__)).replace(".py", ""), \
                log_opt = 1, \
                print_opt = 1, \
                format_length = {"name": 10, "function": 20, "line": 4}, \
                show_demarcation = True):
    
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
        parent_dir = abspath(join(dirname(abspath(__file__)), ".."))
        logger.add(join(parent_dir, file_name + ".log"),  \
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