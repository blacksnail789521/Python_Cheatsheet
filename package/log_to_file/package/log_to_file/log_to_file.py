import sys
from os.path import join, abspath, dirname, basename
import inspect
from DBSrv5ConnPy.db_object import DBObject

package_path = join(dirname(abspath(__file__)))
if package_path not in sys.path: sys.path.insert(0, package_path)

from loguru import logger



def print(*args, **kwargs):
    
    # If we don't sep, we set sep to " " by default.
    sep = kwargs.get("sep", " ")
    
    # If we don't end, we set end to "" by default.
    end = kwargs.get("end", "")
    
    # This line will let print to log to every added handler.
    logger.opt(depth = 1).info \
        ( sep.join([ str(element) for element in list(args) ]) + end )


def insert_to_DB(full_message, db_obj, table_name, ap_name):
    
    # Split full_message into multiple single-line message.
    message_list = str(full_message).split("\n")
    
    # Get the sql for insert all messages.
    sql = \
        """
        INSERT ALL
        """
    sql_with_single_row = \
        """
            INTO {table_name} (AP_NAME, REMARK) VALUES ('{AP_NAME}', q'{{{REMARK}}}')
        """
    for index, message in enumerate(message_list):
        sql = sql + sql_with_single_row.format(table_name = table_name, \
                                               AP_NAME = ap_name + "_" + "{:03d}".format(index + 1), \
                                               REMARK = message)
    sql = sql + """SELECT 1 FROM DUAL"""
    
    # Insert the table.
    db_obj.non_query(sql)


@logger.catch
def call_function(function):
    return function()


def log_to_file(function, file_name = None, FILE_OPTION = True, \
                PRINT_OPTION = True, print_with_logging_style = False, \
                format_length = {"name": 25, "function": 20, "line": 4}, \
                show_demarcation = True, remove_logger = True, \
                DB_OPTION = False, db_config = None):
    
    """
    function : |function|.
        The function you want to use logging.
    
    file_name : |str|, optional, default to "None".
        The .log file's name. If you didn't assign one, it would automatically 
        assign the caller's file_name and add "(logging)_" at the beginning.
    
    FILE_OPTION : |bool|, optional, default to "True".
        Determine whether or not to log on file.
    
    PRINT_OPTION : |bool|, optional, default to "True".
        Determine whether or not to print on the console. Notice that it will 
        print the errors on the console even you set PRINT_OPTION to False.
    
    print_with_logging_style: |bool|, optional, default to "False".
        Only matter when PRINT_OPTION is set to True.
        Determine whether or not to print on the console with the logging style.
    
    format_length: |dict|, optional, default to "{"name": 25, "function": 20, 
                                                  "line": 4}".
        Set the space_padding of the logging style. "name" is for the file's
        name, "function" is for the function's name, and "line" is for the line
        number.
    
    show_demarcation : |bool|, optional, default to "True".
        Determine whether or not to log a demarcation at the beginning. The 
        purpose of this is to seperate the logging information for different 
        execution.
    
    remove_logger : |bool|, optional, default to "True".
        Determine whether or not to remove all the handlers in the logger after
        execution. The purpose of this is to prevent other execution to log on 
        the same logger's handlers when you're using the same kernel.
    
    DB_OPTION : |bool|, optional, default to "False".
        Determine whether or not to log on DB.
    
    db_config: |dict|, optional, default to "None".
        Only matter when DB_OPTION is set to True.
        Set the "level", "fab", "db", "table_name", "ap_name". We recommend to 
        set level to "ERROR", but if you want to log everything, feel free to 
        set level to "INFO". For the ap_name, if you didn't assign one, it 
        would automatically assign the caller's file_name to it. 
    """
    
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
                     
    # Set the file's logging.
    if FILE_OPTION == True:
        caller_path = abspath(inspect.stack()[1].filename)
        caller_dir = dirname(caller_path)
        if file_name is None:
            file_name = basename(caller_path).replace(".py", "")
            file_name = "(logging)_" + file_name
        logger.add(join(caller_dir, file_name + ".log"),  \
                   level = "DEBUG", diagnose = True, format = default_format)
    
    # Set the console's printing.    
    if PRINT_OPTION == False:
        logger.add(sys.stderr, level = "ERROR", colorize = True, \
                   diagnose = False, format = default_format)
    elif PRINT_OPTION == True:
        if print_with_logging_style == False:
            logger.add(sys.stderr, level = "INFO",  colorize = True, \
                       diagnose = False, format = "{message}")
        elif print_with_logging_style == True:
            logger.add(sys.stderr, level = "INFO",  colorize = True, \
                       diagnose = False, format = default_format)
    
    # Set the DB's logging.
    if DB_OPTION == True:
        db_obj = DBObject(fab = db_config["fab"], db = db_config["db"])
        assert db_config["level"] in ["INFO", "ERROR"]
        ap_name = db_config.get("ap_name", None)
        if ap_name is None:
            ap_name = basename(caller_path).replace(".py", "") \
                      .replace("_caller_example", "")
        logger.add(lambda full_message : insert_to_DB(full_message, db_obj, \
                                                      db_config["table_name"], \
                                                      ap_name), \
                   level = db_config["level"], format = default_format)
    
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