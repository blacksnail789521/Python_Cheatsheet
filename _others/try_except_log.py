import os
from datetime import datetime
import traceback


exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
log_path = os.path.join(exec_path, "(logging)_try_except_log.log")


def write_log(message, level = "INFO"):
    
    with open(log_path, "a") as log_file:
        
        datetime_message = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level_message = "INFO " if level == "INFO" else "ERROR"
        message = message if level == "INFO" else "An error has been caught!\n" + message
        log_message = f"{datetime_message} | {level_message} | {message}\n"
        log_file.write(log_message)
        print(message)


def delete_log_file():
    
    if os.path.exists(log_path):
        os.remove(log_path)


def main():
    
    delete_log_file()
    
    write_log("Start!")
    1/0


if __name__ == "__main__":
    
    try:
        main()
    except:
        write_log(traceback.format_exc(), level = "ERROR")