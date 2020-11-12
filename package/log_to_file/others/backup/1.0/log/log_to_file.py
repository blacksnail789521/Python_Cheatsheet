import os
import subprocess
import datetime

def log_to_file(argv, print_on_console):
    
    # Create subprocess (the one we want to log).
    os.environ['PYTHONUNBUFFERED'] = "1"
    argv.append("log") # To avoid the deadlock.
    proc = subprocess.Popen(argv,
                            stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE,
                            universal_newlines = True)
    
    log_file = open(os.path.join("log", "log.log"), "a")
    
    # When the stdout or stderr has something, we print it and write it into log at the same time.
    while proc.poll() is None:
        line = proc.stdout.readline()
        if line != "":
            if print_on_console == True:
                print(line, end = "")
            log_info = {"level": "     ", \
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), \
                        "line": line}
            log_file.write(log_info["time"] + " | " + log_info["level"] + " | " + log_info["line"])
            log_file.flush() # Write file in real-time.
 
        line = proc.stderr.readline()
        if line != "":
            if print_on_console == True:
                print(line, end = "")
            log_info = {"level": "ERROR", \
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), \
                        "line": line}
            log_file.write(log_info["time"] + " | " + log_info["level"] + " | " + log_info["line"])
            log_file.flush() # Write file in real-time.
    
    log_file.write("==================================================================================\n")
    log_file.close()