import warnings
import sys
import time

from log.log_to_file import log_to_file

def test_program(number):
    
    print(number)
    time.sleep(1) # To prove that the writing is in real-time.
    warnings.warn(r"I'm Warning!")
    time.sleep(1) # To prove that the writing is in real-time.
    1/0


def main():
    
    number = 1
    test_program(number)


if __name__ == "__main__":
    
    # Some parameters.
    need_to_log = True
    print_on_console = True
    show_warnings = True
    
    # Don't change anything here!
    if show_warnings == False:
        warnings.simplefilter("ignore")
    
    if sys.argv[-1] == "log":
        main()
    else:
        if need_to_log == True:
            log_to_file([sys.executable, *sys.argv], print_on_console)
        else:
            main()