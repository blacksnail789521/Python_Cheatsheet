import difflib
import os
from datetime import datetime
import ctypes


def get_file_modified_time(file_path):
    
    return datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def popup_message(text, title = "", style = 0):
    
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)


def compare_two_files(fromfile = "old.py", tofile = "new.py"):
    
    text1 = open(fromfile).readlines()
    text2 = open(tofile).readlines()
    
    diff_line_list = list( difflib.unified_diff(text1, text2, fromfile = fromfile, tofile = tofile, \
                                                fromfiledate = get_file_modified_time(fromfile), \
                                                tofiledate = get_file_modified_time(tofile)) )
    
    # We don't have any difference.
    if len(diff_line_list) == 0:
        print("No difference!")
        popup_message("No difference!")
        return
    
    # We do have some differences.
    with open("diff.txt", "w") as diff_file:
        
        for diff_line in diff_line_list:
            print(diff_line)
            diff_file.write(diff_line)


if __name__ == "__main__":
    
    fromfile = "old.py"
    tofile = "new.py"
    compare_two_files(fromfile, tofile)