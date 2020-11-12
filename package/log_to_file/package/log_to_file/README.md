[Features]
只需要把package這個資料夾copy到要執行的檔案的路徑，
而在"要寫log的程式碼"做一些相對應的修改(請參照caller_example來修改，詳細步驟請參考[使用說明])，
這樣就可以達到print的"同時"也可以寫log(即時的)。


[Instructions for use]
1. 複製 if __name__ == "__main__": 以下的全部程式碼。
(如果原本就有if __name__ == "__main__": ，那就把原本的程式碼包在main()裡面。
如果想要在spyder的variable explorer看參數，直接在你新定義的main()最下面回傳就可以了，
所有回傳的值都會存到everything_you_want_to_see。)
2. 在程式最一開始輸入這行
exec("""import os\ntry:\n    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "package", "log_to_file", "import.txt")) as f: exec(f.read())\nexcept:pass""")
(切記!!!所有相關的子檔案都要記得寫這行，或者是單純寫 from log_to_file import print 也可以，這樣所有子程式的print才可以都log進去)


[Parameters]
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
    Set the "level", "fab", "db", "ap_name", "table_name". We recommend to set level 
    to "ERROR", but if you want to log everything, feel free to set level 
    to "INFO".

    
[FAQs]
1. 為什麼不用python內建的logging?
因為兩個原因。
第一，如果我要同時print以及log到檔案，我要寫同樣的資訊兩次，這樣會造成程式碼重工。
第二，這樣會抓不到exception。(除非用try except)

2. 為什麼不用try-except?
因為兩個原因。
第一，開發者在開發程式的時候，每次都要把try-except註解掉，造成很多時間的浪費。
第二，try-except的用意在於抓到exception以後"可以不用停止script的執行"，
所以try-except應該是包在你認為有可能會發生錯誤的block外面，而不是把整個程式碼包起來。

[Known issues]
使用tqdm會造成output不是彩色的。
解法: import tqdm以後加入以下程式碼:
import colorama
colorama.deinit()