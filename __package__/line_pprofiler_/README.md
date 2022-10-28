[功能介紹]
只需要把package這個資料夾copy到要執行的檔案的路徑，
而在"要做profile的程式碼"做一些相對應的修改(請參照caller_example來修改，詳細步驟請參考[使用說明])。

[使用說明]
1. 複製 if __name__ == "__main__": 以下的全部程式碼。
(如果原本就有if __name__ == "__main__": ，那就把原本的程式碼包在main()裡面。
如果想要在spyder的variable explorer看參數，直接在你新定義的main()最下面回傳就可以了，
所有回傳的值都會存到everything_you_want_to_see。)
2. 在程式最一開始輸入這行
exec("""import os\ntry:\n    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "package", "line_profiler", "import.txt")) as f: exec(f.read())\nexcept:pass""")

[參數設定]
1. enable用來控制是否要使用line profiler。
2. file_name預設是caller的名稱，想要其他的就自己設定就可以了。
3. blacklist是用來設定不想看到的package name。(只要該package的路徑包含這個名字，其所有的line profiler的結果都不會顯示出來。)
4. default_blacklist預設是"site-packages", "Anaconda"。
5. whitelist剛好跟blacklist相反，是用來設定一定要看到的package。(就算在default_blacklist裡面也一樣看得到。)
6. max_line_number跟max_ranking_number用來設定最終output顯示的行數。
7. output_ranking用來控制是否要output txt。

[FAQs]
1. 為什麼不用python內建的profiler?
因為python的profiler是function based的，我們要的是line based的。


2. 為什麼不用rkern的line_profiler? (link: https://github.com/rkern/line_profiler)
引用pprofile的作者的話 (link: https://github.com/vpelletier/pprofile)
<i>   It requires source code modification to select what should be profiled. I prefer to have the option to do an in-depth, non-intrusive profiling.
<ii>  It is not pure-python. This choice makes sense for performance but makes usage with pypy difficult and requires installation (I value execution straight from checkout).
<iii> As an effect of previous point, it does not have a notion above individual callable, annotating functions but not whole files - preventing module import profiling.
<iv>  Profiling recursive code provides unexpected results (recursion cost is accumulated on callable's first line) because it doesn't track call stack. This may be unintended, and may be fixed at some point in line_profiler.