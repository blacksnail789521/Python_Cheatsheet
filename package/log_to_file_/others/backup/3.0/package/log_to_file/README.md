[功能介紹]
只需要把package這個資料夾copy到要執行的檔案的路徑，
而在"要寫log的程式碼"做一些相對應的修改(請參照caller_example來修改，詳細步驟請參考[使用說明])，
這樣就可以達到print的"同時"也可以寫log(即時的)。

[使用說明]
1. 複製 if __name__ == "__main__": 以下的全部程式碼。
(如果原本就有if __name__ == "__main__": ，那就把原本的程式碼包在main()裡面。
如果想要在spyder的variable explorer看參數，直接在你新定義的main()最下面回傳就可以了，
所有回傳的值都會存到everything_you_want_to_see。)
2. 在程式最一開始輸入這行
exec("""import os\ntry:\n    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "package", "log_to_file", "import.txt")) as f: exec(f.read())\nexcept:pass""")
(切記!!!所有相關的子檔案都要記得寫這行，或者是單純寫 from log_to_file import print 也可以，這樣所有子程式的print才可以都log進去)

[參數設定]
1. 想要寫log，就把log_opt設成1，想關掉寫log就設成0。
2. 如果只想要單純寫log(不想print)，就把print_opt設成0。想要一般的print，設成1。想要有logging風格的print，設成2。
3. format_length是拿來對齊log用的，覺得預設的太長或太短就改成自己喜歡的就可以了。
4. file_name預設是caller的名稱，想要其他的就自己設定就可以了。

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

[已知問題]
使用tqdm會造成output不是彩色的。
解法: import tqdm以後加入以下程式碼:
import colorama
colorama.deinit()