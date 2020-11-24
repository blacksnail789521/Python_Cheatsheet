[功能介紹]
只需要把log這個資料夾copy到要執行的檔案的路徑，
而要寫log的程式碼作揖些相對應的修改(請參照test_program來修改，記得need_to_log射程True)，
這樣就可以達到print的"同時"也可以寫log。

[使用說明]
1. 複製 if __name__ == "__main__": 以下的全部程式碼。
(如果原本就有if __name__ == "__main__": ，那就把原本的程式碼包在main()裡面。
如果想要在spyder的variable explorer看參數，直接在main()最下面回傳就可以了，
所有回傳的值都會存到everything_you_want_to_see。)
2. 在程式一開始import log的包 (from log.log_to_file import log_to_file)，
以及import warnings, sys。
3. 想要寫log，就把need_to_log設成True
4. 如果只想要單純寫log(不想print)，把print_on_console設定成False即可。
5. 不想把warning也寫到log，把show_warnings設定成False即可。

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

3. 為什麼不用decorator?
這應該很明顯吧，decorator是"額外"執行，而不是"替換"原本的執行功能。