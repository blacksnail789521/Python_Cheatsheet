import time
import os

exec("""try:\n    with open(os.path.join("log", "log_to_file", "import.txt")) as f: exec(f.read())\nexcept:pass""")



def test_function(number):
    
    for i in range(number):
        print(i)
    time.sleep(1) # To prove that the writing is in real-time.
    1/0