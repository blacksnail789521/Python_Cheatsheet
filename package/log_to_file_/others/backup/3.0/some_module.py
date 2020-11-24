exec("""import os\ntry:\n    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "package", "log_to_file", "import.txt")) as f: exec(f.read())\nexcept:pass""")

import time
from tqdm import tqdm, trange
import colorama
colorama.deinit()


def test_function(number):
    
    for i in trange(number):
        print(i)
    time.sleep(1) # To prove that the writing is in real-time.
    1/0