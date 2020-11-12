import os
import sys
import time

use_tqdm = True

# Determine that we're using tqdm or not.
print_statement = "range"
if use_tqdm == False:
    print(print_statement)
    for_statement = """range(3, len(a))"""
else:
    for_statement = """trange(3, len(a), desc = print_statement)"""
    
    # Import tqdm.
    package_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "package")
    if package_path not in sys.path: sys.path.insert(0, package_path)
    from tqdm import trange
    import colorama
    colorama.deinit()



a = [10, 11, 12, 13, 14]
for index in eval(for_statement):
#    print(index)
    time.sleep(1)