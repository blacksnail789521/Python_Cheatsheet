import os
import sys
import time

use_tqdm = True

# Determine that we're using tqdm or not.
print_statement = "enumerate"
if use_tqdm == False:
    print(print_statement)
    for_statement = """enumerate(a)"""
else:
    for_statement = """enumerate(tqdm(a, desc = print_statement))"""
    
    # Import tqdm.
    package_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "package")
    if package_path not in sys.path: sys.path.insert(0, package_path)
    from tqdm import tqdm
    import colorama
    colorama.deinit()



a = [10, 11, 12, 13, 14]
for index, element in eval(for_statement):
#    print(index, element)
    time.sleep(1)