import os
import sys
import time
import pandas as pd

use_tqdm = True

# Determine that we're using tqdm or not.
print_statement = "df.iterrows()"
if use_tqdm == False:
    print(print_statement)
    for_statement = """df.iterrows()"""
else:
    for_statement = """tqdm(df.iterrows(), total = df.shape[0], desc = print_statement)"""
    
    # Import tqdm.
    package_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "package")
    if package_path not in sys.path: sys.path.insert(0, package_path)
    from tqdm import tqdm
    import colorama
    colorama.deinit()



df = pd.DataFrame({"Brand": ["Honda Civic", "Toyota Corolla", "Ford Focus", "Audi A4"], \
                   "Price": [22000, 25000, 27000, 35000]})
#print(df)

for index, row in eval(for_statement):
#    print(row)
    time.sleep(1)