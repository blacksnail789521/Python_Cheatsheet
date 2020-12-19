from timeit import timeit 


def check_sequence(sequence_x, sequence_y):
            
    legal_for_sequence = True
    
    # Merge two sequences into one sequence.
    merged_sequence = [ list(cell) for cell in zip( list(sequence_x), list(sequence_y) ) ]
    
    # We can't have any cell that contains more than two values.
    for cell in pd.isnull(merged_sequence):
        if list(cell).count(False) == 2:
            legal_for_sequence = False
            break
    
    if legal_for_sequence == False:
        merged_sequence = None
    else:
        # For each cell in merged_sequence, we keep the one that is not nan.
        for cell_index, cell in enumerate(merged_sequence):
            if pd.isnull(cell[0]) == True:
                merged_sequence[cell_index] = merged_sequence[cell_index][1]
            else:
                merged_sequence[cell_index] = merged_sequence[cell_index][0]
                
    return legal_for_sequence, merged_sequence



SETUP_CODE = \
"""
import pandas as pd
import numpy as np
from __main__ import check_sequence

df = pd.DataFrame({"A": [0, 1, np.nan, np.nan], \
                   "B": [np.nan, np.nan, 2, 3]})
a = df["A"]
b = df["B"]
"""

TEST_CODE = \
"""
legal_for_sequence, merged_sequence = check_sequence(a, b)
"""

print(timeit(setup = SETUP_CODE, stmt = TEST_CODE, number = 10000)) 