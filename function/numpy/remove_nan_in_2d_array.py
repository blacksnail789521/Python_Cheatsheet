import numpy as np


some_array = np.array([[0, 1], [2, np.nan], [4, 5]])
print(some_array)

# Drop by row.
some_array_drop_by_row = some_array[ ~ np.isnan(some_array).any(axis = 1) ]
print(some_array_drop_by_row)

# Drop by col.
some_array_drop_by_col = some_array[ : , ~ np.isnan(some_array).any(axis = 0) ]
print(some_array_drop_by_col)