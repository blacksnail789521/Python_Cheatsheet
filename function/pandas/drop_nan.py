import pandas as pd
import numpy as np


print("---------------")
df = pd.DataFrame({"A": [1, np.nan, 3, 4], "B": [5, 6, 7, np.nan], "C": [9, 10, 11, 12]})
print(df)

# Drop by rows.
print("---------------")
df_drop_by_rows = df.dropna(axis = 0, how = "any")
print(f"df_drop_by_rows: \n{df_drop_by_rows}")

# Drop by cols.
print("---------------")
df_drop_by_cols = df.dropna(axis = 1, how = "any")
print(f"df_drop_by_cols: \n{df_drop_by_cols}")