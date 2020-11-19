import pandas as pd
import numpy as np


df = pd.DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]}, \
                  index = pd.Index([-1, -2, -3], name = "index"))
print(df)

# Modify some cols.
print("---------------")
column_name_list = ["A", "C"]
df[column_name_list] = df[column_name_list].apply(lambda x : x + 1)
print(df)

# Modify some rows.
print("---------------")
index_label_list = [-3, -1]
df.loc[index_label_list] = df.loc[index_label_list].apply(lambda x : x + 1)
print(df)

# Get infos for each col.
print("---------------")
print("Sum for each col:")
sum_by_cols = df.apply(np.sum, axis = 0)
print(sum_by_cols)

# Get infos for each row.
print("---------------")
print("Sum for each row:")
sum_by_rows = df.apply(np.sum, axis = 1)
print(sum_by_rows)