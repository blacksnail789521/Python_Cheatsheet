import pandas as pd


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"df.drop" only supports dropping by indices' labels or columns' names.
If you need to drop by complicated rules, use de-select instead.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print("---------------")
df = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["A", "B", "C", "D"]}, \
                  index = pd.Index([-1, -2, -3, -4], name = "index"))
print(df)

# Drop by rows.
print("---------------")
drop_index_label_list = [-2, -5]
df_drop_by_rows = df.drop(drop_index_label_list, axis = 0, errors = "ignore")
print(f"df_drop_by_rows: \n{df_drop_by_rows}")

# Drop by cols.
print("---------------")
drop_column_name_list = ["A", "C"]
df_drop_by_cols = df.drop(drop_column_name_list, axis = 1, errors = "ignore")
print(f"df_drop_by_cols: \n{df_drop_by_cols}")