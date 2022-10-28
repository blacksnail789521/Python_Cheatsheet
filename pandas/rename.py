import pandas as pd


print("----------------------")
df = pd.DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]}, \
                  index = pd.Index(["a", "b", "c"]))
print(df)

# Rename column_name.
print("----------------------")
rename_column_name_dict = {}
for index, column_name in enumerate(df.columns):
    rename_column_name_dict[column_name] = column_name + str(index + 1)
df_rename_by_column_name = df.rename(columns = rename_column_name_dict)
print(df_rename_by_column_name)

# Rename index_label.
print("----------------------")
rename_index_label_dict = {}
for index, index_label in enumerate(df.index):
    rename_index_label_dict[index_label] = index_label + str(index + 1)
df_rename_by_index_label = df.rename(index = rename_index_label_dict)
print(df_rename_by_index_label)