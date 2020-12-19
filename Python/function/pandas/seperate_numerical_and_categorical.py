import pandas as pd


df = pd.DataFrame({"numerical": [1, 2, 3, 4], \
                   "categorical": ["A", "B", "C", "D"]})

# Select df.
print("==========================================")
numerical_df   = df.select_dtypes(exclude = ["object"])
categorical_df = df.select_dtypes(include = ["object"])
print(f"numerical_df: \n{numerical_df}")
print("------------------")
print(f"categorical_df: \n{categorical_df}")

# Select column_name_list.
print("==========================================")
numerical_column_name_list   = list(df.dtypes[df.dtypes != "object"].index)
categorical_column_name_list = list(df.dtypes[df.dtypes == "object"].index)
print(f"numerical_column_name_list: {numerical_column_name_list}")
print(f"categorical_column_name_list: {categorical_column_name_list}")