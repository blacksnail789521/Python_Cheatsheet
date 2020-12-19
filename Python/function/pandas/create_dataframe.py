import pandas as pd


# dict of list.
print("------------------")
df = pd.DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]})
print(df)

# 2d_array and factor_name_list.
print("------------------")
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns = ["A", "B", "C"])
print(df)

# list of dict.
print("------------------")
df = pd.DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 4, "B": 5, "C": 6}, {"A": 7, "B": 8, "C": 9}])
print(df)