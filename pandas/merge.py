import pandas as pd


# Index won't effect the result.
df1 = pd.DataFrame({"key_1": [1, 2, 3], "key_2": [4, 5, 6], "value": ["half_match", "full_match", "no_match"]}, \
                   index = pd.Index([0, 1, 2], name = "index"))
df2 = pd.DataFrame({"key_1": [1, 2, 7], "key_2": [8, 5, 9], "value": ["half_match", "full_match", "no_match"]}, \
                   index = pd.Index([3, 4, 5], name = "index"))
print(df1)
print(df2)

join_on_list = ["key_1", "key_2"]

# Use merge.
print("-------------------------------------------")
df = pd.merge(df1, df2, on = join_on_list, how = "outer", suffixes = ("_x", "_y"))
print(df)

# Use join.
print("-------------------------------------------")
df1.set_index(join_on_list, inplace = True)
df2.set_index(join_on_list, inplace = True)
df = df1.join(df2, how = "outer", lsuffix = "_x", rsuffix = "_y").reset_index()
print(df)